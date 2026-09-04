// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The routed-call server trait and the routing decision it carries.
//!
//! [`RoutedLlmClient`] is the one piece of I/O the protocol does not own: a host
//! implements it to actually perform a model call. [`Decision`] is the routing
//! decision that produced the call, carried alongside so the client and any
//! observer can see which model was chosen and why. Both live here — rather than
//! in libsy's orchestration crate — so a client crate that depends only on the
//! protocol can serve routed calls without pulling in the orchestrator.

use async_trait::async_trait;
use thiserror::Error;

use crate::{Request, Response};

/// A boxed client-specific error preserved as the source of a routed call failure.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Failures a routed LLM client can surface to its caller.
///
/// The variants classify failures that routing hosts commonly need to handle,
/// while boxed sources preserve implementation-specific detail. `General` is the
/// escape hatch for failures that do not fit a shared category.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum LlmClientError {
    /// The request cannot be served as supplied.
    #[error("invalid request: {message}")]
    InvalidRequest {
        /// Human-readable request validation failure.
        message: String,
    },

    /// Decoding the inbound request failed in the translation engine.
    #[error("request translation failed: {0}")]
    RequestTranslation(String),

    /// Encoding the request for the upstream failed in the translation engine.
    #[error("outbound request encoding failed: {0}")]
    RequestEncoding(String),

    /// Decoding or encoding the response failed in the translation engine.
    #[error("response translation failed: {0}")]
    ResponseTranslation(String),

    /// The client is not configured to serve the selected target.
    #[error("client configuration error: {message}")]
    Configuration {
        /// Human-readable configuration failure.
        message: String,
    },

    /// The upstream could not be reached or the request could not be sent.
    #[error("upstream transport error: {source}")]
    Transport {
        /// Client-specific transport failure.
        #[source]
        source: BoxError,
    },

    /// The upstream request exceeded its timeout.
    #[error("upstream request timed out: {source}")]
    Timeout {
        /// Client-specific timeout failure.
        #[source]
        source: BoxError,
    },

    /// The upstream rejected the request because it exceeds the model's context window.
    #[error("context window exceeded for model {model}: {message}")]
    ContextWindowExceeded {
        /// Model whose context window was exceeded.
        model: String,
        /// Upstream error message.
        message: String,
    },

    /// The upstream returned a non-success HTTP response.
    #[error("upstream returned HTTP {status}: {body}")]
    UpstreamHttp {
        /// Upstream HTTP status code.
        status: u16,
        /// Raw upstream error body.
        body: String,
    },

    /// The upstream returned a response the client could not decode.
    #[error("invalid upstream response: {source}")]
    InvalidResponse {
        /// Client-specific decoding or validation failure.
        #[source]
        source: BoxError,
    },

    /// A call across a foreign-function boundary (e.g. a Python-implemented client)
    /// failed. The boxed source is the foreign error itself.
    #[error("foreign function interface error: {source}")]
    Ffi {
        /// Foreign-language failure, preserved verbatim.
        #[source]
        source: BoxError,
    },

    /// A routing host classified the failure itself and instructed routing what to do.
    ///
    /// Prefer this over the transport- and status-shaped variants above: it carries an
    /// explicit [`RoutingDisposition`], so routing never re-derives fallback policy from
    /// an HTTP status.
    #[error("routed call failed: {failure}")]
    RoutedCall {
        /// Provider-neutral classification and routing instruction.
        failure: RoutedCallFailure,
    },

    /// A string message. Useful in testing, but prefer adding variants over using this.
    #[error("{0}")]
    General(String),
}

/// Largest retry advice, in milliseconds, a routed-call failure may carry.
///
/// Advice above this bound is not accepted: a routing host must bound its own
/// provider hint before it becomes public contract.
pub const MAX_ROUTED_RETRY_AFTER_MS: u64 = 300_000;

/// Largest number of provider candidates one exhaustion summary may account for.
pub const MAX_PROVIDER_TARGETS: u8 = 16;

/// Whether routing may replace the failed target with another candidate.
///
/// This is the routing host's explicit instruction. Routing consumes it directly and
/// never re-derives fallback policy from a transport or HTTP status.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RoutingDisposition {
    /// The failure is terminal for this route; do not try another target.
    Stop,
    /// Another eligible target may serve the request.
    NextTarget,
}

impl RoutingDisposition {
    /// Stable value embedded in routing reasoning and telemetry.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::NextTarget => "next_target",
        }
    }
}

impl std::fmt::Display for RoutingDisposition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Provider-neutral class of a routed-call failure.
///
/// The class names *what kind* of failure occurred, never which provider produced it.
/// It carries no provider body, message, target name, or health evidence: a routing
/// host keeps that private. New classes may be added, so match non-exhaustively.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RoutedCallFailureClass {
    /// The target was skipped because its availability circuit was open.
    CircuitOpen,
    /// Every provider candidate for the selected model was attempted or bypassed.
    ProviderTargetsExhausted,
    /// The target cannot serve this request's capabilities or profile.
    TargetIncompatible,
    /// The request exceeds the target's context window.
    ContextWindow,
    /// A rate, spend, or quota limit was enforced.
    RateLimit,
    /// The provider or model timed out processing the request.
    ProviderTimeout,
    /// The routing host's own per-attempt timeout elapsed.
    AttemptTimeout,
    /// The provider identified itself as overloaded or out of capacity.
    Overloaded,
    /// The call failed before a provider response head arrived.
    Transport,
    /// The provider returned a failure that identifies no more specific class.
    Upstream,
    /// The provider response could not be decoded or validated.
    InvalidResponse,
    /// The provider rejected the request on its own terms.
    ProviderRejected,
    /// Host policy denied the call.
    PolicyDenied,
    /// The credential for the target could not be obtained or used.
    CredentialUnavailable,
    /// The target's configuration is invalid.
    Configuration,
    /// The call's work or attempt budget is spent.
    WorkBudget,
    /// The caller cancelled, or the route deadline elapsed.
    Cancelled,
    /// The failure could not be classified.
    Unknown,
}

impl RoutedCallFailureClass {
    /// Stable tag for reasoning, telemetry, and aggregate ordering.
    ///
    /// These values are contract: they are immutable once published, so a host may
    /// order and label by them across releases.
    pub const fn stable_tag(self) -> &'static str {
        match self {
            Self::CircuitOpen => "circuit_open",
            Self::ProviderTargetsExhausted => "provider_targets_exhausted",
            Self::TargetIncompatible => "target_incompatible",
            Self::ContextWindow => "context_window",
            Self::RateLimit => "rate_limit",
            Self::ProviderTimeout => "provider_timeout",
            Self::AttemptTimeout => "attempt_timeout",
            Self::Overloaded => "overloaded",
            Self::Transport => "transport",
            Self::Upstream => "upstream",
            Self::InvalidResponse => "invalid_response",
            Self::ProviderRejected => "provider_rejected",
            Self::PolicyDenied => "policy_denied",
            Self::CredentialUnavailable => "credential_unavailable",
            Self::Configuration => "configuration",
            Self::WorkBudget => "work_budget",
            Self::Cancelled => "cancelled",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for RoutedCallFailureClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.stable_tag())
    }
}

/// A routed-call contract invariant a caller tried to violate.
///
/// A routing host maps this to its own terminal routing-invariant error rather than
/// guessing a status: an unenforceable summary is a host bug, not a provider outcome.
#[derive(Clone, Copy, Debug, Eq, Error, Hash, PartialEq)]
#[error("routed-call contract violated: {reason}")]
pub struct RoutedFailureInvariant {
    reason: &'static str,
}

impl RoutedFailureInvariant {
    /// Which invariant was violated.
    pub const fn reason(self) -> &'static str {
        self.reason
    }
}

/// How many candidates failed with one class.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RoutedFailureCount {
    /// The failure class these candidates shared.
    pub class: RoutedCallFailureClass,
    /// How many candidates failed this way. Never zero in a valid summary.
    pub count: u8,
}

impl RoutedFailureCount {
    /// Builds one class entry.
    pub const fn new(class: RoutedCallFailureClass, count: u8) -> Self {
        Self { class, count }
    }
}

/// Bounded evidence that every provider candidate for one model was spent.
///
/// The summary is a partition: each attempted candidate and each candidate bypassed by
/// an open circuit is counted exactly once, under exactly one class. It names no
/// target, carries no provider text, and never nests another exhaustion.
///
/// [`Self::new`] is the only constructor, so a value of this type always satisfies
/// every invariant in the contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderTargetsExhaustedSummary {
    attempted: u8,
    bypassed: u8,
    failures: Vec<RoutedFailureCount>,
    retry_after_ms: Option<u64>,
}

impl ProviderTargetsExhaustedSummary {
    /// Builds a validated summary, ordering entries by immutable
    /// [`stable_tag`](RoutedCallFailureClass::stable_tag) rather than declaration order.
    ///
    /// # Errors
    ///
    /// Returns [`RoutedFailureInvariant`] unless every invariant holds:
    /// `attempted + bypassed` is in `1..=`[`MAX_PROVIDER_TARGETS`]; each class appears
    /// once with a nonzero count; no entry is
    /// [`ProviderTargetsExhausted`](RoutedCallFailureClass::ProviderTargetsExhausted),
    /// so exhaustion never recurses; the counts sum to `attempted + bypassed`;
    /// the [`CircuitOpen`](RoutedCallFailureClass::CircuitOpen) count equals `bypassed`
    /// exactly, leaving every other count to sum to `attempted`; and any retry advice
    /// is within [`MAX_ROUTED_RETRY_AFTER_MS`].
    pub fn new(
        attempted: u8,
        bypassed: u8,
        failures: Vec<RoutedFailureCount>,
        retry_after_ms: Option<u64>,
    ) -> Result<Self, RoutedFailureInvariant> {
        let total = attempted
            .checked_add(bypassed)
            .ok_or(invariant("candidate total overflows"))?;
        if total == 0 || total > MAX_PROVIDER_TARGETS {
            return Err(invariant("candidate total is outside 1..=16"));
        }

        // One pass proves the partition: unique nonzero classes, no nested exhaustion,
        // and the two sums that pin bypasses to CircuitOpen and the rest to attempts.
        let mut counted: u32 = 0;
        let mut circuit_open: u32 = 0;
        let mut seen: Vec<RoutedCallFailureClass> = Vec::with_capacity(failures.len());
        for entry in &failures {
            if entry.count == 0 {
                return Err(invariant("class entry has a zero count"));
            }
            if entry.class == RoutedCallFailureClass::ProviderTargetsExhausted {
                return Err(invariant("exhaustion cannot nest inside a summary"));
            }
            if seen.contains(&entry.class) {
                return Err(invariant("class entry is repeated"));
            }
            seen.push(entry.class);
            counted += u32::from(entry.count);
            if entry.class == RoutedCallFailureClass::CircuitOpen {
                circuit_open = u32::from(entry.count);
            }
        }
        if counted != u32::from(total) {
            return Err(invariant("class counts do not sum to the candidate total"));
        }
        if circuit_open != u32::from(bypassed) {
            return Err(invariant("CircuitOpen count does not equal bypassed"));
        }
        if retry_after_ms.is_some_and(|ms| ms > MAX_ROUTED_RETRY_AFTER_MS) {
            return Err(invariant("retry advice exceeds the 300000ms cap"));
        }

        let mut failures = failures;
        failures.sort_by_key(|entry| entry.class.stable_tag());
        Ok(Self {
            attempted,
            bypassed,
            failures,
            retry_after_ms,
        })
    }

    /// Candidates that reached a real provider attempt.
    pub const fn attempted(&self) -> u8 {
        self.attempted
    }

    /// Candidates skipped before contact because their circuit was open.
    pub const fn bypassed(&self) -> u8 {
        self.bypassed
    }

    /// Per-class counts, ordered by stable tag.
    pub fn failures(&self) -> &[RoutedFailureCount] {
        &self.failures
    }

    /// Aggregate retry advice, when every exhausted candidate supplied a truthful hint.
    pub const fn retry_after_ms(&self) -> Option<u64> {
        self.retry_after_ms
    }
}

/// A routing host's typed, provider-neutral call failure.
///
/// It carries the host's explicit [`RoutingDisposition`] so routing never re-derives
/// fallback policy from an HTTP status, plus a bounded class, an optional bounded
/// provider status, and optional bounded retry advice. It deliberately holds no
/// provider body, message, target name, or availability evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutedCallFailure {
    class: RoutedCallFailureClass,
    disposition: RoutingDisposition,
    retry_after_ms: Option<u64>,
    provider_status: Option<u16>,
    exhausted: Option<ProviderTargetsExhaustedSummary>,
}

impl RoutedCallFailure {
    /// Builds a direct failure for any class other than
    /// [`ProviderTargetsExhausted`](RoutedCallFailureClass::ProviderTargetsExhausted),
    /// which has a fixed shape and is built by [`Self::targets_exhausted`].
    ///
    /// # Errors
    ///
    /// Returns [`RoutedFailureInvariant`] for the exhaustion class, or for retry advice
    /// above [`MAX_ROUTED_RETRY_AFTER_MS`].
    pub fn new(
        class: RoutedCallFailureClass,
        disposition: RoutingDisposition,
        retry_after_ms: Option<u64>,
        provider_status: Option<u16>,
    ) -> Result<Self, RoutedFailureInvariant> {
        if class == RoutedCallFailureClass::ProviderTargetsExhausted {
            return Err(invariant("exhaustion requires a bounded summary"));
        }
        if retry_after_ms.is_some_and(|ms| ms > MAX_ROUTED_RETRY_AFTER_MS) {
            return Err(invariant("retry advice exceeds the 300000ms cap"));
        }
        Ok(Self {
            class,
            disposition,
            retry_after_ms,
            provider_status,
            exhausted: None,
        })
    }

    /// Builds the provider-exhaustion failure from a validated summary.
    ///
    /// The shape is fixed by the contract, so this cannot fail: the class is
    /// [`ProviderTargetsExhausted`](RoutedCallFailureClass::ProviderTargetsExhausted),
    /// the disposition is [`NextTarget`](RoutingDisposition::NextTarget) because another
    /// logical model may still serve the request, there is no provider status, and the
    /// retry advice is the summary's.
    pub fn targets_exhausted(summary: ProviderTargetsExhaustedSummary) -> Self {
        Self {
            class: RoutedCallFailureClass::ProviderTargetsExhausted,
            disposition: RoutingDisposition::NextTarget,
            retry_after_ms: summary.retry_after_ms(),
            provider_status: None,
            exhausted: Some(summary),
        }
    }

    /// What kind of failure this is.
    pub const fn class(&self) -> RoutedCallFailureClass {
        self.class
    }

    /// Whether routing may try another target.
    pub const fn disposition(&self) -> RoutingDisposition {
        self.disposition
    }

    /// Bounded retry advice, when the host had a truthful one.
    pub const fn retry_after_ms(&self) -> Option<u64> {
        self.retry_after_ms
    }

    /// The provider's status code, when one identified the failure.
    pub const fn provider_status(&self) -> Option<u16> {
        self.provider_status
    }

    /// The bounded exhaustion summary, present only for the exhaustion class.
    pub const fn exhausted(&self) -> Option<&ProviderTargetsExhaustedSummary> {
        self.exhausted.as_ref()
    }
}

impl std::fmt::Display for RoutedCallFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.class.stable_tag(), self.disposition)?;
        if let Some(status) = self.provider_status {
            write!(f, ", provider status {status}")?;
        }
        if let Some(summary) = &self.exhausted {
            write!(
                f,
                ", {} attempted, {} bypassed",
                summary.attempted(),
                summary.bypassed()
            )?;
        }
        if let Some(retry_after_ms) = self.retry_after_ms {
            write!(f, ", retry after {retry_after_ms}ms")?;
        }
        Ok(())
    }
}

/// Builds a contract violation. Reasons are fixed strings, never caller or provider text.
const fn invariant(reason: &'static str) -> RoutedFailureInvariant {
    RoutedFailureInvariant { reason }
}

/// Why routing replaced a selected target with another eligible target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutingFallbackReason {
    /// The selected target rejected the request because its context window was too small.
    ContextWindow,
    /// The selected target was unavailable after its client retries finished.
    Unavailable,
}

impl RoutingFallbackReason {
    /// Stable value embedded in routing reasoning.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ContextWindow => "context_window",
            Self::Unavailable => "unavailable",
        }
    }
}

/// A routing choice produced by an algorithm.
#[derive(Clone, Debug)]
pub struct Decision {
    /// The model identifier selected for the call.
    selected_model_id: String,
    /// Why, for logs and traces.
    reasoning: Option<String>,
    /// True for an answer-generating call. False for classifier and judge calls.
    is_answer_call: bool,
}

impl Decision {
    /// Creates a decision and records whether its call produces the answer.
    pub fn new(
        selected_model_id: impl Into<String>,
        reasoning: Option<String>,
        is_answer_call: bool,
    ) -> Self {
        Self {
            selected_model_id: selected_model_id.into(),
            reasoning,
            is_answer_call,
        }
    }

    /// The model identifier selected for the call.
    pub fn selected_model_id(&self) -> &str {
        self.selected_model_id.as_str()
    }

    /// Why this decision was made.
    pub fn reasoning(&self) -> Option<&str> {
        self.reasoning.as_deref()
    }

    /// Whether this call generates an answer rather than a routing verdict.
    pub fn is_answer_call(&self) -> bool {
        self.is_answer_call
    }
}

/// Performs the actual model call for a target. This is the one piece of I/O the
/// library does not own — a host implements it over its own transport (HTTP SDK,
/// in-process model, mock). It serves a call the stream consumer chose not to
/// override, reached as a routed request's `default_client`.
///
/// # Concurrency
///
/// A client may be shared by many targets and concurrent algorithm runs. Calls may
/// overlap, so implementations must synchronize mutable state internally and should
/// not serialize requests unless their transport requires it.
#[async_trait]
pub trait RoutedLlmClient: Send + Sync {
    /// Serve the model identified by
    /// [`decision.selected_model_id()`](Decision::selected_model_id), resolving it to the
    /// provider model this client calls.
    /// `request.llm_request.model` is the agent's original name, carried through for
    /// reference, not a call target.
    async fn call(&self, request: Request, decision: Decision) -> Result<Response, LlmClientError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every class, so the tag table below cannot silently miss one.
    const ALL_CLASSES: [RoutedCallFailureClass; 18] = [
        RoutedCallFailureClass::CircuitOpen,
        RoutedCallFailureClass::ProviderTargetsExhausted,
        RoutedCallFailureClass::TargetIncompatible,
        RoutedCallFailureClass::ContextWindow,
        RoutedCallFailureClass::RateLimit,
        RoutedCallFailureClass::ProviderTimeout,
        RoutedCallFailureClass::AttemptTimeout,
        RoutedCallFailureClass::Overloaded,
        RoutedCallFailureClass::Transport,
        RoutedCallFailureClass::Upstream,
        RoutedCallFailureClass::InvalidResponse,
        RoutedCallFailureClass::ProviderRejected,
        RoutedCallFailureClass::PolicyDenied,
        RoutedCallFailureClass::CredentialUnavailable,
        RoutedCallFailureClass::Configuration,
        RoutedCallFailureClass::WorkBudget,
        RoutedCallFailureClass::Cancelled,
        RoutedCallFailureClass::Unknown,
    ];

    /// Stable tags are published contract: a host orders and labels by them, so changing
    /// one silently reorders aggregates and breaks telemetry continuity.
    #[test]
    fn stable_tags_are_immutable_and_unique() {
        let expected = [
            "circuit_open",
            "provider_targets_exhausted",
            "target_incompatible",
            "context_window",
            "rate_limit",
            "provider_timeout",
            "attempt_timeout",
            "overloaded",
            "transport",
            "upstream",
            "invalid_response",
            "provider_rejected",
            "policy_denied",
            "credential_unavailable",
            "configuration",
            "work_budget",
            "cancelled",
            "unknown",
        ];
        let tags: Vec<&str> = ALL_CLASSES.iter().map(|class| class.stable_tag()).collect();
        assert_eq!(tags, expected);

        let mut unique = tags.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), tags.len(), "stable tags must be unique");
    }

    #[test]
    fn dispositions_carry_stable_tags() {
        assert_eq!(RoutingDisposition::Stop.as_str(), "stop");
        assert_eq!(RoutingDisposition::NextTarget.as_str(), "next_target");
    }

    #[test]
    fn summary_accepts_a_complete_partition() {
        let summary = ProviderTargetsExhaustedSummary::new(
            3,
            1,
            vec![
                RoutedFailureCount::new(RoutedCallFailureClass::RateLimit, 2),
                RoutedFailureCount::new(RoutedCallFailureClass::CircuitOpen, 1),
                RoutedFailureCount::new(RoutedCallFailureClass::Overloaded, 1),
            ],
            Some(1_500),
        )
        .expect("valid partition");

        assert_eq!(summary.attempted(), 3);
        assert_eq!(summary.bypassed(), 1);
        assert_eq!(summary.retry_after_ms(), Some(1_500));
    }

    /// Entries order by immutable tag, never by declaration order: `CircuitOpen` is
    /// declared first but sorts after `attempt_timeout`.
    #[test]
    fn summary_orders_entries_by_stable_tag() {
        let summary = ProviderTargetsExhaustedSummary::new(
            2,
            1,
            vec![
                RoutedFailureCount::new(RoutedCallFailureClass::CircuitOpen, 1),
                RoutedFailureCount::new(RoutedCallFailureClass::Upstream, 1),
                RoutedFailureCount::new(RoutedCallFailureClass::AttemptTimeout, 1),
            ],
            None,
        )
        .expect("valid partition");

        let tags: Vec<&str> = summary
            .failures()
            .iter()
            .map(|entry| entry.class.stable_tag())
            .collect();
        assert_eq!(tags, ["attempt_timeout", "circuit_open", "upstream"]);
    }

    #[test]
    fn summary_rejects_every_broken_invariant() {
        // Empty candidate set: exhaustion needs at least one candidate.
        assert!(ProviderTargetsExhaustedSummary::new(0, 0, Vec::new(), None).is_err());

        // Above the 16-candidate bound.
        assert!(
            ProviderTargetsExhaustedSummary::new(
                17,
                0,
                vec![RoutedFailureCount::new(
                    RoutedCallFailureClass::RateLimit,
                    17
                )],
                None,
            )
            .is_err()
        );

        // Zero count.
        assert!(
            ProviderTargetsExhaustedSummary::new(
                1,
                0,
                vec![
                    RoutedFailureCount::new(RoutedCallFailureClass::RateLimit, 1),
                    RoutedFailureCount::new(RoutedCallFailureClass::Upstream, 0),
                ],
                None,
            )
            .is_err()
        );

        // Repeated class.
        assert!(
            ProviderTargetsExhaustedSummary::new(
                2,
                0,
                vec![
                    RoutedFailureCount::new(RoutedCallFailureClass::RateLimit, 1),
                    RoutedFailureCount::new(RoutedCallFailureClass::RateLimit, 1),
                ],
                None,
            )
            .is_err()
        );

        // Exhaustion never nests inside its own summary.
        assert!(
            ProviderTargetsExhaustedSummary::new(
                1,
                0,
                vec![RoutedFailureCount::new(
                    RoutedCallFailureClass::ProviderTargetsExhausted,
                    1,
                )],
                None,
            )
            .is_err()
        );

        // Counts do not sum to attempted + bypassed.
        assert!(
            ProviderTargetsExhaustedSummary::new(
                3,
                0,
                vec![RoutedFailureCount::new(
                    RoutedCallFailureClass::RateLimit,
                    2
                )],
                None,
            )
            .is_err()
        );

        // CircuitOpen count must equal bypassed exactly.
        assert!(
            ProviderTargetsExhaustedSummary::new(
                1,
                1,
                vec![
                    RoutedFailureCount::new(RoutedCallFailureClass::CircuitOpen, 2),
                    RoutedFailureCount::new(RoutedCallFailureClass::RateLimit, 1),
                ],
                None,
            )
            .is_err()
        );

        // A bypass counted as an attempt leaves the non-CircuitOpen sum wrong.
        assert!(
            ProviderTargetsExhaustedSummary::new(
                2,
                1,
                vec![
                    RoutedFailureCount::new(RoutedCallFailureClass::CircuitOpen, 1),
                    RoutedFailureCount::new(RoutedCallFailureClass::RateLimit, 1),
                ],
                None,
            )
            .is_err()
        );

        // Retry advice above the cap is not truthful public contract.
        assert!(
            ProviderTargetsExhaustedSummary::new(
                1,
                0,
                vec![RoutedFailureCount::new(
                    RoutedCallFailureClass::RateLimit,
                    1
                )],
                Some(MAX_ROUTED_RETRY_AFTER_MS + 1),
            )
            .is_err()
        );
    }

    #[test]
    fn summary_accepts_advice_at_the_cap() {
        let summary = ProviderTargetsExhaustedSummary::new(
            1,
            0,
            vec![RoutedFailureCount::new(
                RoutedCallFailureClass::RateLimit,
                1,
            )],
            Some(MAX_ROUTED_RETRY_AFTER_MS),
        )
        .expect("advice at the cap is in bounds");
        assert_eq!(summary.retry_after_ms(), Some(MAX_ROUTED_RETRY_AFTER_MS));
    }

    #[test]
    fn direct_failure_carries_disposition_and_no_summary() {
        let failure = RoutedCallFailure::new(
            RoutedCallFailureClass::RateLimit,
            RoutingDisposition::NextTarget,
            Some(2_000),
            Some(429),
        )
        .expect("valid direct failure");

        assert_eq!(failure.class(), RoutedCallFailureClass::RateLimit);
        assert_eq!(failure.disposition(), RoutingDisposition::NextTarget);
        assert_eq!(failure.retry_after_ms(), Some(2_000));
        assert_eq!(failure.provider_status(), Some(429));
        assert!(failure.exhausted().is_none());
    }

    #[test]
    fn direct_failure_rejects_exhaustion_class_and_over_cap_advice() {
        assert!(
            RoutedCallFailure::new(
                RoutedCallFailureClass::ProviderTargetsExhausted,
                RoutingDisposition::NextTarget,
                None,
                None,
            )
            .is_err()
        );
        assert!(
            RoutedCallFailure::new(
                RoutedCallFailureClass::RateLimit,
                RoutingDisposition::NextTarget,
                Some(MAX_ROUTED_RETRY_AFTER_MS + 1),
                None,
            )
            .is_err()
        );
    }

    /// The exhaustion failure's shape is fixed: `NextTarget`, no provider status, and
    /// retry advice that always equals the summary's.
    #[test]
    fn exhaustion_failure_has_the_contract_shape() {
        let summary = ProviderTargetsExhaustedSummary::new(
            2,
            0,
            vec![
                RoutedFailureCount::new(RoutedCallFailureClass::RateLimit, 1),
                RoutedFailureCount::new(RoutedCallFailureClass::Overloaded, 1),
            ],
            Some(4_000),
        )
        .expect("valid partition");
        let failure = RoutedCallFailure::targets_exhausted(summary.clone());

        assert_eq!(
            failure.class(),
            RoutedCallFailureClass::ProviderTargetsExhausted
        );
        assert_eq!(failure.disposition(), RoutingDisposition::NextTarget);
        assert_eq!(failure.provider_status(), None);
        assert_eq!(failure.retry_after_ms(), summary.retry_after_ms());
        assert_eq!(failure.exhausted(), Some(&summary));
    }

    /// The rendered failure is bounded and provider-neutral: tags and numbers only.
    #[test]
    fn display_is_bounded_and_provider_neutral() {
        let failure = RoutedCallFailure::new(
            RoutedCallFailureClass::Overloaded,
            RoutingDisposition::NextTarget,
            Some(1_000),
            Some(529),
        )
        .expect("valid direct failure");
        assert_eq!(
            failure.to_string(),
            "overloaded (next_target), provider status 529, retry after 1000ms"
        );

        let error = LlmClientError::RoutedCall { failure };
        assert_eq!(
            error.to_string(),
            "routed call failed: overloaded (next_target), provider status 529, retry after 1000ms"
        );
    }
}
