// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared LLM judge primitives.
//!
//! [`Judge`] owns algorithm-specific request construction and verdict parsing.
//! [`JudgeClassifier`] owns the judge model call and hands its verdict to a policy that chooses
//! the route.

use std::marker::PhantomData;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;
use switchyard_protocol::{
    AggLlmResponse, Category, InstructionBlock, LlmRequest, Message, ModelId, OutputParams, Role,
    completion_text,
};

use super::judge_diagnostics::{ReplyShape, diagnose_parse};
use super::robustness::{safe_client_error, safe_error_summary};

use super::classifier_contract::ClassifierContract;
use crate::core::algorithm::{CallAssessment, CallAssessor, Driver};
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::State;
use crate::{LibsyError, Result};
use switchyard_protocol::{LlmClientError, Request, Response};

/// Builds the classifier-specific message view presented to a structured judge.
pub(crate) trait ClassifierInput: Send + Sync {
    fn build_messages(&self, state: &State, request: &Request) -> Vec<Message>;
}

/// Converts one structured model response into the verdict type consumed by a policy.
pub(crate) trait VerdictDecoder: Send + Sync {
    type Verdict: DeserializeOwned + Send + Sync;

    fn decode(
        &self,
        response: &AggLlmResponse,
        contract: &ClassifierContract,
    ) -> Result<Self::Verdict>;
}

/// Deserializes a structured response directly into a typed verdict.
pub(crate) struct SerdeDecoder<V> {
    verdict: PhantomData<fn() -> V>,
}

impl<V> SerdeDecoder<V> {
    pub(crate) const fn new() -> Self {
        Self {
            verdict: PhantomData,
        }
    }
}

impl<V> VerdictDecoder for SerdeDecoder<V>
where
    V: DeserializeOwned + Send + Sync,
{
    type Verdict = V;

    fn decode(
        &self,
        response: &AggLlmResponse,
        contract: &ClassifierContract,
    ) -> Result<Self::Verdict> {
        if !contract.validates_locally() {
            return parse_json_verdict(response);
        }
        let verdict = parse_json_verdict::<Value>(response)?;
        contract.validate_verdict(&verdict)?;
        serde_json::from_value(verdict).map_err(|error| LibsyError::AlgorithmError {
            message: format!(
                "judge reply did not parse as {}: {error}",
                std::any::type_name::<Self::Verdict>()
            ),
        })
    }
}

/// Parses a JSON value and enforces the custom contract's compiled response schema.
pub(crate) struct JsonSchemaDecoder;

impl JsonSchemaDecoder {
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl VerdictDecoder for JsonSchemaDecoder {
    type Verdict = Value;

    fn decode(
        &self,
        response: &AggLlmResponse,
        contract: &ClassifierContract,
    ) -> Result<Self::Verdict> {
        let verdict = parse_json_verdict(response)?;
        contract.validate_verdict(&verdict)?;
        Ok(verdict)
    }
}

/// Runtime limits shared by structured classifier judges.
pub(crate) struct JudgeRuntimeConfig {
    max_output_tokens: u64,
}

impl JudgeRuntimeConfig {
    pub(crate) fn new(max_output_tokens: u64) -> Result<Self> {
        if max_output_tokens == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "max_output_tokens must be at least 1".to_string(),
            });
        }
        Ok(Self { max_output_tokens })
    }
}

/// Reusable structured judge assembled from an input view, contract, and verdict decoder.
pub(crate) struct StructuredJudge<I, D> {
    input: I,
    contract: ClassifierContract,
    decoder: D,
    runtime: JudgeRuntimeConfig,
}

impl<I, D> StructuredJudge<I, D> {
    pub(crate) fn new(
        input: I,
        contract: ClassifierContract,
        decoder: D,
        runtime: JudgeRuntimeConfig,
    ) -> Self {
        Self {
            input,
            contract,
            decoder,
            runtime,
        }
    }

    #[cfg(test)]
    pub(crate) fn contract(&self) -> &ClassifierContract {
        &self.contract
    }
}

impl<I, D> Judge for StructuredJudge<I, D>
where
    I: ClassifierInput,
    D: VerdictDecoder,
{
    type Verdict = D::Verdict;

    fn build_request(&self, state: &State, request: &Request) -> Request {
        let messages = self.input.build_messages(state, request);
        Request {
            llm_request: LlmRequest {
                model: request.llm_request.model.clone(),
                instructions: vec![InstructionBlock {
                    role: Role::System,
                    content: Message::text(Role::System, self.contract.system_prompt().to_string())
                        .content,
                }],
                messages,
                output: OutputParams {
                    max_output_tokens: Some(self.runtime.max_output_tokens),
                    response_format: Some(self.contract.response_format().clone()),
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: request.metadata.clone(),
        }
    }

    fn parse(&self, response: &AggLlmResponse) -> Result<Self::Verdict> {
        self.decoder.decode(response, &self.contract)
    }

    fn response_schema(&self) -> Option<&Value> {
        Some(self.contract.schema())
    }
}

/// Builds and parses requests for one algorithm-specific LLM judge.
pub trait Judge: Send + Sync {
    type Verdict: DeserializeOwned + Send + Sync;

    fn build_request(&self, state: &State, request: &Request) -> Request;

    fn parse(&self, response: &AggLlmResponse) -> Result<Self::Verdict> {
        parse_json_verdict(response)
    }

    /// The JSON Schema a verdict must match. Rejected replies are checked against it so
    /// logs can name missing or mistyped schema fields.
    fn response_schema(&self) -> Option<&Value> {
        None
    }
}

/// Converts a parsed verdict, or an unavailable verdict, into a routing classification.
/// Consider this as a deterministic policy which can act on the signals predicted from the classifier
/// and choose the route based on the verdict.
pub trait JudgePolicy: Send + Sync {
    type Verdict: Send + Sync;

    fn to_classification(
        &self,
        verdict: Option<&Self::Verdict>,
        driver: &Driver,
    ) -> Result<Classification>;
}

type EvidenceFn<V, P> = fn(&P, Option<&V>) -> Option<Value>;

/// A classifier that calls the runtime judge models and routes through its verdict policy.
pub struct JudgeClassifier<J, P>
where
    J: Judge,
    P: JudgePolicy<Verdict = J::Verdict>,
{
    judge: J,
    policy: P,
    evidence: Option<EvidenceFn<J::Verdict, P>>,
}

impl<J, P> JudgeClassifier<J, P>
where
    J: Judge,
    P: JudgePolicy<Verdict = J::Verdict>,
{
    /// Combines a judge with a verdict policy.
    pub fn new(judge: J, policy: P) -> Self {
        Self {
            judge,
            policy,
            evidence: None,
        }
    }

    /// Enables bounded evidence for built-in judges without widening the public policy trait.
    pub(crate) fn with_evidence(mut self, evidence: EvidenceFn<J::Verdict, P>) -> Self {
        self.evidence = Some(evidence);
        self
    }

    /// Logs and counts a judge call that failed before it produced a reply.
    /// `error` must already be redacted: `LlmClientError::UpstreamHttp`'s `Display` interpolates
    /// the raw upstream body, which can quote the conversation back. Callers pass a
    /// `robustness::safe_*` summary rather than the error itself.
    fn report_fail_open(&self, driver: &Driver, error: String, reason: &'static str) {
        let judge_target = driver
            .first_model_for(&Category::Judge)
            .map(|c| c.as_str())
            .unwrap_or("missing");
        tracing::warn!(
            target: "libsy",
            judge_model = judge_target,
            reason,
            error = %error,
            "judge verdict unavailable; routing without one"
        );
        self.record_fail_open(driver, judge_target, reason);
    }

    /// Counts a fail-open. Evidence is added only for evidence-enabled judges and never
    /// replaces an earlier decision.
    fn record_fail_open(&self, driver: &Driver, judge_model: &str, reason: &'static str) {
        crate::observability::record_classifier_fail_open(judge_model, reason);
        if self.evidence.is_some() {
            driver.set_evidence_if_empty(serde_json::json!({
                "source": "fail_open",
                "reason_code": reason,
            }));
        }
    }

    /// Consults the judge, yielding `None` when it is unavailable or unintelligible.
    ///
    /// A judge is an optimization, not a dependency: failing the caller's request because the
    /// judge is down would be worse than routing without it, so every failure — transport,
    /// mid-stream, or unparseable reply — is logged and folded into `None` for the policy's
    /// fallback branch. A closed driver stream is folded too; the algorithm's next driver
    /// call surfaces it, so nothing is masked.
    ///
    /// An unusable reply is reported to the host here. A parsed verdict comes back with the
    /// handle for that report, because only the policy can say whether it was usable.
    async fn verdict(
        &self,
        state: &mut State,
        request: &Request,
        driver: &Driver,
        judge_models: &[ModelId],
    ) -> Option<(J::Verdict, CallAssessor)> {
        let judge_model = judge_models.first()?.as_str();

        tracing::info!(target = judge_model, "consulting llm judge");
        let judge_request = self.judge.build_request(state, request);
        let requested_max_output_tokens = judge_request.llm_request.output.max_output_tokens;
        let (response, assessor) = driver
            .call_model_assessed(judge_request, judge_models.to_vec())
            .await
            .inspect_err(|error| {
                self.report_fail_open(driver, safe_error_summary(error), libsy_error_reason(error));
            })
            .ok()?;
        let aggregate = response.llm_response.into_agg().await.inspect_err(|error| {
            self.report_fail_open(driver, safe_client_error(error), client_error_reason(error));
        });
        let Ok(aggregate) = aggregate else {
            assessor.report(CallAssessment::Unusable {
                reason: "response_error",
            });
            return None;
        };
        let shape = ReplyShape::of(&aggregate);
        match self.judge.parse(&aggregate) {
            Ok(verdict) => {
                tracing::info!(
                    target: "libsy",
                    judge_model,
                    requested_max_output_tokens,
                    stop_reason = shape.stop_reason,
                    output_limit_reached = shape.output_limit_reached,
                    input_tokens = shape.input_tokens,
                    output_tokens = shape.output_tokens,
                    reasoning_tokens = shape.reasoning_tokens,
                    visible_chars = shape.visible_chars,
                    reasoning_chars = shape.reasoning_chars,
                    fenced = shape.fenced,
                    "llm judge reply parsed"
                );
                Some((verdict, assessor))
            }
            Err(_) => {
                let diagnosis =
                    diagnose_parse(&completion_text(&aggregate), self.judge.response_schema());
                tracing::warn!(
                    target: "libsy",
                    judge_model,
                    reason = "parse_error",
                    reason_code = diagnosis.reason_code,
                    json_category = diagnosis.json_category,
                    json_line = diagnosis.json_line,
                    json_column = diagnosis.json_column,
                    top_level_keys = diagnosis.top_level_keys,
                    missing_fields = %diagnosis.missing_fields,
                    wrong_type_fields = %diagnosis.wrong_type_fields,
                    unknown_key_count = diagnosis.unknown_key_count,
                    requested_max_output_tokens,
                    stop_reason = shape.stop_reason,
                    output_limit_reached = shape.output_limit_reached,
                    input_tokens = shape.input_tokens,
                    output_tokens = shape.output_tokens,
                    reasoning_tokens = shape.reasoning_tokens,
                    visible_chars = shape.visible_chars,
                    reasoning_chars = shape.reasoning_chars,
                    fenced = shape.fenced,
                    "judge verdict unavailable; routing without one"
                );
                self.record_fail_open(driver, judge_model, "parse_error");
                assessor.report(CallAssessment::Unusable {
                    reason: diagnosis.reason_code,
                });
                None
            }
        }
    }
}

/// Returns a bounded reason for a judge call that failed at the libsy layer.
pub(crate) fn libsy_error_reason(error: &LibsyError) -> &'static str {
    match error {
        LibsyError::ClientCall { source, .. } => client_error_reason(source),
        _ => "call_error",
    }
}

/// Returns a bounded reason from the error kind and HTTP status only.
fn client_error_reason(error: &LlmClientError) -> &'static str {
    match error {
        LlmClientError::RoutedCall { failure } => failure.class().stable_tag(),
        LlmClientError::Timeout { .. } => "timeout",
        LlmClientError::Transport { .. } => "transport",
        LlmClientError::UpstreamHttp { status, .. } if status.is_server_error() => "upstream_5xx",
        LlmClientError::UpstreamHttp { .. } => "upstream_non_5xx",
        LlmClientError::InvalidResponse { .. } | LlmClientError::ResponseTranslation(_) => {
            "invalid_response"
        }
        _ => "client_error",
    }
}

#[async_trait]
impl<J, P> Classifier<State> for JudgeClassifier<J, P>
where
    J: Judge,
    P: JudgePolicy<Verdict = J::Verdict>,
{
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        let judge_models = driver.models_for(&Category::Judge);
        if judge_models.is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "no models available for category Judge".to_string(),
            });
        }
        let (verdict, assessor) = self
            .verdict(state, request, driver, judge_models)
            .await
            .unzip();
        let classification = self.policy.to_classification(verdict.as_ref(), driver)?;
        if let Some(assessor) = assessor {
            assessor.report(match &classification {
                Classification::Scores(scores) if !scores.is_empty() => CallAssessment::Usable,
                _ => CallAssessment::Unusable {
                    reason: "verdict_rejected",
                },
            });
        }
        if let Some(evidence) = self
            .evidence
            .and_then(|evidence| evidence(&self.policy, verdict.as_ref()))
        {
            match &classification {
                Classification::Scores(scores) if !scores.is_empty() => {
                    driver.set_evidence(evidence);
                }
                _ => driver.set_evidence_if_empty(evidence),
            }
        }
        // A judge consultation is a side call, never the turn's answer.
        Ok((classification, None))
    }
}

fn parse_json_verdict<T: DeserializeOwned>(response: &AggLlmResponse) -> Result<T> {
    // Providers sometimes wrap otherwise valid JSON in a Markdown fence.
    let reply = completion_text(response);
    serde_json::from_str(strip_json_fence(reply.trim())).map_err(|err| LibsyError::AlgorithmError {
        message: format!(
            "judge reply did not parse as {}: {err}",
            std::any::type_name::<T>()
        ),
    })
}

pub(super) fn strip_json_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    let rest = rest.trim_start_matches(['\n', '\r']);
    rest.strip_suffix("```").map(str::trim).unwrap_or(rest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithm::RuntimeModels;
    use std::sync::Arc;

    use futures::StreamExt;
    use http::StatusCode;
    use serde::Deserialize;
    use switchyard_protocol::{ContentBlock, LlmClientError, text_request, text_response};

    use crate::core::algorithm::{AssessmentState, Step};
    use crate::core::classifier::Score;
    use switchyard_protocol::{LlmResponse, LlmResponseChunk, Response};

    const VERDICT: &str = r#"{"ok":true}"#;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestVerdict {
        ok: bool,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct ScoreVerdict {
        score: f64,
    }

    struct TestJudge;

    impl Judge for TestJudge {
        type Verdict = TestVerdict;

        fn build_request(&self, _state: &State, request: &Request) -> Request {
            request.clone()
        }
    }

    /// Reports only whether a verdict arrived, and abstains on a verdict of `false`.
    struct TestPolicy;

    impl JudgePolicy for TestPolicy {
        type Verdict = TestVerdict;

        fn to_classification(
            &self,
            verdict: Option<&Self::Verdict>,
            _driver: &Driver,
        ) -> Result<Classification> {
            let target = match verdict {
                Some(verdict) if !verdict.ok => return Ok(Classification::Ambiguous(vec![])),
                Some(_) => "verdict",
                None => "no-verdict",
            };
            Ok(Classification::Scores(vec![Score {
                target: ModelId::from(target),
                confidence: 1.0,
                category: None,
            }]))
        }
    }

    fn classifier() -> JudgeClassifier<TestJudge, TestPolicy> {
        JudgeClassifier::new(TestJudge, TestPolicy)
    }

    fn request() -> Request {
        let mut llm_request = text_request(Some("auto".to_string()), "judge this");
        llm_request.output.max_output_tokens = Some(128);
        Request {
            llm_request,
            raw_request: None,
            metadata: None,
        }
    }

    #[test]
    fn the_verdict_is_read_from_the_completion() -> Result<()> {
        // A judge's reasoning is not its answer: only `content` carries the verdict, so a
        // reply that never reached one — a run truncated mid-thought — is an error rather
        // than a guess.
        let mut response = text_response(None, VERDICT);
        if let Some(output) = response.outputs.first_mut() {
            output.content.insert(
                0,
                ContentBlock::Reasoning {
                    text: r#"{"ok":false}"#.to_string(),
                    signature: None,
                    details: Vec::new(),
                    openai_chat_field: Default::default(),
                },
            );
        }
        let parsed: TestVerdict = parse_json_verdict(&response)?;
        assert_eq!(parsed, TestVerdict { ok: true });

        assert!(parse_json_verdict::<TestVerdict>(&text_response(None, "still thinking")).is_err());
        Ok(())
    }

    #[test]
    fn typed_decoder_enforces_a_json_object_contract_locally() -> Result<()> {
        use super::super::classifier_contract::{
            ClassifierContractConfig, ClassifierResponseFormat,
        };

        let config = ClassifierContractConfig::default()
            .with_response_format_type(ClassifierResponseFormat::JsonObject);
        let contract = ClassifierContract::from_config(
            &config,
            "Return one JSON score.",
            r#"{
                "type": "json_schema",
                "json_schema": {
                    "name": "ScoreVerdict",
                    "schema": {
                        "type": "object",
                        "properties": {"score": {"type": "number"}},
                        "required": ["score"],
                        "additionalProperties": false
                    }
                }
            }"#,
        )?;
        let decoder = SerdeDecoder::<ScoreVerdict>::new();

        let error = decoder
            .decode(
                &text_response(None, r#"{"score":0.5,"unexpected":true}"#),
                &contract,
            )
            .expect_err("an extra property should fail the local schema");

        assert!(error.to_string().contains("did not match response_schema"));
        assert_eq!(
            decoder.decode(&text_response(None, r#"{"score":0.5}"#), &contract)?,
            ScoreVerdict { score: 0.5 }
        );
        Ok(())
    }

    fn buffered(completion: &str) -> Response {
        Response {
            llm_response: LlmResponse::Agg(text_response(None, completion)),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        }
    }

    fn streamed(chunks: Vec<LlmResponseChunk>) -> Response {
        Response {
            llm_response: LlmResponse::Stream(
                futures::stream::iter(chunks.into_iter().map(|chunk| Ok(chunk.into()))).boxed(),
            ),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        }
    }

    fn streamed_then_failing(chunk: LlmResponseChunk) -> Response {
        let items = futures::stream::iter([
            Ok(chunk.into()),
            Err(LlmClientError::Timeout {
                source: Box::new(std::io::Error::other("stream died")),
            }),
        ]);
        Response {
            llm_response: LlmResponse::Stream(items.boxed()),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        }
    }

    fn selected(classification: Classification) -> Result<ModelId> {
        classification
            .argmax(false)?
            .map(|score| score.target)
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "policy abstained".to_string(),
            })
    }

    /// Serves the single offloaded judge call with `reply` through a standalone step receiver.
    async fn score_served_with(reply: Result<Response>) -> Result<ModelId> {
        score_with(&classifier(), reply).await
    }

    async fn score_with<J>(
        classifier: &JudgeClassifier<J, TestPolicy>,
        reply: Result<Response>,
    ) -> Result<ModelId>
    where
        J: Judge<Verdict = TestVerdict>,
    {
        let (classification, _) = score_and_assess(classifier, reply).await?;
        selected(classification)
    }

    /// Serves the judge call like [`score_with`] and returns what the judge reported about
    /// the reply once scoring finished.
    async fn score_and_assess<J>(
        classifier: &JudgeClassifier<J, TestPolicy>,
        reply: Result<Response>,
    ) -> Result<(Classification, Option<AssessmentState>)>
    where
        J: Judge<Verdict = TestVerdict>,
    {
        let models = RuntimeModels::new([(Category::Judge, vec![ModelId::from("judge")])].into());
        let (driver, step_rx) = Driver::new("test", Arc::new(models));
        let mut steps = tokio_stream::wrappers::ReceiverStream::new(step_rx);
        let mut state = State::default();
        let mut request = request();

        let serve = async {
            let Some(Ok(Step::CallModel(mut call))) = steps.next().await else {
                return None;
            };
            let assessment = call.take_assessment();
            let _ = call.respond(reply);
            assessment
        };
        let (classification, assessment) =
            tokio::join!(classifier.score(&mut state, &mut request, &driver), serve);
        let (classification, _) = classification?;
        Ok((
            classification,
            assessment.map(|mut assessment| assessment.try_take()),
        ))
    }

    #[tokio::test]
    async fn a_buffered_verdict_reaches_the_policy() -> Result<()> {
        assert_eq!(score_served_with(Ok(buffered(VERDICT))).await?, "verdict");
        Ok(())
    }

    #[tokio::test]
    async fn a_streamed_verdict_is_drained_before_parsing() -> Result<()> {
        let chunks = VERDICT
            .chars()
            .map(|character| LlmResponseChunk::TextDelta {
                index: 0,
                text: character.to_string(),
            })
            .collect();
        assert_eq!(score_served_with(Ok(streamed(chunks))).await?, "verdict");
        Ok(())
    }

    #[tokio::test]
    async fn an_in_band_stream_error_falls_back_to_the_policy() -> Result<()> {
        let chunks = vec![
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "{\"ok\":".to_string(),
            },
            LlmResponseChunk::StreamError {
                message: "upstream exploded".to_string(),
            },
        ];
        assert_eq!(score_served_with(Ok(streamed(chunks))).await?, "no-verdict");
        Ok(())
    }

    #[tokio::test]
    async fn a_transport_failure_mid_stream_falls_back_to_the_policy() -> Result<()> {
        let partial = LlmResponseChunk::TextDelta {
            index: 0,
            text: "{\"ok\":".to_string(),
        };
        assert_eq!(
            score_served_with(Ok(streamed_then_failing(partial))).await?,
            "no-verdict"
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_unparseable_reply_falls_back_to_the_policy() -> Result<()> {
        assert_eq!(
            score_served_with(Ok(buffered("sorry, I can't help with that"))).await?,
            "no-verdict"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_judge_call_falls_back_to_the_policy() -> Result<()> {
        let error = LibsyError::client_call(
            "judge",
            LlmClientError::Timeout {
                source: Box::new(std::io::Error::other("judge unreachable")),
            },
        );
        assert_eq!(score_served_with(Err(error)).await?, "no-verdict");
        Ok(())
    }

    #[test]
    fn client_errors_map_to_bounded_fail_open_reasons() {
        let cases = vec![
            (
                LlmClientError::Timeout {
                    source: "deadline exceeded".into(),
                },
                "timeout",
            ),
            (
                LlmClientError::Transport {
                    source: "connection refused".into(),
                },
                "transport",
            ),
            (
                LlmClientError::UpstreamHttp {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    body: "server error".to_string(),
                },
                "upstream_5xx",
            ),
            (
                LlmClientError::UpstreamHttp {
                    status: StatusCode::FOUND,
                    body: "redirect".to_string(),
                },
                "upstream_non_5xx",
            ),
            (
                LlmClientError::InvalidResponse {
                    source: "invalid JSON".into(),
                },
                "invalid_response",
            ),
            (
                LlmClientError::General("unexpected client failure".to_string()),
                "client_error",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(client_error_reason(&error), expected);
        }

        let error = LibsyError::AlgorithmError {
            message: "driver failed".to_string(),
        };
        assert_eq!(libsy_error_reason(&error), "call_error");
    }

    #[test]
    fn fenced_replies_parse_as_verdicts() -> Result<()> {
        let judge = TestJudge;
        for reply in ["```json\n{\"ok\":true}\n```", "```\n{\"ok\":true}\n```"] {
            assert!(judge.parse(&text_response(None, reply))?.ok);
        }
        Ok(())
    }

    /// A judge reply cut off at its output cap: hidden reasoning, then the start of a verdict.
    fn truncated_at_cap(reasoning: &str, visible: &str) -> Response {
        let mut response = text_response(None, visible);
        response.usage.input_tokens = Some(959);
        response.usage.output_tokens = Some(128);
        let output = &mut response.outputs[0];
        output.stop_reason = Some(switchyard_protocol::StopReason::MaxTokens);
        output.content.insert(
            0,
            ContentBlock::Reasoning {
                text: reasoning.to_string(),
                signature: None,
                details: Vec::new(),
                openai_chat_field: Default::default(),
            },
        );
        Response {
            llm_response: LlmResponse::Agg(response),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        }
    }

    async fn logs_for<J>(
        classifier: &JudgeClassifier<J, TestPolicy>,
        reply: impl Fn() -> Response,
    ) -> (ModelId, String)
    where
        J: Judge<Verdict = TestVerdict>,
    {
        let capture = super::super::judge_diagnostics::LogCapture::default();
        let selected = {
            let _guard = capture.install();
            let _ = score_with(classifier, Ok(reply())).await;
            capture.rearm();
            score_with(classifier, Ok(reply())).await
        };
        (selected.expect("policy selection"), capture.text())
    }

    #[tokio::test]
    async fn a_reply_truncated_at_the_output_cap_is_logged_without_its_content() {
        let (selected, logs) = logs_for(&classifier(), || {
            truncated_at_cap("PRIVATE-REASONING", r#"{"ok": "SECRET-ECHO"#)
        })
        .await;

        assert_eq!(selected, "no-verdict");
        assert!(logs.contains("WARN"), "{logs}");
        for field in [
            "judge verdict unavailable; routing without one",
            "judge_model=\"judge\"",
            "reason=\"parse_error\"",
            "reason_code=\"truncated_json\"",
            "json_category=\"eof\"",
            "json_line=1",
            "requested_max_output_tokens=128",
            "stop_reason=\"max_tokens\"",
            "output_limit_reached=true",
            "input_tokens=959",
            "output_tokens=128",
            "visible_chars=19",
            "reasoning_chars=17",
        ] {
            assert!(logs.contains(field), "missing {field} in {logs}");
        }
        assert!(!logs.contains("SECRET-ECHO"), "{logs}");
        assert!(!logs.contains("PRIVATE-REASONING"), "{logs}");
        assert!(!logs.contains("llm judge reply parsed"), "{logs}");
    }

    #[tokio::test]
    async fn a_reply_spent_entirely_on_reasoning_is_an_empty_completion() {
        let (selected, logs) =
            logs_for(&classifier(), || truncated_at_cap("PRIVATE-REASONING", "")).await;

        assert_eq!(selected, "no-verdict");
        assert!(logs.contains("reason_code=\"empty_completion\""), "{logs}");
        assert!(logs.contains("visible_chars=0"), "{logs}");
        assert!(logs.contains("output_limit_reached=true"), "{logs}");
        assert!(!logs.contains("PRIVATE-REASONING"), "{logs}");
    }

    #[tokio::test]
    async fn a_parsed_verdict_is_logged_with_its_shape() {
        let (selected, logs) = logs_for(&classifier(), || buffered(VERDICT)).await;

        assert_eq!(selected, "verdict");
        assert!(logs.contains("INFO"), "{logs}");
        assert!(logs.contains("llm judge reply parsed"), "{logs}");
        assert!(logs.contains("requested_max_output_tokens=128"), "{logs}");
        assert!(logs.contains("output_limit_reached=false"), "{logs}");
        assert!(logs.contains("visible_chars=11"), "{logs}");
        assert!(!logs.contains("judge verdict unavailable"), "{logs}");
        assert!(!logs.contains("\"ok\""), "{logs}");
    }

    struct PassThroughInput;

    impl ClassifierInput for PassThroughInput {
        fn build_messages(&self, _state: &State, request: &Request) -> Vec<Message> {
            request.llm_request.messages.clone()
        }
    }

    #[tokio::test]
    async fn schema_mismatches_name_schema_fields_but_never_invented_keys() -> Result<()> {
        use super::super::classifier_contract::ClassifierContractConfig;

        let contract = ClassifierContract::from_config(
            &ClassifierContractConfig::default(),
            "Return one JSON verdict.",
            r#"{
                "type": "json_schema",
                "json_schema": {
                    "name": "TestVerdict",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}, "note": {"type": "string"}},
                        "required": ["ok", "note"],
                        "additionalProperties": false
                    }
                }
            }"#,
        )?;
        let judge = StructuredJudge::new(
            PassThroughInput,
            contract,
            SerdeDecoder::<TestVerdict>::new(),
            JudgeRuntimeConfig::new(256)?,
        );
        let (selected, logs) = logs_for(&JudgeClassifier::new(judge, TestPolicy), || {
            buffered(r#"{"ok":"SECRET-VALUE","INVENTED-KEY":1}"#)
        })
        .await;

        assert_eq!(selected, "no-verdict");
        for field in [
            "reason_code=\"schema_mismatch\"",
            "missing_fields=note",
            "wrong_type_fields=ok",
            "unknown_key_count=1",
            "top_level_keys=2",
            "requested_max_output_tokens=256",
            "stop_reason=\"unreported\"",
        ] {
            assert!(logs.contains(field), "missing {field} in {logs}");
        }
        assert!(!logs.contains("SECRET-VALUE"), "{logs}");
        assert!(!logs.contains("INVENTED-KEY"), "{logs}");
        Ok(())
    }

    async fn assessment_of(reply: Result<Response>) -> Option<AssessmentState> {
        score_and_assess(&classifier(), reply)
            .await
            .expect("scoring")
            .1
    }

    fn unusable(reason: &'static str) -> Option<AssessmentState> {
        Some(AssessmentState::Reported(CallAssessment::Unusable {
            reason,
        }))
    }

    #[tokio::test]
    async fn a_verdict_the_policy_acts_on_is_reported_usable() {
        assert_eq!(
            assessment_of(Ok(buffered(VERDICT))).await,
            Some(AssessmentState::Reported(CallAssessment::Usable))
        );
    }

    #[tokio::test]
    async fn an_unusable_reply_is_reported_with_its_parse_reason() {
        assert_eq!(
            assessment_of(Ok(truncated_at_cap("thinking", r#"{"ok": tr"#))).await,
            unusable("truncated_json")
        );
        assert_eq!(
            assessment_of(Ok(truncated_at_cap("thinking", ""))).await,
            unusable("empty_completion")
        );
        assert_eq!(
            assessment_of(Ok(buffered("sorry, I can't help with that"))).await,
            unusable("invalid_json")
        );
    }

    #[tokio::test]
    async fn a_parsed_verdict_the_policy_cannot_act_on_is_rejected() {
        assert_eq!(
            assessment_of(Ok(buffered(r#"{"ok":false}"#))).await,
            unusable("verdict_rejected")
        );
    }

    #[tokio::test]
    async fn a_reply_that_breaks_mid_stream_is_a_response_error() {
        let partial = LlmResponseChunk::TextDelta {
            index: 0,
            text: "{\"ok\":".to_string(),
        };
        assert_eq!(
            assessment_of(Ok(streamed_then_failing(partial))).await,
            unusable("response_error")
        );
    }

    #[tokio::test]
    async fn a_failed_call_leaves_nothing_to_assess() {
        let error = LibsyError::client_call(
            "judge",
            LlmClientError::Timeout {
                source: Box::new(std::io::Error::other("judge unreachable")),
            },
        );
        assert_eq!(
            assessment_of(Err(error)).await,
            Some(AssessmentState::NotReported)
        );
    }
}
