// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral conversation types shared by routing, clients, and translation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::format::FormatId;

/// Actor role normalized across provider APIs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// System-level instructions.
    System,
    /// Developer-level instructions supported by some provider APIs.
    Developer,
    /// End-user input.
    User,
    /// Model-generated output.
    Assistant,
    /// Tool execution output.
    Tool,
}

/// Instruction content separated from normal conversation messages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InstructionBlock {
    /// Instruction authority, normally [`Role::System`] or [`Role::Developer`].
    pub role: Role,
    /// Ordered instruction content.
    pub content: Vec<ContentBlock>,
}

/// One normalized conversation message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Actor that produced the message.
    pub role: Role,
    /// Ordered message content.
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Creates a text-only message for the given role.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// Concatenates text-like content blocks when the message has any.
    pub fn text_content(&self, separator: &str) -> Option<String> {
        let parts = self
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                ContentBlock::Refusal { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(separator))
        }
    }
}

/// Wire-level field used for OpenAI Chat plaintext reasoning.
///
/// The two names are provider dialects rather than interchangeable aliases:
/// callers may need to replay the exact field returned by a model.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiChatReasoningField {
    /// The `reasoning` field.
    #[default]
    Reasoning,
    /// The `reasoning_content` field.
    ReasoningContent,
}

impl OpenAiChatReasoningField {
    /// Returns true when the field uses the historical default spelling.
    pub const fn is_default(&self) -> bool {
        matches!(self, Self::Reasoning)
    }
}

/// Normalized content block variants carried by messages and tool results.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text content.
    Text {
        /// Text value.
        text: String,
    },
    /// Model reasoning or thinking content.
    Reasoning {
        /// Reasoning text.
        text: String,
        /// Provider signature used to validate or continue the reasoning block.
        signature: Option<String>,
        /// Structured reasoning details, such as an encrypted `{ "type":
        /// "reasoning.encrypted", "data": "..." }` object, replayed without modification.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        details: Vec<Value>,
        /// OpenAI Chat field that carried plaintext reasoning.
        #[serde(default, skip_serializing_if = "OpenAiChatReasoningField::is_default")]
        openai_chat_field: OpenAiChatReasoningField,
    },
    /// Image content.
    Image {
        /// Image location or inline payload.
        source: ImageSource,
    },
    /// Audio content.
    Audio {
        /// Audio location or inline payload.
        source: MediaSource,
    },
    /// Video content.
    Video {
        /// Video location or inline payload.
        source: MediaSource,
    },
    /// File content.
    File {
        /// File identifier or inline payload.
        source: FileSource,
    },
    /// Tool invocation requested by the assistant.
    ToolCall(ToolCall),
    /// Result of an earlier tool invocation.
    ToolResult(ToolResult),
    /// Free-form input tool invocation requested by the assistant.
    CustomToolCall(CustomToolCall),
    /// Result of an earlier free-form input tool invocation.
    CustomToolResult(CustomToolResult),
    /// Computer interaction requested by the assistant.
    ComputerToolCall(ComputerToolCall),
    /// Screenshot and safety acknowledgements returned after computer interaction.
    ComputerToolResult(ComputerToolResult),
    /// Provider-hosted tool invocation requested by the assistant.
    HostedToolCall(HostedToolCall),
    /// Result produced by a provider-hosted tool.
    HostedToolResult(HostedToolResult),
    /// Bounded opaque state retained for a later request.
    OpaqueState(OpaqueState),
    /// Provider-created conversation compaction state.
    Compaction(CompactionItem),
    /// A provider-native generated image.
    GeneratedImage(GeneratedImage),
    /// A provider pause that requires another turn instead of ordinary completion.
    PauseTurn(PauseTurn),
    /// Provider refusal content.
    Refusal {
        /// Human-readable refusal text.
        text: String,
    },
    /// Provider block that has no normalized representation.
    Unknown {
        /// Wire format that supplied the block.
        provider: FormatId,
        /// Exact provider block.
        raw: Value,
    },
}

/// Image payload forms supported by the conversation model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ImageSource {
    /// Image fetched from a URL.
    Url {
        /// Image URL.
        url: String,
        /// Optional provider-specific image detail level.
        detail: Option<String>,
    },
    /// Base64-encoded image bytes.
    Base64 {
        /// MIME type, when supplied.
        media_type: Option<String>,
        /// Base64-encoded bytes.
        data: String,
    },
    /// Provider image source with no normalized representation.
    Raw(Value),
}

/// File payload forms supported by the conversation model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum FileSource {
    /// Provider-managed file identifier.
    FileId(String),
    /// Inline file payload.
    FileData {
        /// Encoded or textual file data as supplied by the provider format.
        data: String,
        /// Original filename, when supplied.
        filename: Option<String>,
    },
    /// Provider file source with no normalized representation.
    Raw(Value),
}

/// Audio and video payload forms supported by the conversation model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum MediaSource {
    /// Media fetched from a URL.
    Url {
        /// Media URL.
        url: String,
        /// MIME type, when supplied.
        media_type: Option<String>,
    },
    /// Base64-encoded media bytes.
    Base64 {
        /// MIME type, when supplied.
        media_type: Option<String>,
        /// Base64-encoded bytes.
        data: String,
    },
    /// Provider media source with no normalized representation.
    Raw(Value),
}

/// Normalized assistant tool call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider tool-call identifier used to pair the result.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Parsed tool arguments.
    pub arguments: Value,
}

/// Normalized tool result message content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Identifier of the [`ToolCall`] this result answers.
    pub tool_call_id: String,
    /// Ordered tool output content.
    pub content: Vec<ContentBlock>,
    /// Whether tool execution failed, when the provider reports it.
    pub is_error: Option<bool>,
}

/// Free-form input tool invocation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomToolCall {
    /// Provider tool-call identifier used to pair the result.
    pub id: String,
    /// Provider item identifier, when the protocol distinguishes items from calls.
    pub item_id: Option<String>,
    /// Tool name.
    pub name: String,
    /// Unparsed text input supplied to the tool.
    pub input: String,
}

/// Result of an earlier [`CustomToolCall`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomToolResult {
    /// Identifier of the custom tool call this result answers.
    pub tool_call_id: String,
    /// Text returned by the tool.
    pub output: String,
    /// Whether tool execution failed, when reported.
    pub is_error: Option<bool>,
}

/// Screen geometry and environment exposed to a computer tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComputerToolConfig {
    /// Width of the interactive surface in pixels.
    pub display_width: u32,
    /// Height of the interactive surface in pixels.
    pub display_height: u32,
    /// Kind of interactive surface.
    pub environment: ComputerEnvironment,
}

/// Kind of interactive surface exposed to a computer tool.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerEnvironment {
    /// Browser viewport.
    Browser,
    /// Desktop session.
    Desktop,
    /// Mobile-device session.
    Mobile,
}

/// Coordinate on a computer tool's interactive surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ComputerPoint {
    /// Horizontal coordinate.
    pub x: i64,
    /// Vertical coordinate.
    pub y: i64,
}

/// Mouse button used by a computer action.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerMouseButton {
    /// Primary mouse button.
    Left,
    /// Middle mouse button.
    Middle,
    /// Secondary mouse button.
    Right,
}

/// One provider-neutral computer interaction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ComputerAction {
    /// Click one point.
    Click {
        /// Click location.
        point: ComputerPoint,
        /// Mouse button to click.
        button: ComputerMouseButton,
    },
    /// Double-click one point.
    DoubleClick {
        /// Click location.
        point: ComputerPoint,
        /// Mouse button to click.
        button: ComputerMouseButton,
    },
    /// Drag through an ordered path.
    Drag {
        /// Ordered drag path.
        path: Vec<ComputerPoint>,
    },
    /// Press one or more keys together.
    KeyPress {
        /// Provider-neutral key names.
        keys: Vec<String>,
    },
    /// Move the pointer without clicking.
    Move {
        /// Destination.
        point: ComputerPoint,
    },
    /// Request a screenshot without another interaction.
    Screenshot,
    /// Scroll at a point.
    Scroll {
        /// Pointer location for the scroll.
        point: ComputerPoint,
        /// Horizontal scroll distance.
        delta_x: i64,
        /// Vertical scroll distance.
        delta_y: i64,
    },
    /// Enter literal text.
    Type {
        /// Text to enter.
        text: String,
    },
    /// Wait for the remote surface.
    Wait,
}

/// One safety check attached to a computer action.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ComputerSafetyCheck {
    /// Stable identifier echoed when the check is acknowledged.
    pub id: String,
    /// Human-readable check description.
    pub description: String,
}

/// Computer actions requested by the assistant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComputerToolCall {
    /// Provider tool-call identifier used to pair the result.
    pub id: String,
    /// Provider item identifier, when supplied.
    pub item_id: Option<String>,
    /// Ordered actions to perform.
    pub actions: Vec<ComputerAction>,
    /// Checks that must be acknowledged before executing the actions.
    pub pending_safety_checks: Vec<ComputerSafetyCheck>,
}

/// Result of an earlier [`ComputerToolCall`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComputerToolResult {
    /// Identifier of the computer tool call this result answers.
    pub tool_call_id: String,
    /// Screenshot or other native image returned after executing the action.
    pub output: ImageSource,
    /// Safety checks acknowledged by the executor.
    pub acknowledged_safety_checks: Vec<String>,
}

/// A provider-hosted tool capability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HostedTool {
    /// Search public web content.
    WebSearch {
        /// Optional allowlist of domains.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        allowed_domains: Vec<String>,
    },
    /// Search provider-managed files.
    FileSearch {
        /// Provider-managed collection identifiers.
        vector_store_ids: Vec<String>,
        /// Maximum result count, when constrained.
        max_results: Option<u32>,
    },
    /// Execute code in a provider-managed container.
    CodeInterpreter {
        /// Existing container identifier, when one is reused.
        container_id: Option<String>,
    },
    /// Generate an image.
    ImageGeneration,
}

/// Invocation of a provider-hosted tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HostedToolCall {
    /// Provider call identifier.
    pub id: String,
    /// Provider item identifier, when supplied.
    pub item_id: Option<String>,
    /// Hosted capability being invoked.
    pub tool: HostedTool,
    /// Capability arguments expressed by the neutral JSON input contract.
    pub arguments: Value,
}

/// Result of an earlier [`HostedToolCall`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HostedToolResult {
    /// Identifier of the hosted tool call this result answers.
    pub tool_call_id: String,
    /// Ordered typed output content.
    pub content: Vec<ContentBlock>,
    /// Whether hosted execution failed, when reported.
    pub is_error: Option<bool>,
}

/// Grammar accepted by a free-form input tool.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CustomToolFormat {
    /// Unconstrained text.
    Text,
    /// Input constrained by a grammar.
    Grammar {
        /// Grammar language.
        syntax: GrammarSyntax,
        /// Grammar definition.
        definition: String,
    },
}

/// Supported grammar languages for free-form tools.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrammarSyntax {
    /// Lark grammar.
    Lark,
    /// Regular expression.
    Regex,
}

/// Provider-neutral declaration for a non-function tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SpecializedToolDefinition {
    /// Free-form input tool.
    Custom {
        /// Tool name exposed to the model.
        name: String,
        /// Human-readable tool description.
        description: Option<String>,
        /// Accepted free-form input.
        format: CustomToolFormat,
    },
    /// Computer interaction tool.
    Computer {
        /// Screen and environment exposed to the model.
        configuration: ComputerToolConfig,
    },
    /// Provider-hosted capability.
    Hosted {
        /// Capability exposed to the model.
        tool: HostedTool,
    },
}

/// Normalized tool definition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name exposed to the model.
    pub name: String,
    /// Human-readable tool description.
    pub description: Option<String>,
    /// JSON Schema describing accepted arguments.
    pub parameters: Value,
    /// Whether the provider should enforce the schema strictly.
    pub strict: Option<bool>,
}

/// Normalized tool choice policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the model decide whether to call a tool.
    Auto,
    /// Require at least one tool call.
    Required,
    /// Disallow tool calls.
    None,
    /// Require a specific tool.
    Tool {
        /// Required tool name.
        name: String,
    },
    /// Provider tool-choice value with no normalized representation.
    Raw(Value),
}

/// Provider sampling parameters with common cross-provider names.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Nucleus-sampling probability mass.
    pub top_p: Option<f64>,
    /// Maximum number of candidate tokens considered at each step.
    pub top_k: Option<i64>,
}

/// Output budget and structured-output options.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OutputParams {
    /// Maximum tokens the model may generate.
    pub max_output_tokens: Option<u64>,
    /// Provider-neutral or provider-specific structured-output configuration.
    pub response_format: Option<Value>,
}

/// Provider reasoning controls preserved by translation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningParams {
    /// Requested reasoning effort or level.
    pub effort: Option<String>,
    /// Provider reasoning controls without a normalized field.
    pub raw: Option<Value>,
}

/// Purpose of encrypted or otherwise opaque provider state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpaqueStatePurpose {
    /// Encrypted reasoning continuation state.
    Reasoning,
    /// Conversation continuation state.
    Conversation,
    /// Remote compaction state.
    Compaction,
    /// Within-turn routing state.
    TurnRouting,
}

/// Error returned when opaque provider state exceeds its protocol bound.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[error("opaque provider state exceeds {MAX_OPAQUE_STATE_BYTES} bytes")]
pub struct OpaqueStateTooLarge;

/// Maximum encoded size of one opaque provider state value.
pub const MAX_OPAQUE_STATE_BYTES: usize = 1024 * 1024;

/// Bounded opaque state that can be replayed without exposing provider spellings.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OpaqueState {
    purpose: OpaqueStatePurpose,
    data: String,
}

impl OpaqueState {
    /// Creates bounded opaque state.
    pub fn new(
        purpose: OpaqueStatePurpose,
        data: impl Into<String>,
    ) -> Result<Self, OpaqueStateTooLarge> {
        let data = data.into();
        if data.len() > MAX_OPAQUE_STATE_BYTES {
            return Err(OpaqueStateTooLarge);
        }
        Ok(Self { purpose, data })
    }

    /// Purpose of this state.
    pub fn purpose(&self) -> OpaqueStatePurpose {
        self.purpose
    }

    /// Encoded state value.
    pub fn data(&self) -> &str {
        &self.data
    }
}

impl<'de> Deserialize<'de> for OpaqueState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            purpose: OpaqueStatePurpose,
            data: String,
        }

        let repr = Repr::deserialize(deserializer)?;
        Self::new(repr.purpose, repr.data).map_err(serde::de::Error::custom)
    }
}

/// Provider-created conversation compaction output.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompactionItem {
    /// Provider item identifier, when supplied.
    pub id: Option<String>,
    /// Human-readable summary, when this compaction form exposes one.
    pub summary: Option<String>,
    /// Opaque continuation state, when this compaction form is encrypted.
    pub state: Option<OpaqueState>,
}

/// Provider-native generated image output.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeneratedImage {
    /// Provider item identifier, when supplied.
    pub id: Option<String>,
    /// Generated image location or inline payload.
    pub source: ImageSource,
}

/// Update to a turn's generation configuration.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TurnConfigurationUpdate {
    /// New reasoning effort, when changed.
    pub reasoning_effort: Option<String>,
    /// New response verbosity, when changed.
    pub text_verbosity: Option<String>,
}

/// A provider pause that requires another request to continue the turn.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PauseTurn {
    /// Stable continuation identifier, when supplied.
    pub continuation_id: Option<String>,
    /// Safe provider-neutral reason, when supplied.
    pub reason: Option<String>,
}

/// Non-message input appended to a turn in order.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum TurnInputItem {
    /// Change turn configuration after the preceding conversation items.
    ConfigurationUpdate(TurnConfigurationUpdate),
    /// Request provider-native remote compaction as the final turn item.
    RemoteCompaction,
}

/// Safe provider error classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderErrorClass {
    /// Credential was not accepted.
    Authentication,
    /// Credential was valid but lacked authority.
    Authorization,
    /// Provider rate limit.
    RateLimited,
    /// Request violated the provider contract.
    InvalidRequest,
    /// Request exceeded the model context window.
    ContextWindow,
    /// Provider safety policy rejected content.
    ContentFiltered,
    /// Provider or dependency was unavailable.
    Unavailable,
    /// Provider response violated the expected protocol.
    Protocol,
    /// Provider reported an internal failure.
    Internal,
}

/// Structured, body-free provider error.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StructuredProviderError {
    /// Stable provider-neutral classification.
    pub class: ProviderErrorClass,
    /// HTTP status, when the failure came from HTTP.
    pub status: Option<u16>,
    /// Bounded safe provider code, when allowlisted by an adapter.
    pub code: Option<String>,
    /// Provider-advertised retry delay.
    pub retry_after_ms: Option<u64>,
}

/// Provider-neutral response terminal state.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ResponseTerminal {
    /// Response completed successfully.
    Completed,
    /// Response ended without completing.
    Incomplete {
        /// Safe provider-neutral reason, when supplied.
        reason: Option<String>,
    },
    /// Response failed with a structured body-free error.
    Failed(StructuredProviderError),
    /// Provider paused the turn for explicit continuation.
    Paused(PauseTurn),
}

/// Nonsemantic response metadata retained across compatible requests.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Model-catalog entity tag returned by the provider.
    pub model_etag: Option<String>,
    /// Bounded within-turn routing state.
    pub turn_state: Option<OpaqueState>,
}

impl ResponseMetadata {
    pub(crate) fn is_empty(&self) -> bool {
        self.model_etag.is_none() && self.turn_state.is_none()
    }
}

/// Provider-specific fields that do not have first-class conversation fields.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderExtensions {
    /// Provider fields keyed by their original wire names.
    ///
    /// Codecs use these fields when translating to another format. Exact
    /// same-format replay is controlled separately by [`PreservationMetadata`].
    pub fields: Map<String, Value>,
}

/// Exact source payloads retained for lossless same-format round trips.
///
/// Translation's default preservation policy prefers a stored same-format body
/// over reconstructing one from normalized fields. A caller that mutates the IR
/// must clear the corresponding entry or use a policy with preservation disabled
/// when those mutations must be encoded.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PreservationMetadata {
    /// Original request bodies keyed by source format.
    pub requests: BTreeMap<FormatId, Value>,
    /// Original response bodies keyed by source format.
    pub responses: BTreeMap<FormatId, Value>,
}

/// Normalized request representation shared by Switchyard components.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmRequest {
    /// Model currently addressed by the request.
    ///
    /// This initially contains the name supplied by the inbound client. A routing host may
    /// replace it with the selected target before serving the request.
    pub model: Option<String>,
    /// System and developer instructions separated from conversation turns.
    pub instructions: Vec<InstructionBlock>,
    /// Ordered conversation messages.
    pub messages: Vec<Message>,
    /// Ordered non-message items appended after the conversation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trailing_items: Vec<TurnInputItem>,
    /// Tools available to the model.
    pub tools: Vec<ToolDefinition>,
    /// Non-function tools available to the model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub specialized_tools: Vec<SpecializedToolDefinition>,
    /// Policy controlling model tool selection.
    pub tool_choice: Option<ToolChoice>,
    /// Common sampling controls.
    pub sampling: SamplingParams,
    /// Output budget and shape controls.
    pub output: OutputParams,
    /// Reasoning controls.
    pub reasoning: ReasoningParams,
    /// Whether the caller requested a streamed response.
    pub stream: bool,
    /// Provider fields without first-class normalized equivalents.
    pub extensions: ProviderExtensions,
    /// Exact provider bodies used by codecs for lossless same-format round trips.
    /// This is separate from a host's optional
    /// [`Request::raw_request`](crate::Request::raw_request).
    pub preservation: PreservationMetadata,
}

/// Normalized token usage counts.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Non-cached input tokens. Provider codecs normalize aggregate OpenAI
    /// input counts by subtracting the cache detail fields.
    pub input_tokens: Option<u64>,
    /// Cache-read and cache-creation token detail, when reported.
    #[serde(flatten)]
    pub cache: Option<Box<InputCacheUsage>>,
    /// Generated output tokens, excluding reasoning detail when the provider reports it separately.
    pub output_tokens: Option<u64>,
    /// Provider-reported or codec-computed total token count.
    ///
    /// Codecs normalize OpenAI aggregate input counts before computing totals, so
    /// this may equal non-cached input plus cache detail plus output.
    pub total_tokens: Option<u64>,
    /// Reasoning output tokens, when reported separately.
    pub reasoning_tokens: Option<u64>,
}

/// Optional cache-token detail kept out of the common, cache-free usage allocation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InputCacheUsage {
    /// Input tokens read from a provider cache.
    pub cached_input_tokens: Option<u64>,
    /// Input tokens written into a provider cache.
    pub cache_creation_input_tokens: Option<u64>,
}

impl Usage {
    /// Builds cache detail when at least one count is present.
    pub fn cache_details(
        cached_input_tokens: Option<u64>,
        cache_creation_input_tokens: Option<u64>,
    ) -> Option<Box<InputCacheUsage>> {
        if cached_input_tokens.is_none() && cache_creation_input_tokens.is_none() {
            return None;
        }
        Some(Box::new(InputCacheUsage {
            cached_input_tokens,
            cache_creation_input_tokens,
        }))
    }

    /// Returns the cache-read input-token count.
    pub fn cached_input_tokens(&self) -> Option<u64> {
        self.cache
            .as_ref()
            .and_then(|cache| cache.cached_input_tokens)
    }

    /// Returns the cache-creation input-token count.
    pub fn cache_creation_input_tokens(&self) -> Option<u64> {
        self.cache
            .as_ref()
            .and_then(|cache| cache.cache_creation_input_tokens)
    }

    /// Sets the cache-read count, allocating cache detail when needed.
    pub fn set_cached_input_tokens(&mut self, value: u64) {
        self.cache
            .get_or_insert_with(|| Box::new(InputCacheUsage::default()))
            .cached_input_tokens = Some(value);
    }

    /// Sets the cache-creation count, allocating cache detail when needed.
    pub fn set_cache_creation_input_tokens(&mut self, value: u64) {
        self.cache
            .get_or_insert_with(|| Box::new(InputCacheUsage::default()))
            .cache_creation_input_tokens = Some(value);
    }
}

/// Normalized reason a model stopped producing output.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Model completed its turn normally.
    EndTurn,
    /// Model reached the configured output limit.
    MaxTokens,
    /// Model stopped to request a tool call.
    ToolUse,
    /// Provider safety or content filtering stopped generation.
    ContentFilter,
    /// Generation terminated because of an error.
    Error,
    /// Provider stop reason with no normalized equivalent.
    Unknown,
}

/// One assistant output item in a normalized response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseOutput {
    /// Actor that produced the output, normally [`Role::Assistant`].
    pub role: Role,
    /// Ordered output content.
    pub content: Vec<ContentBlock>,
    /// Why this output stopped, when known.
    pub stop_reason: Option<StopReason>,
}

/// Normalized, fully-buffered response — the aggregate of a completed generation.
/// This is the terminal form of a streamed [`LlmResponse`](crate::LlmResponse).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AggLlmResponse {
    /// Provider response identifier.
    pub id: Option<String>,
    /// Model reported by the provider.
    pub model: Option<String>,
    /// Ordered response output items.
    pub outputs: Vec<ResponseOutput>,
    /// Normalized token usage.
    pub usage: Usage,
    /// Nonsemantic response metadata.
    #[serde(default, skip_serializing_if = "ResponseMetadata::is_empty")]
    pub metadata: ResponseMetadata,
    /// Provider-neutral terminal state, when explicitly reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<ResponseTerminal>,
    /// Provider response fields without normalized equivalents.
    pub extensions: ProviderExtensions,
    /// Exact provider bodies retained for lossless round trips.
    pub preservation: PreservationMetadata,
}

impl AggLlmResponse {
    /// Returns the first output item when a response has any output.
    pub fn first_output(&self) -> Option<&ResponseOutput> {
        self.outputs.first()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn serde_uses_python_friendly_dictionary_shapes() -> Result<(), serde_json::Error> {
        let request: LlmRequest = serde_json::from_value(json!({
            "model": "auto",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hello"}]
            }]
        }))?;
        assert_eq!(request.messages[0], Message::text(Role::User, "hello"));

        let tool_call = ContentBlock::ToolCall(ToolCall {
            id: "call-1".to_string(),
            name: "lookup".to_string(),
            arguments: json!({"query": "rust"}),
        });
        assert_eq!(
            serde_json::to_value(tool_call)?,
            json!({
                "type": "tool_call",
                "id": "call-1",
                "name": "lookup",
                "arguments": {"query": "rust"}
            })
        );
        Ok(())
    }
}
