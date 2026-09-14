// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming half of the neutral IR: incremental response chunks ([`LlmResponseChunk`]),
//! their stream envelope ([`LlmResponseStreamEvent`]), and the streamed response
//! ([`LlmResponse`]) that carries either a live stream or the terminal [`AggLlmResponse`].

use std::collections::BTreeMap;
use std::pin::Pin;

use futures::{Stream, StreamExt};
use http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::{
    LlmClientError,
    format::FormatId,
    llm::{
        AggLlmResponse, CompactionItem, ComputerToolCall, ContentBlock, CustomToolCall,
        GeneratedImage, HostedToolCall, OpaqueState, PauseTurn, ResponseMetadata, ResponseOutput,
        ResponseTerminal, Role, StopReason, ToolCall, Usage,
    },
};

/// Status reported for an upstream error delivered inside a streaming body. The
/// upstream already sent a success status line before failing, so there is no real
/// code to propagate; 502 matches how a failed upstream call surfaces elsewhere.
const MID_STREAM_UPSTREAM_STATUS: StatusCode = StatusCode::BAD_GATEWAY;

/// Why a translated event stream stopped early.
#[derive(Debug, Error)]
pub enum LlmStreamError {
    /// An upstream sse error
    #[error("upstream stream error: {0}")]
    Upstream(Value),

    /// The stream itself failed
    #[error(transparent)]
    Client(#[from] LlmClientError),
}

/// A boxed, `Send` stream of response events. Each item may fail independently mid-stream.
pub type LlmResponseStream =
    Pin<Box<dyn Stream<Item = Result<LlmResponseStreamEvent, LlmClientError>> + Send>>;

/// Parsed provider event retained for same-format replay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderStreamEvent {
    source: FormatId,
    raw: Value,
}

impl ProviderStreamEvent {
    /// Source wire format of the retained event.
    pub fn source(&self) -> &FormatId {
        &self.source
    }

    /// Parsed provider JSON retained for replay.
    pub fn raw(&self) -> &Value {
        &self.raw
    }

    /// Consumes the preservation value into its source format and parsed JSON.
    pub fn into_parts(self) -> (FormatId, Value) {
        (self.source, self.raw)
    }
}

/// One streaming item crossing the host/algorithm boundary.
///
/// `normalized` contains only provider-neutral chunks. `preservation` is opaque to
/// algorithms and is interpreted by `switchyard-translation` for same-format replay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmResponseStreamEvent {
    preservation: Option<ProviderStreamEvent>,
    normalized: Vec<LlmResponseChunk>,
}

impl LlmResponseStreamEvent {
    /// Creates an event containing only normalized chunks.
    pub fn new(normalized: Vec<LlmResponseChunk>) -> Self {
        Self {
            preservation: None,
            normalized,
        }
    }

    /// Creates an event with parsed provider JSON retained for replay.
    pub fn preserved(
        source: impl Into<FormatId>,
        raw: Value,
        normalized: Vec<LlmResponseChunk>,
    ) -> Self {
        Self {
            preservation: Some(ProviderStreamEvent {
                source: source.into(),
                raw,
            }),
            normalized,
        }
    }

    /// Retained provider event, when this event came directly from a provider.
    pub fn preservation(&self) -> Option<&ProviderStreamEvent> {
        self.preservation.as_ref()
    }

    /// Provider-neutral chunks carried by this event.
    pub fn normalized(&self) -> &[LlmResponseChunk] {
        &self.normalized
    }

    /// Consumes the event into its preservation and normalized content.
    pub fn into_parts(self) -> (Option<ProviderStreamEvent>, Vec<LlmResponseChunk>) {
        (self.preservation, self.normalized)
    }

    /// Replaces semantic content and drops raw replay data that no longer describes it.
    pub fn replace_normalized(self, normalized: Vec<LlmResponseChunk>) -> Self {
        Self::new(normalized)
    }
}

impl From<LlmResponseChunk> for LlmResponseStreamEvent {
    fn from(chunk: LlmResponseChunk) -> Self {
        Self::new(vec![chunk])
    }
}

/// A model response: either a live [`Stream`](LlmResponse::Stream) of events or a
/// terminal buffered [`LlmResponse::Agg`] response.
///
/// Not `Clone` — the `Stream` variant owns a single-consumption stream. A buffered
/// backend returns `Agg` directly; a streaming one returns `Stream` and the consumer
/// drives it, folding to an [`AggLlmResponse`] when it needs the whole response.
#[allow(clippy::large_enum_variant)]
pub enum LlmResponse {
    /// Live, single-consumption response stream.
    Stream(LlmResponseStream),
    /// Fully buffered terminal response.
    Agg(AggLlmResponse),
}

impl LlmResponse {
    /// Borrow the aggregate; `None` while this is still a stream.
    pub fn as_agg(&self) -> Option<&AggLlmResponse> {
        match self {
            LlmResponse::Agg(agg) => Some(agg),
            LlmResponse::Stream(_) => None,
        }
    }

    /// Reduce to the buffered aggregate: return an `Agg` unchanged, or drive a `Stream`
    /// to completion, folding its normalized chunks into an [`AggLlmResponse`] via
    /// [`ResponseAccumulator`]. A stream item error aborts with `Err`, as does an
    /// in-band [`LlmResponseChunk::DecodeError`] (as `ResponseTranslation`) or
    /// [`LlmResponseChunk::StreamError`] (as `UpstreamHttp`).
    pub async fn into_agg(self) -> Result<AggLlmResponse, LlmClientError> {
        match self {
            LlmResponse::Agg(agg) => Ok(agg),
            LlmResponse::Stream(mut stream) => {
                let mut accumulator = ResponseAccumulator::new();
                while let Some(item) = stream.next().await {
                    for chunk in item?.normalized {
                        push_checked_chunk(&mut accumulator, chunk)?;
                    }
                }
                Ok(accumulator.finish())
            }
        }
    }

    /// Returns the model recorded by a buffered response.
    ///
    /// Returns `None` for a live stream because reading its model-bearing
    /// [`LlmResponseChunk::MessageStart`] event would consume the stream.
    pub fn selected_model(&self) -> Option<&str> {
        match self {
            LlmResponse::Agg(agg) => agg.model.as_deref(),
            // TODO: How do we get the model name on a stream?
            LlmResponse::Stream(_) => None,
        }
    }
}

/// An encrypted reasoning detail that names its provider item id but carries no payload yet.
fn is_reasoning_id_announcement(detail: &Value) -> bool {
    detail.get("type").and_then(Value::as_str) == Some("reasoning.encrypted")
        && detail.get("data").is_none()
}

/// Appends a reasoning detail to an accumulated block. A Responses stream decoder announces a
/// reasoning item's provider id (`{"type": "reasoning.encrypted", "id"}`) before the payload
/// arrives; the payload detail then replaces that announcement so history holds one detail.
fn push_reasoning_detail(details: &mut Vec<Value>, detail: Value) {
    if detail.get("type").and_then(Value::as_str) == Some("reasoning.encrypted")
        && let Some(id) = detail.get("id").and_then(Value::as_str)
        && let Some(announcement) = details.iter_mut().find(|existing| {
            is_reasoning_id_announcement(existing)
                && existing.get("id").and_then(Value::as_str) == Some(id)
        })
    {
        *announcement = detail;
        return;
    }
    details.push(detail);
}

impl AggLlmResponse {
    /// Converts a fully-buffered response into a synthetic chunk stream.
    ///
    /// Useful when a caller had to buffer the response (e.g. to judge it) but the
    /// downstream expects `LlmResponse::Stream` — for instance when `stream: true` was
    /// requested and the algorithm had to aggregate before it could return.
    ///
    /// This conversion retains semantic output, typed continuation state,
    /// response metadata, and explicit terminal state. Tool results, input media,
    /// files, unknown blocks, response extensions, and preservation metadata are omitted.
    pub fn into_stream(self) -> LlmResponseStream {
        let mut chunks: Vec<LlmResponseChunk> = Vec::new();
        chunks.push(LlmResponseChunk::MessageStart {
            id: self.id,
            model: self.model,
        });
        let mut tool_call_index = 0usize;
        for (output_index, output) in self.outputs.into_iter().enumerate() {
            for block in output.content {
                match block {
                    ContentBlock::Text { text } => {
                        chunks.push(LlmResponseChunk::TextDelta {
                            index: output_index,
                            text,
                        });
                    }
                    ContentBlock::Reasoning { text, details, .. } => {
                        if !details.is_empty() {
                            chunks.push(LlmResponseChunk::ReasoningDetailsDelta {
                                index: output_index,
                                details,
                                text,
                            });
                        } else {
                            chunks.push(LlmResponseChunk::ReasoningDelta {
                                index: output_index,
                                text,
                            });
                        }
                    }
                    ContentBlock::ToolCall(call) => {
                        chunks.push(LlmResponseChunk::ToolCallDelta {
                            index: tool_call_index,
                            id: Some(call.id.clone()),
                            name: Some(call.name.clone()),
                            arguments_delta: serde_json::to_string(&call.arguments).ok(),
                        });
                        chunks.push(LlmResponseChunk::ToolCallDone {
                            index: tool_call_index,
                            call,
                        });
                        tool_call_index += 1;
                    }
                    ContentBlock::CustomToolCall(call) => {
                        chunks.push(LlmResponseChunk::CustomToolCallDelta {
                            index: tool_call_index,
                            id: Some(call.id.clone()),
                            item_id: call.item_id.clone(),
                            name: Some(call.name.clone()),
                            input_delta: call.input.clone(),
                        });
                        chunks.push(LlmResponseChunk::CustomToolCallDone {
                            index: tool_call_index,
                            call,
                        });
                        tool_call_index += 1;
                    }
                    ContentBlock::ComputerToolCall(call) => {
                        chunks.push(LlmResponseChunk::ComputerToolCallDone {
                            index: tool_call_index,
                            call,
                        });
                        tool_call_index += 1;
                    }
                    ContentBlock::HostedToolCall(call) => {
                        chunks.push(LlmResponseChunk::HostedToolCallDone {
                            index: tool_call_index,
                            call,
                        });
                        tool_call_index += 1;
                    }
                    ContentBlock::Refusal { text } => {
                        chunks.push(LlmResponseChunk::RefusalDelta {
                            index: output_index,
                            text: text.clone(),
                        });
                        chunks.push(LlmResponseChunk::RefusalDone {
                            index: output_index,
                            text,
                        });
                    }
                    ContentBlock::OpaqueState(state) => {
                        chunks.push(LlmResponseChunk::OpaqueState(state));
                    }
                    ContentBlock::Compaction(item) => {
                        chunks.push(LlmResponseChunk::CompactionDone(item));
                    }
                    ContentBlock::GeneratedImage(image) => {
                        chunks.push(LlmResponseChunk::GeneratedImageDone(image));
                    }
                    ContentBlock::PauseTurn(pause) => {
                        chunks.push(LlmResponseChunk::PauseTurn(pause));
                    }
                    // Results and input media do not appear in assistant output streams.
                    ContentBlock::Image { .. }
                    | ContentBlock::Audio { .. }
                    | ContentBlock::Video { .. }
                    | ContentBlock::File { .. }
                    | ContentBlock::ToolResult(_)
                    | ContentBlock::CustomToolResult(_)
                    | ContentBlock::ComputerToolResult(_)
                    | ContentBlock::HostedToolResult(_)
                    | ContentBlock::Unknown { .. } => {}
                }
            }
            chunks.push(LlmResponseChunk::MessageStop {
                reason: output.stop_reason.and_then(|r| {
                    serde_json::to_value(r)
                        .ok()
                        .and_then(|v| v.as_str().map(String::from))
                }),
            });
        }
        if !self.metadata.is_empty() {
            chunks.push(LlmResponseChunk::ResponseMetadata(self.metadata));
        }
        if let Some(terminal) = self.terminal {
            chunks.push(LlmResponseChunk::ResponseTerminal(terminal));
        }
        chunks.push(LlmResponseChunk::Usage(self.usage));
        Box::pin(futures::stream::iter(
            chunks.into_iter().map(|chunk| Ok(chunk.into())),
        ))
    }
}

fn push_checked_chunk(
    accumulator: &mut ResponseAccumulator,
    chunk: LlmResponseChunk,
) -> Result<(), LlmClientError> {
    match chunk {
        LlmResponseChunk::DecodeError { message } => {
            Err(LlmClientError::ResponseTranslation(message))
        }
        LlmResponseChunk::StreamError { message } => Err(LlmClientError::UpstreamHttp {
            status: MID_STREAM_UPSTREAM_STATUS,
            body: message,
        }),
        chunk => {
            accumulator.push(chunk);
            Ok(())
        }
    }
}

/// One provider-neutral streaming response chunk.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LlmResponseChunk {
    /// Starts a response message.
    MessageStart {
        /// Provider response identifier.
        id: Option<String>,
        /// Model reported by the provider.
        model: Option<String>,
    },
    /// Adds text to one output index.
    TextDelta {
        /// Provider output index.
        index: usize,
        /// Text fragment.
        text: String,
    },
    /// Adds reasoning text to one output index.
    ReasoningDelta {
        /// Provider output index.
        index: usize,
        /// Reasoning fragment.
        text: String,
    },
    /// Adds structured reasoning details to one output index.
    ReasoningDetailsDelta {
        /// Provider output index.
        index: usize,
        /// Reasoning detail objects in provider order.
        details: Vec<Value>,
        /// Normalized reasoning text represented by or accompanying the details.
        text: String,
    },
    /// Adds or updates a tool call at one index.
    ToolCallDelta {
        /// Tool-call index within the response.
        index: usize,
        /// Tool-call identifier, normally supplied by the first delta.
        id: Option<String>,
        /// Tool name, normally supplied by the first delta.
        name: Option<String>,
        /// Fragment of the serialized tool arguments.
        arguments_delta: Option<String>,
    },
    /// Finalizes a JSON-schema function tool call.
    ToolCallDone {
        /// Tool-call index within the response.
        index: usize,
        /// Complete tool call.
        call: ToolCall,
    },
    /// Adds free-form input to a custom tool call.
    CustomToolCallDelta {
        /// Tool-call index within the response.
        index: usize,
        /// Provider call identifier, normally supplied by the first delta.
        id: Option<String>,
        /// Provider item identifier, normally supplied by the first delta.
        item_id: Option<String>,
        /// Tool name, normally supplied by the first delta.
        name: Option<String>,
        /// Free-form input fragment.
        input_delta: String,
    },
    /// Finalizes a free-form input tool call.
    CustomToolCallDone {
        /// Tool-call index within the response.
        index: usize,
        /// Complete custom tool call.
        call: CustomToolCall,
    },
    /// Finalizes a computer tool call.
    ComputerToolCallDone {
        /// Tool-call index within the response.
        index: usize,
        /// Complete computer call.
        call: ComputerToolCall,
    },
    /// Finalizes a provider-hosted tool call.
    HostedToolCallDone {
        /// Tool-call index within the response.
        index: usize,
        /// Complete hosted call.
        call: HostedToolCall,
    },
    /// Starts one reasoning output item.
    ReasoningStarted {
        /// Provider output index.
        index: usize,
    },
    /// Finalizes one reasoning output item and its continuation state.
    ReasoningDone {
        /// Provider output index.
        index: usize,
        /// Complete reasoning text when the provider reports it only at completion.
        text: Option<String>,
        /// Bounded opaque state associated with the reasoning item.
        state: Vec<OpaqueState>,
    },
    /// Adds refusal text to one output index.
    RefusalDelta {
        /// Provider output index.
        index: usize,
        /// Refusal text fragment.
        text: String,
    },
    /// Finalizes a refusal.
    RefusalDone {
        /// Provider output index.
        index: usize,
        /// Complete refusal text.
        text: String,
    },
    /// Final text for providers that deliver completion text atomically.
    TextDone {
        /// Provider output index.
        index: usize,
        /// Complete text.
        text: String,
    },
    /// Bounded opaque provider state that does not itself commit semantic output.
    OpaqueState(OpaqueState),
    /// Nonsemantic response metadata.
    ResponseMetadata(ResponseMetadata),
    /// Final provider-native generated image.
    GeneratedImageDone(GeneratedImage),
    /// Final remote compaction result.
    CompactionDone(CompactionItem),
    /// Final pause-turn result.
    PauseTurn(PauseTurn),
    /// Explicit response terminal state.
    ResponseTerminal(ResponseTerminal),
    /// Reports token usage, normally near the end of the stream.
    Usage(Usage),
    /// Ends a response message.
    MessageStop {
        /// Provider stop-reason string before normalization.
        reason: Option<String>,
    },
    /// Reports that an inbound stream event could not be decoded.
    DecodeError {
        /// Human-readable decoding failure.
        message: String,
    },
    /// Reports an upstream failure delivered inside an otherwise successful stream.
    StreamError {
        /// Human-readable upstream failure.
        message: String,
    },
}

/// Whether a stream chunk commits downstream-visible response semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamSemanticClass {
    /// Metadata, lifecycle, or partial tool state that is safe to discard before commitment.
    NonSemantic,
    /// Output that prohibits replay or provider fallback.
    Semantic,
}

impl LlmResponseChunk {
    /// Classifies the chunk for the shared semantic-commitment boundary.
    pub fn semantic_class(&self) -> StreamSemanticClass {
        match self {
            Self::TextDelta { text, .. }
            | Self::ReasoningDelta { text, .. }
            | Self::ReasoningDetailsDelta { text, .. }
            | Self::RefusalDelta { text, .. }
            | Self::TextDone { text, .. }
                if !text.is_empty() =>
            {
                StreamSemanticClass::Semantic
            }
            Self::ReasoningDone {
                text: Some(text), ..
            } if !text.is_empty() => StreamSemanticClass::Semantic,
            Self::RefusalDone { .. }
            | Self::ToolCallDone { .. }
            | Self::CustomToolCallDone { .. }
            | Self::ComputerToolCallDone { .. }
            | Self::HostedToolCallDone { .. }
            | Self::GeneratedImageDone(_)
            | Self::CompactionDone(_)
            | Self::PauseTurn(_)
            | Self::MessageStop { .. }
            | Self::ResponseTerminal(ResponseTerminal::Completed)
            | Self::ResponseTerminal(ResponseTerminal::Paused(_)) => StreamSemanticClass::Semantic,
            _ => StreamSemanticClass::NonSemantic,
        }
    }

    /// Returns whether this chunk crosses the semantic-commitment boundary.
    pub fn commits_semantics(&self) -> bool {
        self.semantic_class() == StreamSemanticClass::Semantic
    }
}

/// Folds a sequence of [`LlmResponseChunk`]s into the terminal [`AggLlmResponse`].
///
/// Text and reasoning deltas concatenate; tool-call deltas assemble by index (name,
/// id, and a growing arguments string parsed as JSON at the end); `MessageStart`,
/// `Usage`, and `MessageStop` set the corresponding fields.
/// [`DecodeError`](LlmResponseChunk::DecodeError) and
/// [`StreamError`](LlmResponseChunk::StreamError) chunks are ignored here; use
/// [`LlmResponse::into_agg`] when they must become errors.
///
/// Folding is lossy for multiple outputs: text and reasoning indices are ignored,
/// and [`finish`](Self::finish) produces one assistant output. Prefer
/// [`LlmResponse::into_agg`] when stream errors must be surfaced.
///
/// Drive it by `push`-ing each chunk in order, then call [`finish`](Self::finish).
#[derive(Default)]
pub struct ResponseAccumulator {
    id: Option<String>,
    model: Option<String>,
    text: String,
    reasoning: Option<String>,
    reasoning_details: Vec<Value>,
    tool_calls: BTreeMap<usize, PartialToolCall>,
    custom_tool_calls: BTreeMap<usize, PartialCustomToolCall>,
    usage: Usage,
    stop_reason: Option<StopReason>,
    refusal: Option<String>,
    opaque_state: Vec<OpaqueState>,
    finalized_blocks: Vec<ContentBlock>,
    metadata: ResponseMetadata,
    terminal: Option<ResponseTerminal>,
}

/// A tool call being assembled from streamed [`LlmResponseChunk::ToolCallDelta`]s.
#[derive(Default)]
struct PartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// A custom tool call being assembled from streamed input deltas.
#[derive(Default)]
struct PartialCustomToolCall {
    id: Option<String>,
    item_id: Option<String>,
    name: Option<String>,
    input: String,
}

impl ResponseAccumulator {
    /// A fresh accumulator with no chunks applied.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one chunk. Later `MessageStart`/`Usage`/`MessageStop` fields overwrite
    /// earlier ones; text, reasoning, and tool-call arguments append.
    pub fn push(&mut self, chunk: LlmResponseChunk) {
        match chunk {
            LlmResponseChunk::MessageStart { id, model } => {
                if id.is_some() {
                    self.id = id;
                }
                if model.is_some() {
                    self.model = model;
                }
            }
            LlmResponseChunk::TextDelta { text, .. } => self.text.push_str(&text),
            LlmResponseChunk::TextDone { text, .. } => {
                if self.text.is_empty() {
                    self.text = text;
                }
            }
            LlmResponseChunk::ReasoningDelta { text, .. } => {
                self.reasoning
                    .get_or_insert_with(String::new)
                    .push_str(&text);
            }
            LlmResponseChunk::ReasoningDetailsDelta { details, text, .. } => {
                for detail in details {
                    push_reasoning_detail(&mut self.reasoning_details, detail);
                }
                if !text.is_empty() {
                    self.reasoning
                        .get_or_insert_with(String::new)
                        .push_str(&text);
                }
            }
            LlmResponseChunk::ReasoningStarted { .. } => {}
            LlmResponseChunk::ReasoningDone { text, state, .. } => {
                if self.reasoning.as_deref().is_none_or(str::is_empty)
                    && let Some(text) = text
                {
                    self.reasoning = Some(text);
                }
                self.opaque_state.extend(state);
            }
            LlmResponseChunk::RefusalDelta { text, .. } => {
                self.refusal.get_or_insert_with(String::new).push_str(&text);
            }
            LlmResponseChunk::RefusalDone { text, .. } => self.refusal = Some(text),
            LlmResponseChunk::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => {
                let call = self.tool_calls.entry(index).or_default();
                if id.is_some() {
                    call.id = id;
                }
                if name.is_some() {
                    call.name = name;
                }
                if let Some(delta) = arguments_delta {
                    call.arguments.push_str(&delta);
                }
            }
            LlmResponseChunk::ToolCallDone { index, call } => {
                self.tool_calls.remove(&index);
                self.finalized_blocks.push(ContentBlock::ToolCall(call));
            }
            LlmResponseChunk::CustomToolCallDelta {
                index,
                id,
                item_id,
                name,
                input_delta,
            } => {
                let call = self.custom_tool_calls.entry(index).or_default();
                if id.is_some() {
                    call.id = id;
                }
                if item_id.is_some() {
                    call.item_id = item_id;
                }
                if name.is_some() {
                    call.name = name;
                }
                call.input.push_str(&input_delta);
            }
            LlmResponseChunk::CustomToolCallDone { index, call } => {
                self.custom_tool_calls.remove(&index);
                self.finalized_blocks
                    .push(ContentBlock::CustomToolCall(call));
            }
            LlmResponseChunk::ComputerToolCallDone { call, .. } => self
                .finalized_blocks
                .push(ContentBlock::ComputerToolCall(call)),
            LlmResponseChunk::HostedToolCallDone { call, .. } => self
                .finalized_blocks
                .push(ContentBlock::HostedToolCall(call)),
            LlmResponseChunk::OpaqueState(state) => self.opaque_state.push(state),
            LlmResponseChunk::ResponseMetadata(metadata) => {
                if metadata.model_etag.is_some() {
                    self.metadata.model_etag = metadata.model_etag;
                }
                if metadata.turn_state.is_some() {
                    self.metadata.turn_state = metadata.turn_state;
                }
            }
            LlmResponseChunk::GeneratedImageDone(image) => self
                .finalized_blocks
                .push(ContentBlock::GeneratedImage(image)),
            LlmResponseChunk::CompactionDone(item) => {
                self.finalized_blocks.push(ContentBlock::Compaction(item));
            }
            LlmResponseChunk::PauseTurn(pause) => {
                self.finalized_blocks.push(ContentBlock::PauseTurn(pause));
            }
            LlmResponseChunk::ResponseTerminal(terminal) => self.terminal = Some(terminal),
            LlmResponseChunk::Usage(usage) => self.usage = usage,
            LlmResponseChunk::MessageStop { reason } => {
                self.stop_reason = Some(stop_reason_from_str(reason.as_deref()));
            }
            LlmResponseChunk::DecodeError { .. } | LlmResponseChunk::StreamError { .. } => {}
        }
    }

    /// Build the buffered response. Content is ordered reasoning, text, refusal,
    /// partial calls by index, then finalized typed items.
    pub fn finish(self) -> AggLlmResponse {
        let mut content = Vec::new();
        if self.reasoning.is_some() || !self.reasoning_details.is_empty() {
            content.push(ContentBlock::Reasoning {
                text: self.reasoning.unwrap_or_default(),
                signature: None,
                // An announcement whose payload never arrived is a stream-level hint only.
                details: self
                    .reasoning_details
                    .into_iter()
                    .filter(|detail| !is_reasoning_id_announcement(detail))
                    .collect(),
                openai_chat_field: Default::default(),
            });
        }
        if !self.text.is_empty() {
            content.push(ContentBlock::Text { text: self.text });
        }
        if let Some(text) = self.refusal {
            content.push(ContentBlock::Refusal { text });
        }
        for call in self.tool_calls.into_values() {
            content.push(ContentBlock::ToolCall(ToolCall {
                id: call.id.unwrap_or_default(),
                name: call.name.unwrap_or_default(),
                arguments: parse_tool_arguments(&call.arguments),
            }));
        }
        for call in self.custom_tool_calls.into_values() {
            content.push(ContentBlock::CustomToolCall(CustomToolCall {
                id: call.id.unwrap_or_default(),
                item_id: call.item_id,
                name: call.name.unwrap_or_default(),
                input: call.input,
            }));
        }
        content.extend(self.opaque_state.into_iter().map(ContentBlock::OpaqueState));
        content.extend(self.finalized_blocks);
        AggLlmResponse {
            id: self.id,
            model: self.model,
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content,
                stop_reason: self.stop_reason,
            }],
            usage: self.usage,
            metadata: self.metadata,
            terminal: self.terminal,
            ..AggLlmResponse::default()
        }
    }
}

/// Parse an assembled tool-call arguments string as JSON, falling back to a JSON
/// string when it is not valid JSON and to an empty object when it is empty.
fn parse_tool_arguments(arguments: &str) -> Value {
    if arguments.is_empty() {
        return Value::Object(serde_json::Map::new());
    }
    serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_string()))
}

/// Map a provider stop-reason string (as carried by [`LlmResponseChunk::MessageStop`])
/// to a normalized [`StopReason`], covering the common OpenAI and Anthropic spellings.
fn stop_reason_from_str(reason: Option<&str>) -> StopReason {
    match reason {
        Some("length" | "max_tokens") => StopReason::MaxTokens,
        Some("tool_calls" | "function_call" | "tool_use") => StopReason::ToolUse,
        Some("content_filter") => StopReason::ContentFilter,
        Some("stop" | "end_turn" | "stop_sequence") | None => StopReason::EndTurn,
        Some(_) => StopReason::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use futures::stream;

    use super::*;
    use serde_json::json;

    fn fold(chunks: Vec<LlmResponseChunk>) -> AggLlmResponse {
        let mut accumulator = ResponseAccumulator::new();
        for chunk in chunks {
            accumulator.push(chunk);
        }
        accumulator.finish()
    }

    #[test]
    fn folds_text_usage_and_stop_reason() {
        let agg = fold(vec![
            LlmResponseChunk::MessageStart {
                id: Some("id1".to_string()),
                model: Some("m".to_string()),
            },
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "Hel".to_string(),
            },
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "lo".to_string(),
            },
            LlmResponseChunk::Usage(Usage {
                output_tokens: Some(2),
                ..Usage::default()
            }),
            LlmResponseChunk::MessageStop {
                reason: Some("length".to_string()),
            },
        ]);
        assert_eq!(agg.id.as_deref(), Some("id1"));
        assert_eq!(agg.model.as_deref(), Some("m"));
        assert_eq!(agg.usage.output_tokens, Some(2));
        assert_eq!(agg.outputs[0].stop_reason, Some(StopReason::MaxTokens));
        assert_eq!(
            agg.outputs[0].content,
            vec![ContentBlock::Text {
                text: "Hello".to_string()
            }]
        );
    }

    #[test]
    fn aggregates_normalized_chunks_inside_stream_event() {
        let response = LlmResponse::Stream(Box::pin(stream::iter([Ok(
            LlmResponseStreamEvent::preserved(
                crate::WireFormat::OpenAiChat,
                json!({
                "choices": [{"delta": {"content": "hello"}}],
                "system_fingerprint": "fp_exact"
                }),
                vec![LlmResponseChunk::TextDelta {
                    index: 0,
                    text: "hello".to_string(),
                }],
            ),
        )])));
        let aggregate = block_on(response.into_agg()).expect("stream event should aggregate");

        assert_eq!(
            aggregate.outputs[0].content,
            vec![ContentBlock::Text {
                text: "hello".to_string()
            }]
        );
    }

    #[test]
    fn replacing_normalized_content_drops_preservation() {
        let event = LlmResponseStreamEvent::preserved(
            crate::WireFormat::OpenAiChat,
            json!({"choices": [{"delta": {"content": "old"}}]}),
            vec![LlmResponseChunk::TextDelta {
                index: 0,
                text: "old".to_string(),
            }],
        )
        .replace_normalized(vec![LlmResponseChunk::TextDelta {
            index: 0,
            text: "new".to_string(),
        }]);

        assert!(event.preservation().is_none());
        assert_eq!(
            event.normalized(),
            &[LlmResponseChunk::TextDelta {
                index: 0,
                text: "new".to_string(),
            }]
        );
    }

    #[test]
    fn stream_errors_inside_preserved_events_remain_typed() {
        let response = LlmResponse::Stream(Box::pin(stream::iter([Ok(
            LlmResponseStreamEvent::preserved(
                crate::WireFormat::OpenAiChat,
                json!({"error": {"message": "provider failed"}}),
                vec![LlmResponseChunk::StreamError {
                    message: "provider failed".to_string(),
                }],
            ),
        )])));

        let error = block_on(response.into_agg()).err();
        assert!(matches!(
            error,
            Some(LlmClientError::UpstreamHttp {
                status: MID_STREAM_UPSTREAM_STATUS,
                ..
            })
        ));
    }

    #[test]
    fn assembles_tool_calls_by_index() {
        // id/name arrive once, arguments stream across deltas and parse as JSON.
        let agg = fold(vec![
            LlmResponseChunk::ToolCallDelta {
                index: 0,
                id: Some("call_1".to_string()),
                name: Some("lookup".to_string()),
                arguments_delta: Some("{\"q\":".to_string()),
            },
            LlmResponseChunk::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: Some("\"rust\"}".to_string()),
            },
            LlmResponseChunk::MessageStop {
                reason: Some("tool_calls".to_string()),
            },
        ]);
        assert_eq!(agg.outputs[0].stop_reason, Some(StopReason::ToolUse));
        assert_eq!(
            agg.outputs[0].content,
            vec![ContentBlock::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "lookup".to_string(),
                arguments: json!({"q": "rust"}),
            })]
        );
    }

    #[test]
    fn reasoning_precedes_text_in_content() {
        let agg = fold(vec![
            LlmResponseChunk::ReasoningDelta {
                index: 0,
                text: "think".to_string(),
            },
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "answer".to_string(),
            },
        ]);
        assert_eq!(
            agg.outputs[0].content,
            vec![
                ContentBlock::Reasoning {
                    text: "think".to_string(),
                    signature: None,
                    details: Vec::new(),
                    openai_chat_field: Default::default(),
                },
                ContentBlock::Text {
                    text: "answer".to_string(),
                },
            ]
        );
    }

    #[test]
    fn into_stream_round_trips_through_into_agg() {
        let original = AggLlmResponse {
            id: Some("id1".to_string()),
            model: Some("m".to_string()),
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "hello".to_string(),
                }],
                stop_reason: Some(StopReason::EndTurn),
            }],
            usage: Usage {
                output_tokens: Some(3),
                ..Usage::default()
            },
            ..AggLlmResponse::default()
        };
        let stream = LlmResponse::Stream(original.clone().into_stream());
        let recovered = block_on(stream.into_agg()).expect("into_agg failed");
        assert_eq!(recovered.id, original.id);
        assert_eq!(recovered.model, original.model);
        assert_eq!(recovered.usage.output_tokens, original.usage.output_tokens);
        assert_eq!(
            recovered.outputs[0].stop_reason,
            original.outputs[0].stop_reason
        );
        assert_eq!(recovered.outputs[0].content, original.outputs[0].content);
    }

    #[test]
    fn into_stream_retains_encrypted_reasoning_and_text() {
        let details = vec![json!({
            "type": "reasoning.encrypted",
            "data": "opaque-encrypted-reasoning"
        })];
        let original = AggLlmResponse {
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::Reasoning {
                    text: "fallback reasoning".to_string(),
                    signature: None,
                    details: details.clone(),
                    openai_chat_field: Default::default(),
                }],
                stop_reason: Some(StopReason::EndTurn),
            }],
            ..AggLlmResponse::default()
        };

        let recovered = block_on(LlmResponse::Stream(original.into_stream()).into_agg())
            .expect("into_agg failed");
        let ContentBlock::Reasoning {
            text,
            details: recovered_details,
            ..
        } = &recovered.outputs[0].content[0]
        else {
            panic!("expected reasoning block");
        };
        assert_eq!(text, "fallback reasoning");
        assert_eq!(recovered_details, &details);
    }

    #[test]
    fn into_agg_preserves_stream_item_error() {
        let response = LlmResponse::Stream(Box::pin(stream::once(async {
            Err(LlmClientError::Timeout {
                source: Box::new(std::io::Error::other("timed out")),
            })
        })));

        let Err(error) = block_on(response.into_agg()) else {
            panic!("expected stream aggregation to fail");
        };
        assert!(matches!(error, LlmClientError::Timeout { .. }));
    }
}
