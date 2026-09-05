// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Amazon Bedrock ConverseStream event payload codec.
//!
//! AWS EventStream carrier framing is intentionally host-owned. This codec consumes and
//! produces the JSON payload carried by each framed EventStream message.

use serde_json::{Map, Value, json};

use crate::LlmResponseChunk;
use crate::codecs::stream::{StreamCodec, StreamTranslationState, record_source_identity};
use crate::format::{FormatId, WireFormat};

/// Stream codec for Bedrock ConverseStream JSON event payloads.
pub struct BedrockConverseStreamCodec;

impl StreamCodec for BedrockConverseStreamCodec {
    fn format(&self) -> FormatId {
        WireFormat::BedrockConverse.into()
    }

    fn decode_event(
        &self,
        state: &mut StreamTranslationState,
        event: &Value,
    ) -> Vec<LlmResponseChunk> {
        decode_bedrock_event(state, event)
    }

    fn encode_event(
        &self,
        state: &mut StreamTranslationState,
        event: LlmResponseChunk,
    ) -> Vec<Value> {
        encode_bedrock_event(state, event)
    }

    fn finish(&self, state: &mut StreamTranslationState) -> Vec<Value> {
        finish_bedrock_stream(state)
    }
}

fn decode_bedrock_event(
    state: &mut StreamTranslationState,
    event: &Value,
) -> Vec<LlmResponseChunk> {
    let Some(object) = event.as_object() else {
        return vec![LlmResponseChunk::DecodeError {
            message: "Bedrock stream event is not an object".to_string(),
        }];
    };
    if object.get("messageStart").is_some() {
        state.saw_message_start = true;
        return vec![LlmResponseChunk::MessageStart {
            id: None,
            model: None,
        }];
    }
    if let Some(start) = object.get("contentBlockStart").and_then(Value::as_object) {
        return decode_content_start(start);
    }
    if let Some(delta) = object.get("contentBlockDelta").and_then(Value::as_object) {
        return decode_content_delta(delta);
    }
    if let Some(stop) = object.get("messageStop").and_then(Value::as_object) {
        let reason = stop
            .get("stopReason")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        state.stop_reason = reason.clone();
        return vec![LlmResponseChunk::MessageStop { reason }];
    }
    if let Some(metadata) = object.get("metadata").and_then(Value::as_object)
        && let Some(usage) = metadata.get("usage")
    {
        capture_usage(state, usage);
        return vec![LlmResponseChunk::Usage(state.usage.clone())];
    }
    if let Some((name, _)) = object.iter().find(|(name, _)| name.ends_with("Exception")) {
        return vec![LlmResponseChunk::StreamError {
            message: format!("Bedrock stream failed with {name}"),
        }];
    }
    Vec::new()
}

fn decode_content_start(object: &Map<String, Value>) -> Vec<LlmResponseChunk> {
    let index = object
        .get("contentBlockIndex")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let start = object.get("start").and_then(Value::as_object);
    let tool = start
        .and_then(|start| start.get("toolUse"))
        .and_then(Value::as_object);
    tool.map(|tool| {
        vec![LlmResponseChunk::ToolCallDelta {
            index,
            id: tool
                .get("toolUseId")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            name: tool
                .get("name")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            arguments_delta: None,
        }]
    })
    .unwrap_or_default()
}

fn decode_content_delta(object: &Map<String, Value>) -> Vec<LlmResponseChunk> {
    let index = object
        .get("contentBlockIndex")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let Some(delta) = object.get("delta").and_then(Value::as_object) else {
        return Vec::new();
    };
    if let Some(text) = delta.get("text").and_then(Value::as_str) {
        return vec![LlmResponseChunk::TextDelta {
            index,
            text: text.to_string(),
        }];
    }
    if let Some(input) = delta
        .get("toolUse")
        .and_then(Value::as_object)
        .and_then(|tool| tool.get("input"))
        .and_then(Value::as_str)
    {
        return vec![LlmResponseChunk::ToolCallDelta {
            index,
            id: None,
            name: None,
            arguments_delta: Some(input.to_string()),
        }];
    }
    if let Some(reasoning) = delta.get("reasoningContent").and_then(Value::as_object)
        && let Some(text) = reasoning.get("text").and_then(Value::as_str)
    {
        return vec![LlmResponseChunk::ReasoningDelta {
            index,
            text: text.to_string(),
        }];
    }
    Vec::new()
}

fn encode_bedrock_event(state: &mut StreamTranslationState, event: LlmResponseChunk) -> Vec<Value> {
    if state.errored {
        return Vec::new();
    }
    match event {
        LlmResponseChunk::MessageStart { id, model } => {
            record_source_identity(state, id, model);
            if state.emitted_message_start {
                Vec::new()
            } else {
                state.emitted_message_start = true;
                vec![json!({"messageStart": {"role": "assistant"}})]
            }
        }
        LlmResponseChunk::TextDelta { text, .. } => {
            state.output_tokens_seen += 1;
            let index = ensure_text_block(state);
            vec![json!({
                "contentBlockDelta": {
                    "contentBlockIndex": index,
                    "delta": {"text": text}
                }
            })]
        }
        // Bedrock Converse has no wire slot for opaque reasoning details, so only the
        // normalized text that accompanies them survives, on the reasoning-delta path.
        LlmResponseChunk::ReasoningDelta { text, .. }
        | LlmResponseChunk::ReasoningDetailsDelta { text, .. } => {
            if text.is_empty() {
                return Vec::new();
            }
            let index = ensure_reasoning_block(state);
            vec![json!({
                "contentBlockDelta": {
                    "contentBlockIndex": index,
                    "delta": {"reasoningContent": {"text": text}}
                }
            })]
        }
        LlmResponseChunk::ToolCallDelta {
            index,
            id,
            name,
            arguments_delta,
        } => encode_tool_delta(state, index, id, name, arguments_delta),
        LlmResponseChunk::Usage(usage) => {
            state.usage = usage;
            state.saw_backend_usage = true;
            Vec::new()
        }
        LlmResponseChunk::MessageStop { reason } => {
            state.stop_reason = reason.or_else(|| state.stop_reason.clone());
            Vec::new()
        }
        LlmResponseChunk::StreamError { message } | LlmResponseChunk::DecodeError { message } => {
            state.finished = true;
            state.errored = true;
            vec![json!({"modelStreamErrorException": {"message": message}})]
        }
    }
}

fn ensure_text_block(state: &mut StreamTranslationState) -> usize {
    if let Some(index) = state.text_block_index {
        return index;
    }
    let index = state.next_content_index;
    state.next_content_index += 1;
    state.text_block_index = Some(index);
    state.text_block_started = true;
    state.emitted_content_block = true;
    index
}

fn ensure_reasoning_block(state: &mut StreamTranslationState) -> usize {
    if let Some(index) = state.reasoning_block_index {
        return index;
    }
    let index = state.next_content_index;
    state.next_content_index += 1;
    state.reasoning_block_index = Some(index);
    state.reasoning_block_started = true;
    state.emitted_content_block = true;
    index
}

fn encode_tool_delta(
    state: &mut StreamTranslationState,
    index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments_delta: Option<String>,
) -> Vec<Value> {
    let tool = state.tool_states.entry(index).or_default();
    if id.is_some() {
        tool.id = id;
    }
    if name.is_some() {
        tool.name = name;
    }
    let mut out = Vec::new();
    if !tool.started {
        let Some(name) = tool.name.clone() else {
            if let Some(delta) = arguments_delta {
                tool.pending_arguments.push_str(&delta);
            }
            return out;
        };
        let content_index = state.next_content_index;
        state.next_content_index += 1;
        tool.content_index = Some(content_index);
        tool.started = true;
        state.emitted_content_block = true;
        out.push(json!({
            "contentBlockStart": {
                "contentBlockIndex": content_index,
                "start": {"toolUse": {
                    "toolUseId": tool.id.clone().unwrap_or_default(),
                    "name": name
                }}
            }
        }));
    }
    if let Some(delta) = arguments_delta {
        tool.arguments.push_str(&delta);
        out.push(json!({
            "contentBlockDelta": {
                "contentBlockIndex": tool.content_index.unwrap_or(index),
                "delta": {"toolUse": {"input": delta}}
            }
        }));
    }
    out
}

fn finish_bedrock_stream(state: &mut StreamTranslationState) -> Vec<Value> {
    if state.finished {
        return Vec::new();
    }
    let mut out = Vec::new();
    if !state.emitted_message_start {
        out.push(json!({"messageStart": {"role": "assistant"}}));
        state.emitted_message_start = true;
    }
    if let Some(index) = state.text_block_index.take() {
        out.push(json!({"contentBlockStop": {"contentBlockIndex": index}}));
    }
    if let Some(index) = state.reasoning_block_index.take() {
        out.push(json!({"contentBlockStop": {"contentBlockIndex": index}}));
    }
    for tool in state.tool_states.values_mut() {
        if tool.started {
            if let Some(index) = tool.content_index {
                out.push(json!({"contentBlockStop": {"contentBlockIndex": index}}));
            }
            tool.started = false;
        }
    }
    out.push(json!({
        "messageStop": {"stopReason": bedrock_stop_reason(state.stop_reason.as_deref())}
    }));
    out.push(json!({"metadata": {"usage": encode_usage(state)}}));
    state.finished = true;
    out
}

fn capture_usage(state: &mut StreamTranslationState, value: &Value) {
    let value = value.as_object();
    state.usage.input_tokens = value
        .and_then(|value| value.get("inputTokens"))
        .and_then(Value::as_u64);
    state.usage.output_tokens = value
        .and_then(|value| value.get("outputTokens"))
        .and_then(Value::as_u64);
    state.usage.total_tokens = value
        .and_then(|value| value.get("totalTokens"))
        .and_then(Value::as_u64);
    if let Some(tokens) = value
        .and_then(|value| value.get("cacheReadInputTokens"))
        .and_then(Value::as_u64)
    {
        state.usage.set_cached_input_tokens(tokens);
    }
    if let Some(tokens) = value
        .and_then(|value| value.get("cacheWriteInputTokens"))
        .and_then(Value::as_u64)
    {
        state.usage.set_cache_creation_input_tokens(tokens);
    }
    state.saw_backend_usage = true;
}

fn encode_usage(state: &StreamTranslationState) -> Value {
    let output_tokens = state
        .usage
        .output_tokens
        .unwrap_or(state.output_tokens_seen);
    let input_tokens = state.usage.input_tokens.unwrap_or(0);
    json!({
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": state
            .usage
            .total_tokens
            .unwrap_or(input_tokens.saturating_add(output_tokens)),
        "cacheReadInputTokens": state.usage.cached_input_tokens(),
        "cacheWriteInputTokens": state.usage.cache_creation_input_tokens(),
    })
}

fn bedrock_stop_reason(reason: Option<&str>) -> &'static str {
    match reason {
        Some("max_tokens" | "length") => "max_tokens",
        Some("tool_use" | "tool_calls") => "tool_use",
        Some("content_filter" | "content_filtered" | "guardrail_intervened") => "content_filtered",
        _ => "end_turn",
    }
}
