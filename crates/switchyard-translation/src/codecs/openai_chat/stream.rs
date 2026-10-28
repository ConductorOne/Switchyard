// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming codec for OpenAI Chat Completions chunks.

use serde_json::{Map, Value, json};

use super::{decode_google_thought_signature, encode_google_thought_signature};

use super::buffered::{decode_openai_usage, encode_openai_usage};
use crate::LlmResponseChunk;
use crate::codecs::common::{first_nonempty_string, reasoning_text_from_details};
use crate::codecs::stream::{
    StreamCodec, StreamTranslationState, record_source_identity, string_field,
    target_message_id_or_source_message_id, target_model_or_source_model,
};
use crate::format::{FormatId, WireFormat};

// Chat SDKs require a function object and append string fields. Keep other raw fields.
pub(crate) fn prepare_google_tool_replay(state: &StreamTranslationState, event: &mut Value) {
    for choice in event
        .get_mut("choices")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
    {
        let Some(calls) = choice
            .get_mut("delta")
            .and_then(|delta| delta.get_mut("tool_calls"))
            .and_then(Value::as_array_mut)
        else {
            continue;
        };
        for position in 0..calls.len() {
            // Borrow the retained first metadata instead of copying per-event state.
            let (earlier, current) = calls.split_at_mut(position);
            let (call, later) = current.split_first_mut().unwrap();
            let Some(index) = call["index"].as_u64() else {
                prepare_unindexed_google_call(state, earlier, call);
                continue;
            };
            let signature = call["extra_content"]["google"]["thought_signature"].as_str();
            let id = call.get("id").and_then(Value::as_str);
            let tool = state.tool_states.get(&(index as usize));
            let mut signed = signature.is_some()
                || tool
                    .and_then(|tool| tool.google_thought_signature.as_ref())
                    .is_some();
            let mut repeated = signature.is_some()
                && tool.and_then(|tool| tool.google_thought_signature.as_deref()) == signature;
            let mut repeated_id = id.is_some() && tool.and_then(|tool| tool.id.as_deref()) == id;
            for previous in earlier
                .iter()
                .rev()
                .filter(|call| call["index"].as_u64() == Some(index))
            {
                let previous_signature =
                    previous["extra_content"]["google"]["thought_signature"].as_str();
                signed |= previous_signature.is_some();
                repeated |= signature.is_some() && previous_signature == signature;
                repeated_id |= id.is_some() && previous.get("id").and_then(Value::as_str) == id;
                if signed && repeated_id && (signature.is_none() || repeated) {
                    break;
                }
            }
            if !signed {
                signed = later.iter().any(|call| {
                    call["index"].as_u64() == Some(index)
                        && call["extra_content"]["google"]["thought_signature"].is_string()
                });
            }
            if !signed {
                continue;
            }
            let Some(call) = call.as_object_mut() else {
                continue;
            };
            if !call.contains_key("function") {
                call.insert("function".to_string(), Value::Object(Map::new()));
            }
            if repeated_id {
                call.remove("id");
            }
            if !repeated {
                continue;
            }
            if let Some(extra) = call.get_mut("extra_content").and_then(Value::as_object_mut) {
                if let Some(google) = extra.get_mut("google").and_then(Value::as_object_mut) {
                    google.remove("thought_signature");
                    if google.is_empty() {
                        extra.remove("google");
                    }
                }
                if extra.is_empty() {
                    call.remove("extra_content");
                }
            }
        }
    }
}

// Google omits `index`, so the call id is the only identity and stays on every update.
fn prepare_unindexed_google_call(
    state: &StreamTranslationState,
    earlier: &[Value],
    call: &mut Value,
) {
    let Some(id) = call.get("id").and_then(Value::as_str).map(str::to_owned) else {
        return;
    };
    let signature = call["extra_content"]["google"]["thought_signature"]
        .as_str()
        .map(str::to_owned);
    let mut seen = state
        .tool_states
        .values()
        .filter(|tool| tool.id.as_deref() == Some(id.as_str()))
        .filter_map(|tool| tool.google_thought_signature.as_deref())
        .chain(
            earlier
                .iter()
                .filter(|previous| {
                    previous["index"].is_null()
                        && previous.get("id").and_then(Value::as_str) == Some(id.as_str())
                })
                .filter_map(|previous| {
                    previous["extra_content"]["google"]["thought_signature"].as_str()
                }),
        );
    let (signed, repeated) = match signature.as_deref() {
        Some(signature) => (true, seen.any(|previous| previous == signature)),
        None => (seen.next().is_some(), false),
    };
    if !signed {
        return;
    }
    let Some(call) = call.as_object_mut() else {
        return;
    };
    if !call.contains_key("function") {
        call.insert("function".to_string(), Value::Object(Map::new()));
    }
    if !repeated {
        return;
    }
    if let Some(extra) = call.get_mut("extra_content").and_then(Value::as_object_mut) {
        if let Some(google) = extra.get_mut("google").and_then(Value::as_object_mut) {
            google.remove("thought_signature");
            if google.is_empty() {
                extra.remove("google");
            }
        }
        if extra.is_empty() {
            call.remove("extra_content");
        }
    }
}

/// Stream codec for OpenAI Chat Completions chunks.
pub struct OpenAiChatStreamCodec;

impl StreamCodec for OpenAiChatStreamCodec {
    fn format(&self) -> FormatId {
        WireFormat::OpenAiChat.into()
    }

    fn decode_event(
        &self,
        state: &mut StreamTranslationState,
        event: &Value,
    ) -> Vec<LlmResponseChunk> {
        decode_openai_chat_stream(state, event)
    }

    fn encode_event(
        &self,
        state: &mut StreamTranslationState,
        event: LlmResponseChunk,
    ) -> Vec<Value> {
        encode_openai_chat_stream(state, event)
    }

    fn finish(&self, state: &mut StreamTranslationState) -> Vec<Value> {
        finish_openai_chat_stream(state)
    }
}

// Decodes one OpenAI Chat chunk into neutral streaming events.
fn decode_openai_chat_stream(
    state: &mut StreamTranslationState,
    event: &Value,
) -> Vec<LlmResponseChunk> {
    let Some(object) = event.as_object() else {
        return vec![LlmResponseChunk::DecodeError {
            message: "OpenAI stream event is not an object".to_string(),
        }];
    };

    // Upstream error frames carry no choices, so without this they would decode to a bare
    // `MessageStart` and the error text would be dropped.
    if let Some(error) = object.get("error") {
        return vec![LlmResponseChunk::StreamError {
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown OpenAI stream error")
                .to_string(),
        }];
    }

    // Choice indexes share the old decoder's tool map. Never attach Google state
    // after another choice could have overwritten a call at the same index.
    let other_choice = object
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| {
            choices.len() > 1
                || choices
                    .iter()
                    .any(|choice| choice["index"].as_u64().unwrap_or(0) != 0)
        });
    state.chat_other_choice_seen |= other_choice;
    if other_choice
        && state
            .tool_states
            .values()
            .any(|tool| tool.decoded_google_thought_signature.is_some())
    {
        return vec![LlmResponseChunk::DecodeError {
            message: "Google tool signature requires one Chat choice".to_string(),
        }];
    }

    let mut out = Vec::new();
    let mut identity_changed = false;
    if state.model.is_none() {
        state.model = string_field(object, "model");
        identity_changed |= state.model.is_some();
    }
    if state.message_id.is_none() {
        state.message_id = string_field(object, "id");
        identity_changed |= state.message_id.is_some();
    }
    if !state.saw_message_start || identity_changed {
        state.saw_message_start = true;
        out.push(LlmResponseChunk::MessageStart {
            id: state.message_id.clone(),
            model: state.model.clone(),
        });
    }

    if let Some(usage) = object.get("usage").filter(|usage| usage.is_object()) {
        let usage = decode_openai_usage(Some(usage));
        state.usage = usage.clone();
        state.saw_backend_usage = true;
        out.push(LlmResponseChunk::Usage(usage));
    }

    for choice in object
        .get("choices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(choice) = choice.as_object() else {
            continue;
        };
        if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
            if let Some(details) = delta
                .get("reasoning_details")
                .and_then(Value::as_array)
                .filter(|details| !details.is_empty())
            {
                let text = reasoning_text_from_details(details)
                    .or_else(|| {
                        first_nonempty_string(delta, &["reasoning_content", "reasoning"])
                            .map(ToOwned::to_owned)
                    })
                    .unwrap_or_default();
                out.push(LlmResponseChunk::ReasoningDetailsDelta {
                    index: 0,
                    details: details.clone(),
                    text,
                });
            } else {
                for reasoning_key in ["reasoning_content", "reasoning"] {
                    if let Some(text) = delta.get(reasoning_key).and_then(Value::as_str)
                        && !text.is_empty()
                    {
                        out.push(LlmResponseChunk::ReasoningDelta {
                            index: 0,
                            text: text.to_string(),
                        });
                    }
                }
            }
            if let Some(text) = delta.get("content").and_then(Value::as_str)
                && !text.is_empty()
            {
                out.push(LlmResponseChunk::TextDelta {
                    index: 0,
                    text: text.to_string(),
                });
            }
            if let Some(text) = delta.get("refusal").and_then(Value::as_str)
                && !text.is_empty()
            {
                // Refusal deltas mark the whole message as a content-filter stop. This state
                // persists across chunks so a later `finish_reason: "stop"` cannot overwrite it.
                state.stop_reason = Some("content_filter".to_string());
                out.push(LlmResponseChunk::TextDelta {
                    index: 0,
                    text: text.to_string(),
                });
            }
            if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for tool_call in tool_calls {
                    if let Some(tool_call) = tool_call.as_object() {
                        let function = tool_call.get("function").and_then(Value::as_object);
                        let id = tool_call
                            .get("id")
                            .and_then(Value::as_str)
                            .filter(|id| !id.is_empty());
                        let explicit_index = tool_call.get("index").and_then(Value::as_u64);
                        let index = match (explicit_index, id) {
                            (Some(index), _) => index as usize,
                            // Google's Chat endpoint streams each call whole and omits `index`;
                            // the call id is then the only identity, so key calls by it.
                            (None, Some(id)) => {
                                match unindexed_chat_tool_index(state, id, function) {
                                    Ok(index) => index,
                                    Err(message) => {
                                        return vec![LlmResponseChunk::DecodeError {
                                            message: message.to_string(),
                                        }];
                                    }
                                }
                            }
                            // Once calls are keyed by id, a fragment with neither cannot be placed.
                            (None, None) if state.chat_unindexed_tool_calls => {
                                return vec![LlmResponseChunk::DecodeError {
                                    message: "Chat tool call has no index or id".to_string(),
                                }];
                            }
                            (None, None) => 0,
                        };
                        let tool = state.tool_states.entry(index).or_default();
                        if let Some(id) = id {
                            tool.decoded_chat_tool_id_ambiguous |= tool
                                .decoded_chat_tool_id
                                .as_deref()
                                .is_some_and(|previous| previous != id);
                        }
                        if tool.decoded_chat_tool_id_ambiguous
                            && (tool.decoded_google_thought_signature.is_some()
                                || tool_call
                                    .get("extra_content")
                                    .and_then(|extra| extra.get("google"))
                                    .and_then(|google| google.get("thought_signature"))
                                    .is_some())
                        {
                            return vec![LlmResponseChunk::DecodeError {
                                message: "conflicting Chat tool call identity".to_string(),
                            }];
                        }
                        if let Some(id) = id
                            && tool.decoded_chat_tool_id.as_deref() != Some(id)
                        {
                            tool.decoded_chat_tool_id = Some(id.to_owned());
                        }
                        let google_thought_signature = match decode_google_thought_signature(
                            tool_call,
                        ) {
                            Ok(Some(signature)) => {
                                if (explicit_index.is_none() && id.is_none())
                                    || state.chat_other_choice_seen
                                    || choice.get("index").and_then(Value::as_u64).unwrap_or(0) != 0
                                {
                                    return vec![LlmResponseChunk::DecodeError {
                                            message: "Google tool signature requires an identified call in one Chat choice".to_string(),
                                        }];
                                }
                                let tool = state.tool_states.entry(index).or_default();
                                match tool.decoded_google_thought_signature.as_deref() {
                                    Some(previous) if previous != signature => {
                                        return vec![LlmResponseChunk::DecodeError {
                                            message: "conflicting Google tool signature"
                                                .to_string(),
                                        }];
                                    }
                                    Some(_) => None,
                                    None => {
                                        tool.decoded_google_thought_signature =
                                            Some(signature.to_owned());
                                        Some(signature.to_owned())
                                    }
                                }
                            }
                            Ok(None) => None,
                            Err(error) => {
                                return vec![LlmResponseChunk::DecodeError {
                                    message: error.to_string(),
                                }];
                            }
                        };
                        if let Some(name) = function
                            .and_then(|function| function.get("name"))
                            .and_then(Value::as_str)
                            .filter(|name| !name.is_empty())
                        {
                            state
                                .pending_chat_tool_names
                                .entry(index)
                                .or_default()
                                .push_str(name);
                        }
                        let arguments_delta = function
                            .and_then(|function| function.get("arguments"))
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned);
                        state.chat_tool_call_decoded = true;
                        out.push(LlmResponseChunk::ToolCallDelta {
                            index,
                            id: tool_call
                                .get("id")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned),
                            // Names may continue after arguments begin; emit them at finish_reason.
                            name: None,
                            arguments_delta,
                            google_thought_signature,
                        });
                    }
                }
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            for (index, name) in std::mem::take(&mut state.pending_chat_tool_names) {
                out.push(LlmResponseChunk::ToolCallDelta {
                    index,
                    id: None,
                    name: Some(name),
                    arguments_delta: None,
                    google_thought_signature: None,
                });
            }
            let reason =
                if reason == "stop" && state.stop_reason.as_deref() == Some("content_filter") {
                    "content_filter"
                } else {
                    reason
                };
            out.push(LlmResponseChunk::MessageStop {
                reason: Some(chat_finish_reason(reason, state.chat_tool_call_decoded).to_string()),
            });
        }
    }
    out
}

/// Some Chat providers (Google among them) end a turn that called tools with `stop`.
/// Report it as a tool-call stop so other formats ask the client to run the tools.
pub(crate) fn chat_finish_reason(reason: &str, called_tools: bool) -> &str {
    if called_tools && reason == "stop" {
        "tool_calls"
    } else {
        reason
    }
}

/// Index for a Chat tool call that carries an `id` but no `index`: the earlier call with
/// that id, else the slot after every call seen so far. Unindexed calls arrive whole, so a
/// second named call with a known id is a different call that reused the id.
fn unindexed_chat_tool_index(
    state: &mut StreamTranslationState,
    id: &str,
    function: Option<&Map<String, Value>>,
) -> Result<usize, &'static str> {
    state.chat_unindexed_tool_calls = true;
    let known = state
        .tool_states
        .iter()
        .find(|(_, tool)| tool.decoded_chat_tool_id.as_deref() == Some(id))
        .map(|(index, _)| *index);
    if let Some(index) = known {
        let named = function
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .is_some_and(|name| !name.is_empty());
        if named && state.pending_chat_tool_names.contains_key(&index) {
            return Err("Chat tool call id is repeated");
        }
        return Ok(index);
    }
    match state.tool_states.keys().next_back() {
        Some(last) => last
            .checked_add(1)
            .ok_or("Chat tool call index is out of range"),
        None => Ok(0),
    }
}

// Encodes neutral streaming events into OpenAI Chat chunks.
fn encode_openai_chat_stream(
    state: &mut StreamTranslationState,
    event: LlmResponseChunk,
) -> Vec<Value> {
    // An in-band error is terminal: once the error is emitted, drop every later chunk.
    if state.errored {
        return Vec::new();
    }
    // A late source ID must not replace a generated ID already sent to the client.
    if state.target_message_id.is_none()
        && state.message_id.is_none()
        && !matches!(&event, LlmResponseChunk::MessageStart { id: Some(_), .. })
    {
        state.target_message_id = Some(openai_stream_id(state));
    }
    match event {
        LlmResponseChunk::MessageStart { id, model } => {
            record_source_identity(state, id, model);
            // `finish_openai_chat_stream` uses this flag to decide whether a clean EOF needs a
            // synthesized terminal chunk. Set it on the encode path as well as the decode path.
            state.saw_message_start = true;
            let emit_role = state.source.as_ref().is_some_and(|source| {
                source.as_str() == WireFormat::AnthropicMessages.as_str()
                    || source.as_str() == WireFormat::OpenAiResponses.as_str()
            });
            if state.emitted_message_start || !emit_role {
                Vec::new()
            } else {
                state.emitted_message_start = true;
                vec![openai_stream_chunk(
                    state,
                    json!({"role": "assistant"}),
                    None,
                    None,
                )]
            }
        }
        LlmResponseChunk::TextDelta { text, .. } | LlmResponseChunk::RefusalDelta { text, .. } => {
            vec![openai_stream_chunk(
                state,
                json!({"content": text}),
                None,
                None,
            )]
        }
        LlmResponseChunk::ReasoningDelta { text, .. } => {
            vec![openai_stream_chunk(
                state,
                json!({"reasoning_content": text}),
                None,
                None,
            )]
        }
        LlmResponseChunk::ReasoningDetailsDelta { details, text, .. } => {
            // Chat cannot replay Anthropic signature fragments or an encrypted item's
            // id announcement without its payload.
            let details: Vec<Value> = details
                .into_iter()
                .filter(|detail| match detail.get("type").and_then(Value::as_str) {
                    Some("anthropic.signature_delta") => false,
                    Some("reasoning.encrypted") => detail.get("data").is_some(),
                    _ => true,
                })
                .collect();
            if details.is_empty() && text.is_empty() {
                return Vec::new();
            }
            let details_text = reasoning_text_from_details(&details);
            let mut delta = json!({"reasoning_details": details});
            if !text.is_empty() && details_text.as_deref() != Some(text.as_str()) {
                delta["reasoning"] = Value::String(text);
            }
            vec![openai_stream_chunk(state, delta, None, None)]
        }
        LlmResponseChunk::ToolCallDelta {
            index,
            id,
            name,
            arguments_delta,
            google_thought_signature,
        } => {
            let tool = state.tool_states.entry(index).or_default();
            let google_thought_signature = match google_thought_signature {
                Some(signature) => match tool.google_thought_signature.as_ref() {
                    Some(previous) if previous != &signature => {
                        return encode_openai_chat_stream(
                            state,
                            LlmResponseChunk::DecodeError {
                                message: "conflicting Google tool signature".to_string(),
                            },
                        );
                    }
                    Some(_) => None,
                    None => {
                        tool.google_thought_signature = Some(signature.clone());
                        Some(signature)
                    }
                },
                None => None,
            };
            let id = id.filter(|id| {
                tool.google_thought_signature.is_none() || tool.id.as_ref() != Some(id)
            });
            if id.is_some() {
                tool.id = id.clone();
            }
            // Repeated full names would be concatenated by Chat clients as new fragments.
            let name = name.filter(|name| tool.name.as_ref() != Some(name));
            if name.is_some() {
                tool.name = name.clone();
            }
            if id.is_none()
                && name.is_none()
                && arguments_delta.is_none()
                && google_thought_signature.is_none()
            {
                return Vec::new();
            }
            // The source index counts every content block (Anthropic) or output item
            // (Responses), so text ahead of the first tool call shifts it. Chat clients use
            // the index as a subscript into `tool_calls`, so number calls in that array
            // instead, in order of first appearance.
            let chat_index = match tool.chat_tool_index {
                Some(chat_index) => chat_index,
                None => {
                    let chat_index = state.next_chat_tool_index;
                    state.next_chat_tool_index += 1;
                    state.tool_states.entry(index).or_default().chat_tool_index = Some(chat_index);
                    chat_index
                }
            };
            // The signature identifies Google Chat; keep other Chat streams unchanged.
            let emit_role = google_thought_signature.is_some() && !state.emitted_message_start;
            let tool_event = openai_tool_call_chunk(
                state,
                chat_index,
                id,
                name,
                arguments_delta,
                google_thought_signature,
            );
            if emit_role {
                state.emitted_message_start = true;
                vec![
                    openai_stream_chunk(state, json!({"role": "assistant"}), None, None),
                    tool_event,
                ]
            } else {
                vec![tool_event]
            }
        }
        LlmResponseChunk::Usage(usage) => {
            state.usage = usage;
            state.saw_backend_usage = true;
            if state.finished {
                vec![openai_usage_chunk(state)]
            } else {
                Vec::new()
            }
        }
        LlmResponseChunk::ToolCallDone { .. }
        | LlmResponseChunk::CustomToolCallDelta { .. }
        | LlmResponseChunk::CustomToolCallDone { .. }
        | LlmResponseChunk::ComputerToolCallDone { .. }
        | LlmResponseChunk::HostedToolCallDone { .. }
        | LlmResponseChunk::ReasoningStarted { .. }
        | LlmResponseChunk::ReasoningDone { .. }
        | LlmResponseChunk::RefusalDone { .. }
        | LlmResponseChunk::TextDone { .. }
        | LlmResponseChunk::OpaqueState(_)
        | LlmResponseChunk::ResponseMetadata(_)
        | LlmResponseChunk::GeneratedImageDone(_)
        | LlmResponseChunk::CompactionDone(_)
        | LlmResponseChunk::PauseTurn(_)
        | LlmResponseChunk::ResponseTerminal(_) => Vec::new(),
        LlmResponseChunk::MessageStop { reason } => {
            if state.finished {
                return Vec::new();
            }
            state.finished = true;
            vec![openai_stream_chunk(
                state,
                json!({}),
                Some(openai_finish_reason(reason.as_deref())),
                state
                    .saw_backend_usage
                    .then(|| encode_openai_usage(&state.usage)),
            )]
        }
        LlmResponseChunk::DecodeError { message } | LlmResponseChunk::StreamError { message } => {
            // An in-band error is terminal: emit the error, then nothing further.
            state.finished = true; // finish() adds no success events
            state.errored = true; // the entry guard drops any later chunk
            vec![json!({"error": {"message": message}})]
        }
    }
}

// Emits a terminal chunk if the source stream ended before a stop event arrived.
fn finish_openai_chat_stream(state: &mut StreamTranslationState) -> Vec<Value> {
    if state.finished || !state.saw_message_start {
        return Vec::new();
    }
    state.finished = true;
    vec![openai_stream_chunk(
        state,
        json!({}),
        Some(openai_finish_reason(state.stop_reason.as_deref())),
        state
            .saw_backend_usage
            .then(|| encode_openai_usage(&state.usage)),
    )]
}

// Builds a single OpenAI Chat stream chunk payload.
fn openai_stream_chunk(
    state: &StreamTranslationState,
    delta: Value,
    finish_reason: Option<String>,
    usage: Option<Value>,
) -> Value {
    let mut payload = json!({
        "id": openai_stream_id(state),
        "object": "chat.completion.chunk",
        "created": state.generated_metadata.created_at,
        "model": target_model_or_source_model(state),
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    });
    if let Some(usage) = usage {
        payload["usage"] = usage;
    }
    payload
}

// Builds the usage-only chunk OpenAI emits after the terminal choices chunk.
fn openai_usage_chunk(state: &StreamTranslationState) -> Value {
    let mut payload = openai_stream_chunk(
        state,
        json!({}),
        None,
        Some(encode_openai_usage(&state.usage)),
    );
    payload["choices"] = json!([]);
    payload
}

// Builds an OpenAI Chat tool-call delta chunk.
fn openai_tool_call_chunk(
    state: &StreamTranslationState,
    index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments: Option<String>,
    google_thought_signature: Option<String>,
) -> Value {
    let mut function = Map::new();
    if let Some(name) = name {
        function.insert("name".to_string(), Value::String(name));
    }
    if let Some(arguments) = arguments {
        function.insert("arguments".to_string(), Value::String(arguments));
    }
    let mut tool_call = Map::new();
    tool_call.insert("index".to_string(), json!(index));
    tool_call.insert("type".to_string(), json!("function"));
    tool_call.insert("function".to_string(), Value::Object(function));
    if let Some(id) = id {
        tool_call.insert("id".to_string(), Value::String(id));
    }
    let mut tool_call = Value::Object(tool_call);
    encode_google_thought_signature(&mut tool_call, google_thought_signature.as_deref());
    openai_stream_chunk(state, json!({"tool_calls": [tool_call]}), None, None)
}

// Converts any upstream message ID into an OpenAI-looking stream ID.
fn openai_stream_id(state: &StreamTranslationState) -> String {
    let Some(id) = target_message_id_or_source_message_id(state) else {
        return format!("chatcmpl_{}", state.generated_metadata.id);
    };
    if id.starts_with("chatcmpl") {
        id.to_string()
    } else if let Some(rest) = id.strip_prefix("msg_").or_else(|| id.strip_prefix("resp_")) {
        format!("chatcmpl_{rest}")
    } else {
        format!("chatcmpl_{id}")
    }
}

// Maps provider stop reasons into OpenAI's finish-reason vocabulary.
fn openai_finish_reason(reason: Option<&str>) -> String {
    match reason {
        Some("end_turn") | Some("stop_sequence") | None => "stop".to_string(),
        Some("max_tokens") => "length".to_string(),
        Some("tool_use") => "tool_calls".to_string(),
        Some("refusal") => "content_filter".to_string(),
        Some(other) => other.to_string(),
    }
}
