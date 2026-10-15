// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Google-compatible Chat signatures must survive semantic translation, not just raw replay.

pub mod common;

use futures::{StreamExt, executor::block_on};
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, LlmRequest, LlmResponseChunk, LlmResponseStreamEvent, Message,
    ResponseAccumulator,
};
use switchyard_translation::{
    StreamTranslationState, TranslationEngine, TranslationPolicy, WireFormat,
};

use common::normalized_policy;

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

const SIGNATURE: &str = "  AQID+/==\n雪🧠\t ";
const NEXT_SIGNATURE: &str = "bmV4dA==\nλ ";

fn call(id: &str, signature: Option<&str>) -> Value {
    let mut call = json!({
        "id": id, "type": "function",
        "function": {"name": "lookup", "arguments": "{\"city\":\"Paris\"}"}
    });
    if let Some(signature) = signature {
        call["extra_content"] = json!({"google": {"thought_signature": signature}});
    }
    call
}

fn response_body() -> Value {
    json!({
        "id": "chatcmpl-google", "object": "chat.completion", "model": "gemini-provider",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": null,
            "tool_calls": [call("call_signed", Some(SIGNATURE)), call("call_unsigned", None)]
        }, "finish_reason": "tool_calls"}]
    })
}

fn assert_calls(calls: &Value, expected: &[(&str, Option<&str>)]) {
    let calls = calls.as_array().expect("Chat tool_calls array");
    assert_eq!(calls.len(), expected.len());
    for (call, (id, signature)) in calls.iter().zip(expected) {
        assert_eq!(call["id"], *id);
        assert_eq!(call["function"]["name"], "lookup");
        assert_eq!(
            serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(),
            json!({"city": "Paris"})
        );
        match signature {
            Some(signature) => assert_eq!(
                call["extra_content"],
                json!({"google": {"thought_signature": signature}})
            ),
            None => assert!(call.get("extra_content").is_none(), "unsigned call: {call}"),
        }
    }
}

fn assert_normalized_calls(response: &AggLlmResponse) {
    let calls: Vec<_> = response.outputs[0]
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id, "call_signed");
    assert_eq!(calls[0].name, "lookup");
    assert_eq!(calls[0].arguments, json!({"city": "Paris"}));
    assert_eq!(
        calls[0].google_thought_signature.as_deref(),
        Some(SIGNATURE)
    );
    assert_eq!(calls[1].id, "call_unsigned");
    assert_eq!(calls[1].name, "lookup");
    assert_eq!(calls[1].arguments, json!({"city": "Paris"}));
    assert_eq!(calls[1].google_thought_signature, None);
}

#[test]
fn buffered_response_retains_signature_after_raw_preservation_is_cleared() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    let mut response = engine
        .decode_response(WireFormat::OpenAiChat, &response_body(), &policy)?
        .response;
    assert_normalized_calls(&response);
    assert!(!response.preservation.responses.is_empty());
    response.preservation.responses.clear();
    response.model = Some("routed-gemini".into());
    let output = engine
        .encode_response(WireFormat::OpenAiChat, &response, &policy)?
        .body;
    assert_eq!(output["model"], "routed-gemini");
    assert_calls(
        &output["choices"][0]["message"]["tool_calls"],
        &[("call_signed", Some(SIGNATURE)), ("call_unsigned", None)],
    );
    Ok(())
}

#[test]
fn reconstructed_assistant_continuation_retains_signature_after_routing() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    let response = engine
        .decode_response(WireFormat::OpenAiChat, &response_body(), &policy)?
        .response;
    let mut request = engine.decode_request(WireFormat::OpenAiChat, &json!({
        "model": "client-gemini", "messages": [{"role": "user", "content": "Look up Paris twice"}]
    }), &policy)?.request;
    let output = response.first_output().expect("assistant output");
    request.messages.push(Message {
        role: output.role,
        content: output.content.clone(),
    });
    request.model = Some("routed-gemini".into());
    request.preservation.requests.clear();
    let encoded = engine
        .encode_request(WireFormat::OpenAiChat, &request, &policy)?
        .body;
    assert_eq!(encoded["model"], "routed-gemini");
    assert_calls(
        &encoded["messages"][1]["tool_calls"],
        &[("call_signed", Some(SIGNATURE)), ("call_unsigned", None)],
    );
    Ok(())
}

#[test]
fn client_returned_history_keeps_parallel_and_sequential_signatures_by_call_id() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({"model": "gemini", "messages": [
        {"role": "user", "content": "Look up Paris"},
        response_body()["choices"][0]["message"].clone(),
        // Results can arrive in a different order from the parallel calls.
        {"role": "tool", "tool_call_id": "call_unsigned", "content": "second result"},
        {"role": "tool", "tool_call_id": "call_signed", "content": "first result"},
        {"role": "assistant", "content": null, "tool_calls": [call("call_next", Some(NEXT_SIGNATURE))]},
        {"role": "tool", "tool_call_id": "call_next", "content": "next result"},
        {"role": "assistant", "content": null, "tool_calls": [call("call_last_unsigned", None)]}
    ]});
    for policy in [TranslationPolicy::default(), normalized_policy()] {
        let mut request = engine
            .decode_request(WireFormat::OpenAiChat, &body, &policy)?
            .request;
        request.preservation.requests.clear();
        let encoded = engine
            .encode_request(WireFormat::OpenAiChat, &request, &policy)?
            .body;
        assert_calls(
            &encoded["messages"][1]["tool_calls"],
            &[("call_signed", Some(SIGNATURE)), ("call_unsigned", None)],
        );
        assert_eq!(encoded["messages"][2]["tool_call_id"], "call_unsigned");
        assert_eq!(encoded["messages"][3]["tool_call_id"], "call_signed");
        assert_calls(
            &encoded["messages"][4]["tool_calls"],
            &[("call_next", Some(NEXT_SIGNATURE))],
        );
        assert_calls(
            &encoded["messages"][6]["tool_calls"],
            &[("call_last_unsigned", None)],
        );
    }
    Ok(())
}

fn stream_frames() -> Vec<Value> {
    vec![
        json!({"id": "chatcmpl-google", "model": "gemini-provider", "choices": [{"index": 0, "delta": {
        "role": "assistant", "tool_calls": [
            {"index": 0, "id": "call_signed", "type": "function", "function": {"name": "look", "arguments": "{\"city\":"}},
            {"index": 1, "id": "call_unsigned", "type": "function", "function": {"name": "loo", "arguments": "{"}}
        ]}}]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 1, "function": {"name": "kup", "arguments": "\"city\":\"Paris\"}"}},
            {"index": 0, "function": {"name": "up", "arguments": "\"Paris\"}"}}
        ]}}]}),
        // Signature-only late delta must bind to index 0, not the most recent call.
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "extra_content": {"google": {"thought_signature": SIGNATURE}}}
        ]}}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ]
}

fn normalize_and_encode(
    engine: &TranslationEngine,
    frames: &[Value],
) -> Result<(AggLlmResponse, Vec<Value>), Box<dyn std::error::Error + Send + Sync>> {
    let mut decoder = StreamTranslationState::new(WireFormat::OpenAiChat, WireFormat::OpenAiChat);
    let mut encoder = StreamTranslationState::new(WireFormat::OpenAiChat, WireFormat::OpenAiChat);
    let mut accumulator = ResponseAccumulator::new();
    let mut encoded = Vec::new();
    for frame in frames {
        let decoded =
            engine.decode_stream_event(&mut decoder, WireFormat::OpenAiChat, frame.clone())?;
        let (_, chunks) = decoded.into_parts();
        for chunk in &chunks {
            accumulator.push(chunk.clone());
        }
        // Deliberately discard raw replay metadata even for same-format SSE.
        encoded.extend(engine.encode_stream_event(
            &mut encoder,
            WireFormat::OpenAiChat,
            LlmResponseStreamEvent::new(chunks),
        )?);
    }
    encoded.extend(engine.finish_stream(&mut encoder, WireFormat::OpenAiChat)?);
    Ok((accumulator.finish(), encoded))
}

fn assert_stream_signature_mapping(frames: &[Value]) {
    assert!(
        frames
            .iter()
            .any(|frame| frame["choices"][0]["delta"]["role"] == "assistant")
    );
    let mut ids = std::collections::BTreeMap::new();
    let mut signatures = std::collections::BTreeMap::<u64, String>::new();
    let mut signature_count = 0;
    for frame in frames {
        if let Some(calls) = frame["choices"][0]["delta"]["tool_calls"].as_array() {
            for call in calls {
                let index = call["index"].as_u64().expect("tool delta index");
                if let Some(id) = call["id"].as_str() {
                    ids.insert(index, id.to_owned());
                }
                if let Some(signature) =
                    call["extra_content"]["google"]["thought_signature"].as_str()
                {
                    signature_count += 1;
                    signatures.insert(index, signature.to_owned());
                }
                if index == 1 {
                    assert!(
                        call.get("extra_content").is_none(),
                        "unsigned delta: {call}"
                    );
                }
            }
        }
    }
    assert_eq!(ids.get(&0).map(String::as_str), Some("call_signed"));
    assert_eq!(ids.get(&1).map(String::as_str), Some("call_unsigned"));
    assert_eq!(signatures.get(&0).map(String::as_str), Some(SIGNATURE));
    assert_eq!(
        signature_count, 1,
        "complete snapshots must not be emitted twice"
    );
    assert!(!signatures.contains_key(&1));
}

#[test]
fn normalized_sse_late_signature_survives_accumulation_and_synthetic_restream() -> TestResult {
    let engine = TranslationEngine::default();
    let (response, encoded) = normalize_and_encode(&engine, &stream_frames())?;
    assert_normalized_calls(&response);
    assert_stream_signature_mapping(&encoded);
    let (round_trip, _) = normalize_and_encode(&engine, &encoded)?;
    assert_normalized_calls(&round_trip);
    let buffered = engine
        .encode_response(WireFormat::OpenAiChat, &response, &normalized_policy())?
        .body;
    assert_calls(
        &buffered["choices"][0]["message"]["tool_calls"],
        &[("call_signed", Some(SIGNATURE)), ("call_unsigned", None)],
    );
    let events = block_on(response.into_stream().collect::<Vec<_>>());
    let mut encoder = StreamTranslationState::new(WireFormat::OpenAiChat, WireFormat::OpenAiChat);
    let mut synthetic = Vec::new();
    for event in events {
        synthetic.extend(engine.encode_stream_event(
            &mut encoder,
            WireFormat::OpenAiChat,
            event?,
        )?);
    }
    synthetic.extend(engine.finish_stream(&mut encoder, WireFormat::OpenAiChat)?);
    assert_stream_signature_mapping(&synthetic);
    let (restreamed, _) = normalize_and_encode(&engine, &synthetic)?;
    assert_normalized_calls(&restreamed);
    Ok(())
}

fn assert_no_google_signature(value: &Value) {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                assert!(
                    !matches!(
                        key.as_str(),
                        "google_thought_signature" | "thought_signature" | "extra_content"
                    ),
                    "Google envelope leaked: {value}"
                );
                assert_no_google_signature(child);
            }
        }
        Value::Array(values) => values.iter().for_each(assert_no_google_signature),
        _ => {}
    }
}

#[test]
fn non_chat_encodes_omit_google_signature_without_changing_foreign_reasoning() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = normalized_policy();
    let body = json!({"id": "msg_foreign", "model": "claude", "type": "message", "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Inspect the tool result.", "signature": "foreign-anthropic-signature"},
            {"type": "tool_use", "id": "call_signed", "name": "lookup", "input": {"city": "Paris"}}
        ], "stop_reason": "tool_use", "usage": {"input_tokens": 1, "output_tokens": 2}
    });
    let baseline = engine
        .decode_response(WireFormat::AnthropicMessages, &body, &policy)?
        .response;
    let mut signed = baseline.clone();
    for block in &mut signed.outputs[0].content {
        if let ContentBlock::ToolCall(call) = block {
            call.google_thought_signature = Some(SIGNATURE.into());
        }
    }
    for target in [
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiResponses,
        WireFormat::BedrockConverse,
    ] {
        let expected = engine.encode_response(target, &baseline, &policy)?.body;
        let actual = engine.encode_response(target, &signed, &policy)?.body;
        assert_no_google_signature(&actual);
        assert_eq!(
            actual, expected,
            "{target:?} reasoning must remain unchanged"
        );
        if target == WireFormat::AnthropicMessages {
            assert_eq!(
                actual["content"][0]["signature"],
                "foreign-anthropic-signature"
            );
        }
        let request = |response: &AggLlmResponse| LlmRequest {
            model: Some("target-model".into()),
            messages: vec![Message {
                role: response.outputs[0].role,
                content: response.outputs[0].content.clone(),
            }],
            ..LlmRequest::default()
        };
        let expected = engine
            .encode_request(target, &request(&baseline), &policy)?
            .body;
        let actual = engine
            .encode_request(target, &request(&signed), &policy)?
            .body;
        assert_no_google_signature(&actual);
        assert_eq!(
            actual, expected,
            "{target:?} history reasoning must remain unchanged"
        );
    }
    Ok(())
}

#[test]
fn repeated_complete_sse_signatures_are_idempotent_on_decode_and_encode() -> TestResult {
    let engine = TranslationEngine::default();
    let mut frames = stream_frames();
    frames.insert(3, frames[2].clone());
    let (response, encoded) = normalize_and_encode(&engine, &frames)?;
    assert_normalized_calls(&response);
    assert_stream_signature_mapping(&encoded);

    // Also drive duplicate neutral snapshots directly so encoder deduplication is exercised.
    let mut decoder = StreamTranslationState::new(WireFormat::OpenAiChat, WireFormat::OpenAiChat);
    let mut encoder = StreamTranslationState::new(WireFormat::OpenAiChat, WireFormat::OpenAiChat);
    let mut output = Vec::new();
    for frame in stream_frames() {
        let decoded = engine.decode_stream_event(&mut decoder, WireFormat::OpenAiChat, frame)?;
        let (_, chunks) = decoded.into_parts();
        let has_signature = chunks.iter().any(|chunk| {
            matches!(
                chunk,
                LlmResponseChunk::ToolCallDelta {
                    google_thought_signature: Some(_),
                    ..
                }
            )
        });
        output.extend(engine.encode_stream_event(
            &mut encoder,
            WireFormat::OpenAiChat,
            LlmResponseStreamEvent::new(chunks.clone()),
        )?);
        if has_signature {
            output.extend(engine.encode_stream_event(
                &mut encoder,
                WireFormat::OpenAiChat,
                LlmResponseStreamEvent::new(chunks),
            )?);
        }
    }
    output.extend(engine.finish_stream(&mut encoder, WireFormat::OpenAiChat)?);
    assert_stream_signature_mapping(&output);
    Ok(())
}

#[test]
fn conflicting_or_non_string_sse_signatures_fail_without_disclosing_secrets() -> TestResult {
    let engine = TranslationEngine::default();
    for invalid in [
        json!(NEXT_SIGNATURE),
        json!(42),
        json!({"secret": NEXT_SIGNATURE}),
    ] {
        let mut decoder =
            StreamTranslationState::new(WireFormat::OpenAiChat, WireFormat::OpenAiChat);
        for frame in stream_frames().into_iter().take(3) {
            engine.decode_stream_event(&mut decoder, WireFormat::OpenAiChat, frame)?;
        }
        let decoded = engine.decode_stream_event(
            &mut decoder,
            WireFormat::OpenAiChat,
            json!({
                "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "extra_content": {"google": {"thought_signature": invalid}}}
                ]}}]
            }),
        )?;
        let messages: Vec<_> = decoded
            .normalized()
            .iter()
            .filter_map(|chunk| match chunk {
                LlmResponseChunk::DecodeError { message }
                | LlmResponseChunk::StreamError { message } => Some(message),
                _ => None,
            })
            .collect();
        assert!(
            !messages.is_empty(),
            "invalid signature must be an explicit stream error"
        );
        for message in messages {
            assert!(!message.contains(SIGNATURE));
            assert!(!message.contains(NEXT_SIGNATURE));
            assert!(!message.contains("AQID"));
            assert!(!message.contains("bmV4dA"));
        }
    }
    let mut encoder = StreamTranslationState::new(WireFormat::OpenAiChat, WireFormat::OpenAiChat);
    let snapshot = |signature: &str| {
        LlmResponseStreamEvent::new(vec![LlmResponseChunk::ToolCallDelta {
            index: 0,
            id: Some("call_signed".into()),
            name: Some("lookup".into()),
            arguments_delta: Some("{\"city\":\"Paris\"}".into()),
            google_thought_signature: Some(signature.into()),
        }])
    };
    engine.encode_stream_event(&mut encoder, WireFormat::OpenAiChat, snapshot(SIGNATURE))?;
    let errors = engine.encode_stream_event(
        &mut encoder,
        WireFormat::OpenAiChat,
        snapshot(NEXT_SIGNATURE),
    )?;
    assert!(
        errors
            .iter()
            .any(|event| event["error"]["message"].is_string())
    );
    let serialized = serde_json::to_string(&errors)?;
    assert!(!serialized.contains("AQID"));
    assert!(!serialized.contains("bmV4dA"));
    Ok(())
}

#[test]
fn google_tool_signature_does_not_replace_chat_foreign_reasoning_details() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = normalized_policy();
    let details = common::text_and_encrypted_reasoning_details();
    let mut body = response_body();
    body["choices"][0]["message"]["reasoning_details"] = details.clone();
    let response = engine
        .decode_response(WireFormat::OpenAiChat, &body, &policy)?
        .response;
    let encoded = engine
        .encode_response(WireFormat::OpenAiChat, &response, &policy)?
        .body;
    assert_eq!(
        encoded["choices"][0]["message"]["reasoning_details"],
        details
    );
    assert_calls(
        &encoded["choices"][0]["message"]["tool_calls"],
        &[("call_signed", Some(SIGNATURE)), ("call_unsigned", None)],
    );
    let request_body = json!({
        "model": common::REASONING_MODEL,
        "messages": [body["choices"][0]["message"].clone()]
    });
    let request = engine
        .decode_request(WireFormat::OpenAiChat, &request_body, &policy)?
        .request;
    let encoded = engine
        .encode_request(WireFormat::OpenAiChat, &request, &policy)?
        .body;
    assert_eq!(encoded["messages"][0]["reasoning_details"], details);
    assert_calls(
        &encoded["messages"][0]["tool_calls"],
        &[("call_signed", Some(SIGNATURE)), ("call_unsigned", None)],
    );
    Ok(())
}

#[test]
fn preserved_sse_does_not_replay_duplicate_or_conflicting_google_signatures() -> TestResult {
    let engine = TranslationEngine::default();
    let mut frames = stream_frames();
    frames.insert(3, frames[2].clone());
    frames[3]["choices"][0]["delta"]["tool_calls"][0]["extra_content"]["google"]["other"] =
        json!("keep");
    frames[3]["choices"][0]["delta"]["tool_calls"][0]["provider_marker"] = json!(true);
    let mut decoder = StreamTranslationState::default();
    let mut encoder = StreamTranslationState::default();
    let mut replay = Vec::new();
    for frame in frames {
        let event = engine.decode_stream_event(&mut decoder, WireFormat::OpenAiChat, frame)?;
        replay.extend(engine.encode_stream_event(&mut encoder, WireFormat::OpenAiChat, event)?);
    }
    assert_stream_signature_mapping(&replay);
    assert_eq!(
        replay[2]["choices"][0]["delta"]["tool_calls"][0]["function"],
        json!({})
    );
    assert_eq!(
        replay[3]["choices"][0]["delta"]["tool_calls"][0]["function"],
        json!({})
    );
    assert_eq!(
        replay[3]["choices"][0]["delta"]["tool_calls"][0]["extra_content"]["google"]["other"],
        "keep"
    );
    assert_eq!(
        replay[3]["choices"][0]["delta"]["tool_calls"][0]["provider_marker"],
        true
    );
    let (response, _) = normalize_and_encode(&engine, &replay)?;
    assert_normalized_calls(&response);
    let conflict = engine.decode_stream_event(
        &mut decoder,
        WireFormat::OpenAiChat,
        json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "extra_content": {"google": {"thought_signature": NEXT_SIGNATURE}}}
            ]}}]
        }),
    )?;
    let error = engine.encode_stream_event(&mut encoder, WireFormat::OpenAiChat, conflict)?;
    assert!(error[0]["error"]["message"].is_string());
    assert!(!serde_json::to_string(&error)?.contains("bmV4dA"));
    Ok(())
}

#[test]
fn signed_sse_requires_unambiguous_call_identity() -> TestResult {
    let engine = TranslationEngine::default();
    for (choice, call) in [
        (
            0,
            json!({"extra_content": {"google": {"thought_signature": SIGNATURE}}}),
        ),
        (
            1,
            json!({"index": 0, "extra_content": {"google": {"thought_signature": SIGNATURE}}}),
        ),
        (
            0,
            json!({"index": 0, "id": "different_call", "extra_content": {"google": {"thought_signature": SIGNATURE}}}),
        ),
    ] {
        let mut decoder = StreamTranslationState::default();
        engine.decode_stream_event(
            &mut decoder,
            WireFormat::OpenAiChat,
            stream_frames()[0].clone(),
        )?;
        let event = engine.decode_stream_event(
            &mut decoder,
            WireFormat::OpenAiChat,
            json!({
                "choices": [{"index": choice, "delta": {"tool_calls": [call]}}]
            }),
        )?;
        assert!(matches!(
            event.normalized(),
            [LlmResponseChunk::DecodeError { .. }]
        ));
    }
    Ok(())
}

#[test]
fn preserved_late_google_signature_and_later_updates_keep_client_call_id() -> TestResult {
    let engine = TranslationEngine::default();
    let mut frames = stream_frames();
    frames[2]["choices"][0]["delta"]["tool_calls"][0]["id"] = json!("call_signed");
    frames.insert(
        3,
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_signed", "function": {"arguments": " "},
                "provider_marker": "keep"}
        ]}}]}),
    );
    frames.insert(
        4,
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_signed", "extra_content": {"other_provider": "keep"}}
        ]}}]}),
    );
    let mut decoder = StreamTranslationState::default();
    let mut encoder = StreamTranslationState::default();
    let mut replay = Vec::new();
    for frame in frames {
        let event = engine.decode_stream_event(&mut decoder, WireFormat::OpenAiChat, frame)?;
        replay.extend(engine.encode_stream_event(&mut encoder, WireFormat::OpenAiChat, event)?);
    }
    // Chat clients append string deltas, including IDs, rather than replacing them.
    let mut client_ids = std::collections::BTreeMap::<u64, String>::new();
    for frame in &replay {
        if let Some(calls) = frame["choices"][0]["delta"]["tool_calls"].as_array() {
            for call in calls {
                if let Some(id) = call["id"].as_str() {
                    client_ids
                        .entry(call["index"].as_u64().unwrap())
                        .or_default()
                        .push_str(id);
                }
            }
        }
    }
    assert_eq!(client_ids[&0], "call_signed");
    assert_eq!(client_ids[&1], "call_unsigned");
    assert_eq!(
        replay[3]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
        " "
    );
    assert_eq!(
        replay[3]["choices"][0]["delta"]["tool_calls"][0]["provider_marker"],
        "keep"
    );
    assert_eq!(
        replay[4]["choices"][0]["delta"]["tool_calls"][0]["extra_content"]["other_provider"],
        "keep"
    );
    assert_stream_signature_mapping(&replay);
    let (response, _) = normalize_and_encode(&engine, &replay)?;
    assert_normalized_calls(&response);
    Ok(())
}

#[test]
fn another_chat_choice_before_during_or_after_google_signature_fails() -> TestResult {
    let engine = TranslationEngine::default();
    let other = json!({"choices": [{"index": 1, "delta": {"tool_calls": [
        {"index": 0, "id": "call_other_choice", "function": {"name": "lookup", "arguments": "{}"}}
    ]}}]});
    let signed = stream_frames()[2].clone();
    let mut cases = vec![
        ("before", vec![other.clone(), signed.clone()]),
        ("after", vec![signed.clone(), other.clone()]),
        (
            "same frame signed first",
            vec![json!({"choices": [
                signed["choices"][0].clone(), other["choices"][0].clone()
            ]})],
        ),
        (
            "same frame unsigned first",
            vec![json!({"choices": [
                other["choices"][0].clone(), signed["choices"][0].clone()
            ]})],
        ),
    ];
    for omit_index in [false, true] {
        let mut first = stream_frames()[0]["choices"][0].clone();
        let mut second = other["choices"][0].clone();
        let mut signed_choice = signed["choices"][0].clone();
        for choice in [&mut first, &mut second, &mut signed_choice] {
            if omit_index {
                choice.as_object_mut().unwrap().remove("index");
            } else {
                choice["index"] = json!(0);
            }
        }
        let ambiguous = json!({"choices": [first, second.clone()]});
        cases.push((
            "cardinality before",
            vec![ambiguous.clone(), signed.clone()],
        ));
        cases.push(("cardinality after", vec![signed.clone(), ambiguous]));
        cases.push((
            "cardinality same frame signed first",
            vec![json!({"choices": [
                signed_choice.clone(), second.clone()
            ]})],
        ));
        cases.push((
            "cardinality same frame unsigned first",
            vec![json!({"choices": [
                second, signed_choice
            ]})],
        ));
    }
    for (case, frames) in cases {
        for preserve_raw in [false, true] {
            let mut decoder = StreamTranslationState::default();
            let mut encoder = StreamTranslationState::default();
            engine.decode_stream_event(
                &mut decoder,
                WireFormat::OpenAiChat,
                stream_frames()[0].clone(),
            )?;
            let mut rejected = false;
            for frame in &frames {
                let event = engine.decode_stream_event(
                    &mut decoder,
                    WireFormat::OpenAiChat,
                    frame.clone(),
                )?;
                if matches!(event.normalized(), [LlmResponseChunk::DecodeError { .. }]) {
                    rejected = true;
                    let event = if preserve_raw {
                        event
                    } else {
                        LlmResponseStreamEvent::new(event.into_parts().1)
                    };
                    let output =
                        engine.encode_stream_event(&mut encoder, WireFormat::OpenAiChat, event)?;
                    assert!(
                        output
                            .iter()
                            .any(|frame| frame["error"]["message"].is_string())
                    );
                    let wire = serde_json::to_string(&output)?;
                    assert!(!wire.contains("AQID"), "{case}: signature in error wire");
                    assert!(
                        !wire.contains("call_other_choice"),
                        "{case}: raw ambiguous call replayed"
                    );
                    break;
                }
            }
            assert!(
                rejected,
                "{case}, preserve_raw={preserve_raw}: ambiguous Google identity accepted"
            );
        }
    }
    Ok(())
}

#[test]
fn same_event_google_snapshots_keep_client_identity_and_argument_fragments() -> TestResult {
    let engine = TranslationEngine::default();
    let frames = vec![
        json!({"id": "chatcmpl-google", "model": "gemini-provider", "choices": [
            {"index": 0, "delta": {"role": "assistant"}}
        ]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_signed", "type": "function",
                "function": {"name": "look", "arguments": "{\"city\":"},
                "extra_content": {"google": {"thought_signature": SIGNATURE}}},
            {"index": 1, "id": "call_unsigned", "type": "function",
                "function": {"name": "lookup", "arguments": "{\"city\":\"Paris\"}"}},
            {"index": 0, "id": "call_signed",
                "function": {"name": "up", "arguments": "\"Paris\"}"},
                "extra_content": {"google": {"thought_signature": SIGNATURE, "other": "keep"}},
                "provider_marker": true}
        ]}}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let mut decoder = StreamTranslationState::default();
    let mut encoder = StreamTranslationState::default();
    let mut replay = Vec::new();
    for frame in frames {
        let event = engine.decode_stream_event(&mut decoder, WireFormat::OpenAiChat, frame)?;
        replay.extend(engine.encode_stream_event(&mut encoder, WireFormat::OpenAiChat, event)?);
    }
    let calls = replay[1]["choices"][0]["delta"]["tool_calls"]
        .as_array()
        .unwrap();
    let mut client_id = String::new();
    let mut client_signature = String::new();
    let mut client_arguments = String::new();
    for call in calls.iter().filter(|call| call["index"] == 0) {
        client_id.push_str(call["id"].as_str().unwrap_or_default());
        client_signature.push_str(
            call["extra_content"]["google"]["thought_signature"]
                .as_str()
                .unwrap_or_default(),
        );
        client_arguments.push_str(call["function"]["arguments"].as_str().unwrap_or_default());
    }
    assert_eq!(client_id, "call_signed");
    assert_eq!(client_signature, SIGNATURE);
    assert_eq!(
        serde_json::from_str::<Value>(&client_arguments)?,
        json!({"city": "Paris"})
    );
    assert_eq!(calls[2]["extra_content"]["google"]["other"], "keep");
    assert_eq!(calls[2]["provider_marker"], true);
    let (response, _) = normalize_and_encode(&engine, &replay)?;
    assert_normalized_calls(&response);
    Ok(())
}

#[test]
fn late_google_signature_rejects_prior_differing_call_identity_without_leaking_values() -> TestResult
{
    let engine = TranslationEngine::default();
    for preserve_raw in [false, true] {
        let mut decoder = StreamTranslationState::default();
        let mut encoder = StreamTranslationState::default();
        for (id, arguments) in [
            ("prior-id-secret-A", "{\"city\":"),
            ("prior-id-secret-B", "\"Paris\"}"),
        ] {
            let event = engine.decode_stream_event(
                &mut decoder,
                WireFormat::OpenAiChat,
                json!({
                    "choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": 0, "id": id, "function": {"arguments": arguments}
                    }]}}]
                }),
            )?;
            // The unsigned stream keeps its existing identity-update behavior.
            assert!(event.normalized().iter().any(|chunk| matches!(
                chunk, LlmResponseChunk::ToolCallDelta { id: Some(actual), .. } if actual == id
            )));
            let event = if preserve_raw {
                event
            } else {
                LlmResponseStreamEvent::new(event.into_parts().1)
            };
            engine.encode_stream_event(&mut encoder, WireFormat::OpenAiChat, event)?;
        }
        let event = engine.decode_stream_event(
            &mut decoder,
            WireFormat::OpenAiChat,
            json!({
                "choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": 0, "extra_content": {"google": {"thought_signature": SIGNATURE}}
                }]}}]
            }),
        )?;
        assert!(matches!(
            event.normalized(),
            [LlmResponseChunk::DecodeError { .. }]
        ));
        let event = if preserve_raw {
            event
        } else {
            LlmResponseStreamEvent::new(event.into_parts().1)
        };
        let output = engine.encode_stream_event(&mut encoder, WireFormat::OpenAiChat, event)?;
        assert!(
            output
                .iter()
                .any(|frame| frame["error"]["message"].is_string())
        );
        let wire = serde_json::to_string(&output)?;
        assert!(!wire.contains("prior-id-secret"));
        assert!(!wire.contains("AQID"));
    }
    Ok(())
}

#[test]
fn google_signature_can_precede_first_call_id_and_stable_id_updates() -> TestResult {
    let engine = TranslationEngine::default();
    let mut frames = stream_frames();
    let call = &mut frames[0]["choices"][0]["delta"]["tool_calls"][0];
    call.as_object_mut().unwrap().remove("id");
    call["extra_content"] = json!({"google": {"thought_signature": SIGNATURE}});
    frames[2]["choices"][0]["delta"]["tool_calls"][0]["id"] = json!("call_signed");
    frames.insert(3, frames[2].clone());
    for preserve_raw in [false, true] {
        let mut decoder = StreamTranslationState::default();
        let mut encoder = StreamTranslationState::default();
        let mut output = Vec::new();
        for frame in &frames {
            let event =
                engine.decode_stream_event(&mut decoder, WireFormat::OpenAiChat, frame.clone())?;
            let event = if preserve_raw {
                event
            } else {
                LlmResponseStreamEvent::new(event.into_parts().1)
            };
            output.extend(engine.encode_stream_event(
                &mut encoder,
                WireFormat::OpenAiChat,
                event,
            )?);
        }
        let (response, _) = normalize_and_encode(&engine, &output)?;
        assert_normalized_calls(&response);
        assert_stream_signature_mapping(&output);
    }
    Ok(())
}
