// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native request fidelity after routing and policy edits.

use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use switchyard_translation::{
    ContentBlock, InstructionBlock, LossyConversionPolicy, NativeAnthropicRequest,
    PreservationPolicy, Role, TargetCapabilities, TranslationEngine, TranslationError,
    TranslationPolicy, WireFormat,
};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn native_body() -> Value {
    // Opaque feature values are synthetic data, not a provider schema or verdict.
    json!({
        "model": "source-model",
        "max_tokens": 512,
        "stream": true,
        "system": [
            {"type": "text", "text": "synthetic attribution marker", "future": {"order": 0}},
            {"type": "text", "text": "policy text", "cache_control": {
                "type": "ephemeral", "ttl": "1h", "future": {"scope": [1, 2]}
            }}
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello",
                "cache_control": {"type": "ephemeral", "future": true}}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_example",
                "name": "lookup", "input": {"query": "sample"}, "future": {"value": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_example",
                "content": [{"type": "text", "text": "sample", "future": [1, 2]}]}]},
            {"role": "system", "clear_at": "turn", "future": {"attribute": true},
                "content": [{"type": "text", "text": "turn policy"}]},
            {"role": "assistant", "content": [{"type": "future_content", "opaque": {"nested": [1, 2]}}]}
        ],
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}, "strict": true,
            "defer_loading": true, "cache_control": {"type": "ephemeral"},
            "future": {"nested": ["opaque"]}}],
        "thinking": {"type": "adaptive", "future_binding": {"opaque": [1, 2]}},
        "output_config": {"effort": "low", "future_budget": {"opaque": 17},
            "format": {"type": "json_schema", "schema": {"type": "object"}, "future": true}},
        "context_management": {"edits": [{"type": "future_edit", "opaque": true}]},
        "safeguards": {"synthetic_opaque_fixture": [
            {"sentinel": "no-provider-schema"}, null, {"bytes": [0, 17, 255]}
        ]},
        "future_request": {"nested": {"opaque": [true, null, {"value": "sample"}]}},
        "metadata": {"example": "synthetic"}
    })
}

#[test]
fn native_model_and_effort_edits_preserve_every_other_json_value() -> TestResult {
    let body = native_body();
    let safeguards_bytes = serde_json::to_vec(&body["safeguards"])?;
    let policy = TranslationPolicy::default();
    let native = NativeAnthropicRequest::decode(&body, &policy)?;
    let mut edited = native.request().clone();
    edited.model = Some("target-model".to_string());
    edited.reasoning.effort = Some("high".to_string());
    edited.preservation.requests.clear();
    let encoded = native.encode(&edited, &policy)?;
    assert_eq!(
        serde_json::to_vec(&encoded["safeguards"])?,
        safeguards_bytes
    );
    let mut expected = body;
    expected["model"] = json!("target-model");
    expected["output_config"]["effort"] = json!("high");
    assert_eq!(encoded, expected);
    // A second continuation must retain the same historical prefix and tool IDs.
    let next = NativeAnthropicRequest::decode(&encoded, &policy)?;
    let mut continuation = next.request().clone();
    continuation.model = Some("next-model".to_string());
    expected["model"] = json!("next-model");
    let continued = next.encode(&continuation, &policy)?;
    assert_eq!(
        serde_json::to_vec(&continued["safeguards"])?,
        safeguards_bytes
    );
    assert_eq!(continued, expected);
    Ok(())
}

#[test]
fn fixed_disabled_and_unconfigured_thinking_survive_model_edits() -> TestResult {
    let policy = TranslationPolicy::default();
    for thinking in [
        json!({"type": "enabled", "budget_tokens": 128, "future": {"opaque": true}}),
        json!({"type": "adaptive", "future": [1, 2]}),
        json!({"type": "disabled"}),
    ] {
        let mut body = native_body();
        body["thinking"] = thinking;
        body["output_config"]
            .as_object_mut()
            .unwrap()
            .remove("effort");
        let native = NativeAnthropicRequest::decode(&body, &policy)?;
        let mut edited = native.request().clone();
        edited.model = Some("target-model".to_string());
        body["model"] = json!("target-model");
        assert_eq!(native.encode(&edited, &policy)?, body);
    }
    Ok(())
}

#[test]
fn scalar_and_extension_policy_edits_are_authoritative() -> TestResult {
    let body = native_body();
    let policy = TranslationPolicy::default();
    let native = NativeAnthropicRequest::decode(&body, &policy)?;
    let mut edited = native.request().clone();
    edited.model = Some("constrained-model".to_string());
    edited.stream = false;
    edited.output.max_output_tokens = Some(64);
    edited.sampling.temperature = Some(0.25);
    edited.reasoning.effort = None;
    edited.reasoning.raw = None;
    edited.extensions.fields.remove("context_management");
    edited.extensions.fields.remove("future_request");
    let mut expected = body;
    expected["model"] = json!("constrained-model");
    expected["stream"] = json!(false);
    expected["max_tokens"] = json!(64);
    expected["temperature"] = json!(0.25);
    expected["output_config"]
        .as_object_mut()
        .unwrap()
        .remove("effort");
    let object = expected.as_object_mut().unwrap();
    object.remove("thinking");
    object.remove("context_management");
    object.remove("future_request");
    assert_eq!(native.encode(&edited, &policy)?, expected);
    Ok(())
}

#[test]
fn stale_preserved_bodies_never_override_edited_ir() -> TestResult {
    let policy = TranslationPolicy::default();
    let native = NativeAnthropicRequest::decode(&native_body(), &policy)?;
    let mut edited = native.request().clone();
    edited.model = Some("selected-model".to_string());
    edited.extensions.fields.remove("context_management");
    edited
        .preservation
        .requests
        .insert(WireFormat::AnthropicMessages.into(), native_body());
    let encoded = native.encode(&edited, &policy)?;
    assert_eq!(encoded["model"], json!("selected-model"));
    assert!(encoded.get("context_management").is_none());
    Ok(())
}

#[test]
fn embedded_preservation_is_rejected_at_ingress_and_after_metadata_edits() -> TestResult {
    let policy = TranslationPolicy {
        preservation: PreservationPolicy::Embed,
        ..Default::default()
    };
    let mut body = native_body();
    body["metadata"]["_switchyard_translation"] = json!({"requests": {
        "openai-chat": {"model": "untrusted", "previous_response_id": "untrusted-state"}
    }});
    assert!(matches!(
        NativeAnthropicRequest::decode(&body, &policy),
        Err(TranslationError::InvalidValue { .. })
    ));
    let native = NativeAnthropicRequest::decode(&native_body(), &policy)?;
    let mut edited = native.request().clone();
    edited
        .extensions
        .fields
        .insert("metadata".to_string(), body["metadata"].clone());
    assert!(native.encode(&edited, &policy).is_err());
    assert!(native.request().preservation.requests.is_empty());
    Ok(())
}

#[test]
fn prompt_history_tool_and_schema_changes_fail_instead_of_replaying_stale_policy() -> TestResult {
    let policy = TranslationPolicy::default();
    let native = NativeAnthropicRequest::decode(&native_body(), &policy)?;
    let baseline = native.request();
    let mut changes = Vec::new();
    let mut prompt = baseline.clone();
    prompt.instructions.insert(
        0,
        InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: "required policy".to_string(),
            }],
        },
    );
    changes.push(prompt);
    let mut history = baseline.clone();
    history.messages.remove(0);
    changes.push(history);
    let mut tools = baseline.clone();
    tools.tools.clear();
    changes.push(tools);
    let mut choice = baseline.clone();
    choice.tool_choice = Some(switchyard_translation::ToolChoice::None);
    changes.push(choice);
    let mut format = baseline.clone();
    format.output.response_format = None;
    changes.push(format);
    let mut collision = baseline.clone();
    collision
        .extensions
        .fields
        .insert("model".to_string(), json!("stale-model"));
    changes.push(collision);
    let mut decision = baseline.clone();
    decision.decision = Some(switchyard_translation::decision::DecisionRequest::default());
    changes.push(decision);
    let mut trailing = baseline.clone();
    trailing
        .trailing_items
        .push(switchyard_translation::TurnInputItem::RemoteCompaction);
    changes.push(trailing);
    for mut edited in changes {
        edited.preservation.requests.clear();
        assert!(matches!(
            native.encode(&edited, &policy),
            Err(TranslationError::LossyConversion(_))
        ));
    }
    Ok(())
}

#[test]
fn debug_is_redacted_when_nested_in_host_contexts() -> TestResult {
    #[derive(Debug)]
    struct Context {
        native: NativeAnthropicRequest,
    }

    let context = Context {
        native: NativeAnthropicRequest::decode(&native_body(), &TranslationPolicy::default())?,
    };
    assert_eq!(
        context.native.request().model.as_deref(),
        Some("source-model")
    );
    assert_eq!(
        format!("{:?}", context.native),
        "NativeAnthropicRequest { .. }"
    );
    assert_eq!(
        format!("{context:?}"),
        "Context { native: NativeAnthropicRequest { .. } }"
    );
    Ok(())
}

#[test]
fn opaque_native_fields_are_data_not_authority_for_host_extension_changes() -> TestResult {
    let policy = TranslationPolicy::default();
    let mut body = native_body();
    body["algorithm_state"] = json!({"opaque_caller_data": [1, 2]});
    body["previous_response_id"] = json!("opaque-caller-state");
    let native = NativeAnthropicRequest::decode(&body, &policy)?;
    assert_eq!(native.encode(native.request(), &policy)?, body);

    for (key, value) in [
        ("algorithm_state", json!({"opaque_host_data": true})),
        ("future_request", json!({"opaque_host_data": true})),
        ("switchyard_codex_tool_namespaces", json!({"opaque": true})),
        ("previous_response_id", json!("synthetic-state")),
        ("classifier_state", json!({"opaque": true})),
    ] {
        let mut edited = native.request().clone();
        edited.extensions.fields.insert(key.to_string(), value);
        assert!(native.encode(&edited, &policy).is_err());
    }
    let mut edited = native.request().clone();
    edited.extensions.fields.remove("algorithm_state");
    edited.extensions.fields.remove("previous_response_id");
    body.as_object_mut().unwrap().remove("algorithm_state");
    body.as_object_mut().unwrap().remove("previous_response_id");
    assert_eq!(native.encode(&edited, &policy)?, body);

    // A caller field with the same name cannot turn a normalized control into
    // a supported top-level policy edit.
    body["parallel_tool_calls"] = json!(true);
    body["tool_choice"] = json!({"type": "auto", "disable_parallel_tool_use": false});
    let native = NativeAnthropicRequest::decode(&body, &policy)?;
    let mut edited = native.request().clone();
    edited.extensions.fields.remove("parallel_tool_calls");
    assert!(native.encode(&edited, &policy).is_err());
    Ok(())
}

#[test]
fn native_internal_bookkeeping_keys_are_rejected_before_routing() -> TestResult {
    for key in [
        "_switchyard_translation",
        "switchyard_anthropic_request",
        "switchyard_codex_tool_namespaces",
        "switchyard_codex_custom_tools",
        "switchyard_codex_additional_tools",
        "switchyard_codex_custom_call_outputs",
    ] {
        let mut body = native_body();
        body[key] = json!({"opaque": true});
        assert!(matches!(
            NativeAnthropicRequest::decode(&body, &TranslationPolicy::default()),
            Err(TranslationError::InvalidValue { .. })
        ));
    }
    Ok(())
}

#[test]
fn declared_restrictions_fail_even_when_lossy_conversion_is_allowed() -> TestResult {
    let native = NativeAnthropicRequest::decode(&native_body(), &TranslationPolicy::default())?;
    for caps in [
        TargetCapabilities {
            supports_tools: Some(false),
            ..Default::default()
        },
        TargetCapabilities {
            supports_images: Some(false),
            ..Default::default()
        },
        TargetCapabilities {
            supports_reasoning_effort: Some(false),
            ..Default::default()
        },
        TargetCapabilities {
            supports_json_schema_response_format: Some(false),
            ..Default::default()
        },
        TargetCapabilities {
            supports_parallel_tool_calls: Some(false),
            ..Default::default()
        },
    ] {
        let policy = TranslationPolicy {
            target_capabilities: caps,
            lossy_conversion_policy: LossyConversionPolicy::AllowWithDiagnostics,
            ..Default::default()
        };
        assert!(native.encode(native.request(), &policy).is_err());
    }
    // A typed native tool without a normalized name must still obey restrictions.
    let mut body = json!({"model": "source-model", "max_tokens": 64, "messages": [],
        "tools": [{"type": "future_tool", "opaque": true}]});
    let native = NativeAnthropicRequest::decode(&body, &TranslationPolicy::default())?;
    assert!(native.request().tools.is_empty());
    let mut policy = TranslationPolicy::default();
    policy.target_capabilities.supports_tools = Some(false);
    assert!(native.encode(native.request(), &policy).is_err());
    policy.target_capabilities.supports_tools = None;
    policy.target_capabilities.supports_code_execution = Some(false);
    assert!(native.encode(native.request(), &policy).is_err());
    body.as_object_mut().unwrap().remove("tools");
    let plain = NativeAnthropicRequest::decode(&body, &policy)?;
    assert_eq!(plain.encode(plain.request(), &policy)?, body);
    Ok(())
}

#[test]
fn native_body_is_not_carried_into_foreign_translation() -> TestResult {
    let policy = TranslationPolicy {
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        ..Default::default()
    };
    let native = NativeAnthropicRequest::decode(&native_body(), &policy)?;
    let engine = TranslationEngine::default();
    // Opaque content has no approved foreign mapping. Strict translation rejects it.
    assert!(
        engine
            .encode_request(WireFormat::OpenAiChat, native.request(), &policy)
            .is_err()
    );
    let body = json!({"model": "source-model", "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
        "future_request": {"opaque": true}, "context_management": {"opaque": true},
        "thinking": {"type": "enabled", "budget_tokens": 128}});
    let native = NativeAnthropicRequest::decode(&body, &policy)?;
    for target in [
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
        WireFormat::BedrockConverse,
    ] {
        let encoded = engine
            .encode_request(target, native.request(), &policy)?
            .body;
        assert!(encoded.get("future_request").is_none());
        assert!(encoded.get("context_management").is_none());
        assert!(encoded.get("thinking").is_none());
    }
    Ok(())
}

#[test]
fn effort_removal_and_native_parallel_tool_restrictions_can_succeed() -> TestResult {
    let mut body = native_body();
    body["tool_choice"] = json!({"type": "auto", "disable_parallel_tool_use": true,
        "future": {"opaque": true}});
    let mut policy = TranslationPolicy::default();
    policy.target_capabilities.supports_parallel_tool_calls = Some(false);
    policy.target_capabilities.supports_reasoning_effort = Some(false);
    let native = NativeAnthropicRequest::decode(&body, &policy)?;
    let mut edited = native.request().clone();
    edited.reasoning.effort = None;
    body["output_config"]
        .as_object_mut()
        .unwrap()
        .remove("effort");
    assert_eq!(native.encode(&edited, &policy)?, body);
    Ok(())
}

#[test]
fn content_restrictions_cover_opaque_tool_results_and_system_blocks() -> TestResult {
    let mut policy = TranslationPolicy::default();
    policy.target_capabilities.supports_images = Some(false);
    for body in [
        json!({"model": "source-model", "max_tokens": 64, "messages": [],
            "system": [{"type": "future_system", "opaque": true}]}),
        json!({"model": "source-model", "max_tokens": 64, "messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_example",
                "content": [{"type": "future_content", "opaque": true}]}]}]}),
    ] {
        let native = NativeAnthropicRequest::decode(&body, &policy)?;
        assert!(native.encode(native.request(), &policy).is_err());
    }
    Ok(())
}

#[test]
fn reconstruction_baseline_demonstrates_why_checked_native_edits_are_needed() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    let body = native_body();
    let mut request = engine
        .decode_request(WireFormat::AnthropicMessages, &body, &policy)?
        .request;
    request.model = Some("selected-model".to_string());
    request.preservation.requests.clear();
    let rebuilt = engine
        .encode_request(WireFormat::AnthropicMessages, &request, &policy)?
        .body;
    assert!(rebuilt.get("future_request").is_none());
    assert!(rebuilt["system"].is_string());
    assert_ne!(rebuilt["tools"], body["tools"]);
    let native = NativeAnthropicRequest::decode(&body, &policy)?;
    let checked = native.encode(&request, &policy)?;
    assert_eq!(checked["future_request"], body["future_request"]);
    assert_eq!(checked["system"], body["system"]);
    assert_eq!(checked["tools"], body["tools"]);
    Ok(())
}
