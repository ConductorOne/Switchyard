// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked edits to a native Anthropic request without rebuilding unchanged fields.

use serde_json::{Map, Value};
use switchyard_protocol::codex_namespaces::TOOL_NAMESPACES_KEY;

use crate::codecs::common::ANTHROPIC_REQUEST_KEY;
use crate::codex_custom_tools::{ADDITIONAL_TOOLS_KEY, CUSTOM_CALL_OUTPUTS_KEY, CUSTOM_TOOLS_KEY};
use crate::util::{SWITCHYARD_METADATA_KEY, validate_request_capabilities};
use crate::{
    ContentBlock, LlmRequest, LossyConversionPolicy, PreservationPolicy, Result, TranslationEngine,
    TranslationError, TranslationPolicy, WireFormat,
};

/// Owns a native Anthropic body and the normalized request decoded from it.
///
/// Keep this value outside the mutable IR. Use [`Self::encode`] after routing and
/// policy edits, even when those edits cleared request preservation. Only the
/// supported edits are applied; other changes fail rather than replay stale JSON.
/// Debug output is redacted. This type has no serialization or deserialization
/// support and cannot encode a foreign provider format.
pub struct NativeAnthropicRequest {
    body: Value,
    request: LlmRequest,
}

impl std::fmt::Debug for NativeAnthropicRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeAnthropicRequest")
            .finish_non_exhaustive()
    }
}

impl NativeAnthropicRequest {
    /// Decodes native input. Embedded preservation is rejected, never trusted.
    ///
    /// Preservation is disabled in the returned IR so ordinary translation cannot
    /// recover this native body from a client-controlled metadata envelope.
    pub fn decode(body: &Value, policy: &TranslationPolicy) -> Result<Self> {
        for key in [
            SWITCHYARD_METADATA_KEY,
            ANTHROPIC_REQUEST_KEY,
            TOOL_NAMESPACES_KEY,
            CUSTOM_TOOLS_KEY,
            ADDITIONAL_TOOLS_KEY,
            CUSTOM_CALL_OUTPUTS_KEY,
        ] {
            if body.get(key).is_some() {
                return Err(TranslationError::InvalidValue {
                    path: format!("$.{key}"),
                    message: "internal request metadata is not accepted as native input"
                        .to_string(),
                });
            }
        }
        if body
            .get("metadata")
            .and_then(Value::as_object)
            .is_some_and(|metadata| metadata.contains_key(SWITCHYARD_METADATA_KEY))
        {
            return Err(TranslationError::InvalidValue {
                path: "$.metadata._switchyard_translation".to_string(),
                message: "embedded request preservation is not accepted".to_string(),
            });
        }
        let policy = native_policy(policy);
        let request = TranslationEngine::default()
            .decode_request(WireFormat::AnthropicMessages, body, &policy)?
            .request;
        Ok(Self {
            body: body.clone(),
            request,
        })
    }

    /// Returns the decoded baseline. Clone it before applying routing or policy.
    pub fn request(&self) -> &LlmRequest {
        &self.request
    }

    /// Encodes checked edits for the same native Anthropic target.
    ///
    /// Supports model, stream, sampling, maximum output tokens, raw thinking,
    /// reasoning effort, and removal of existing native top-level extensions.
    /// Changing instructions, messages, tools, tool choice, response format,
    /// decision data, or extension values returns an error. Added extensions are
    /// also rejected. Edited preservation is ignored.
    /// Declared target capability restrictions always reject incompatible input,
    /// even when the caller permits lossy translation.
    pub fn encode(&self, edited: &LlmRequest, policy: &TranslationPolicy) -> Result<Value> {
        // A full comparison also guards newly added IR fields: they are unsupported
        // until this seam has an explicit native mapping for them.
        let mut supported = self.request.clone();
        supported.model.clone_from(&edited.model);
        supported.stream = edited.stream;
        supported.sampling.clone_from(&edited.sampling);
        supported.output.max_output_tokens = edited.output.max_output_tokens;
        supported.reasoning.clone_from(&edited.reasoning);
        supported.extensions.clone_from(&edited.extensions);
        supported.preservation.clone_from(&edited.preservation);
        if supported != *edited {
            return Err(TranslationError::LossyConversion(
                "native Anthropic structural and decision edits are unsupported".to_string(),
            ));
        }

        let policy = native_policy(policy);
        let mut body = self.body.clone();
        let object = body.as_object_mut().ok_or_else(unsupported_edit)?;
        replace_changed(object, "model", &self.request.model, &edited.model)?;
        replace_changed(object, "stream", &self.request.stream, &edited.stream)?;
        replace_changed(
            object,
            "max_tokens",
            &self.request.output.max_output_tokens,
            &edited.output.max_output_tokens,
        )?;
        replace_changed(
            object,
            "temperature",
            &self.request.sampling.temperature,
            &edited.sampling.temperature,
        )?;
        replace_changed(
            object,
            "top_p",
            &self.request.sampling.top_p,
            &edited.sampling.top_p,
        )?;
        replace_changed(
            object,
            "top_k",
            &self.request.sampling.top_k,
            &edited.sampling.top_k,
        )?;
        replace_changed(
            object,
            "thinking",
            &self.request.reasoning.raw,
            &edited.reasoning.raw,
        )?;
        if self.request.reasoning.effort != edited.reasoning.effort {
            let config = object
                .entry("output_config")
                .or_insert_with(|| Value::Object(Map::new()));
            let config = config.as_object_mut().ok_or_else(unsupported_edit)?;
            replace_changed(
                config,
                "effort",
                &self.request.reasoning.effort,
                &edited.reasoning.effort,
            )?;
        }

        for (key, before) in &self.request.extensions.fields {
            let after = edited.extensions.fields.get(key);
            if after == Some(before) {
                continue;
            }
            // These are normalized/internal fields, not native top-level keys.
            if key == ANTHROPIC_REQUEST_KEY {
                if after.is_some() {
                    return Err(unsupported_extension_edit());
                }
                continue;
            }
            if after.is_some() || key == "parallel_tool_calls" || !object.contains_key(key) {
                return Err(unsupported_extension_edit());
            }
            object.remove(key);
        }
        if edited
            .extensions
            .fields
            .keys()
            .any(|key| !self.request.extensions.fields.contains_key(key))
        {
            return Err(unsupported_extension_edit());
        }
        // Check the emitted native values, not stale or injected preservation.
        let final_request = Self::decode(&body, &policy)?;
        validate_request_capabilities(final_request.request(), &mut Vec::new(), &policy)?;
        validate_native_capabilities(&body, final_request.request(), &policy)?;
        Ok(body)
    }
}

fn native_policy(policy: &TranslationPolicy) -> TranslationPolicy {
    TranslationPolicy {
        preservation: PreservationPolicy::Disabled,
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        ..policy.clone()
    }
}

fn unsupported_edit() -> TranslationError {
    TranslationError::LossyConversion(
        "native Anthropic edit cannot be reconciled without rebuilding unchanged fields"
            .to_string(),
    )
}

fn unsupported_extension_edit() -> TranslationError {
    TranslationError::LossyConversion(
        "native Anthropic extensions support removal only; added, changed, or synthetic extension state is unsupported"
            .to_string(),
    )
}

// Some native tools and content have no normalized capability representation.
// A restrictive profile must not accidentally accept them because decode omitted
// a tool definition or retained a content block as opaque JSON.
fn validate_native_capabilities(
    body: &Value,
    request: &LlmRequest,
    policy: &TranslationPolicy,
) -> Result<()> {
    let caps = &policy.target_capabilities;
    let tools = body.get("tools").and_then(Value::as_array);
    let has_tools = tools.is_some_and(|tools| !tools.is_empty());
    if caps.supports_tools == Some(false) && has_tools {
        return Err(unsupported_edit());
    }
    if caps.supports_code_execution == Some(false)
        && tools.is_some_and(|tools| tools.iter().any(|tool| tool.get("type").is_some()))
    {
        return Err(unsupported_edit());
    }
    if caps.supports_safety_settings == Some(false) && body.get("safety_settings").is_some() {
        return Err(unsupported_edit());
    }
    if caps.supports_parallel_tool_calls == Some(false)
        && has_tools
        && body.pointer("/tool_choice/type").and_then(Value::as_str) != Some("none")
        && body
            .pointer("/tool_choice/disable_parallel_tool_use")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(unsupported_edit());
    }
    let restricts_content = [
        caps.supports_tools,
        caps.supports_images,
        caps.supports_audio,
        caps.supports_video,
        caps.supports_files,
        caps.supports_code_execution,
    ]
    .contains(&Some(false));
    if restricts_content
        && (request
            .messages
            .iter()
            .any(|message| has_opaque_content(&message.content))
            || body
                .get("system")
                .and_then(Value::as_array)
                .is_some_and(|blocks| {
                    blocks
                        .iter()
                        .any(|block| block.get("type").and_then(Value::as_str) != Some("text"))
                }))
    {
        return Err(unsupported_edit());
    }
    Ok(())
}

fn has_opaque_content(blocks: &[ContentBlock]) -> bool {
    blocks.iter().any(|block| match block {
        ContentBlock::Unknown { .. } => true,
        ContentBlock::ToolResult(result) => has_opaque_content(&result.content),
        _ => false,
    })
}

fn replace_changed<T: PartialEq + serde::Serialize>(
    object: &mut Map<String, Value>,
    key: &str,
    before: &T,
    after: &T,
) -> Result<()> {
    if before != after {
        let value = serde_json::to_value(after)?;
        if value.is_null() {
            object.remove(key);
        } else {
            object.insert(key.to_string(), value);
        }
    }
    Ok(())
}
