// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered codec for Amazon Bedrock Converse request and response JSON.

use serde_json::{Map, Value, json};

use crate::codecs::common::provider_extensions;
use crate::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::llm::{
    AggLlmResponse, ContentBlock, ImageSource, InstructionBlock, LlmRequest, Message, OutputParams,
    ProviderExtensions, ResponseOutput, Role, SamplingParams, StopReason, ToolCall, ToolChoice,
    ToolDefinition, ToolResult, Usage,
};
use crate::policy::TranslationPolicy;
use crate::util::{
    capture_request_preservation, capture_response_preservation, embed_preservation,
    exact_preserved_request, exact_preserved_response, push_lossy, validate_request_capabilities,
};

/// Format codec for Amazon Bedrock's provider-neutral Converse JSON contract.
pub struct BedrockConverseCodec;

impl FormatCodec for BedrockConverseCodec {
    fn format(&self) -> FormatId {
        WireFormat::BedrockConverse.into()
    }

    fn decode_request(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedRequest> {
        let body = crate::util::object(body, "$")?;
        let mut diagnostics = Vec::new();
        let inference = body.get("inferenceConfig").and_then(Value::as_object);
        let mut request = LlmRequest {
            sampling: SamplingParams {
                temperature: inference
                    .and_then(|value| value.get("temperature"))
                    .and_then(Value::as_f64),
                top_p: inference
                    .and_then(|value| value.get("topP"))
                    .and_then(Value::as_f64),
                top_k: None,
            },
            output: OutputParams {
                max_output_tokens: inference
                    .and_then(|value| value.get("maxTokens"))
                    .and_then(Value::as_u64),
                response_format: None,
            },
            preservation: capture_request_preservation(
                WireFormat::BedrockConverse,
                &Value::Object(body.clone()),
                policy,
            ),
            ..LlmRequest::default()
        };

        if let Some(system) = body.get("system").and_then(Value::as_array) {
            let content = decode_content(system, Role::System, &mut diagnostics, policy)?;
            if !content.is_empty() {
                request.instructions.push(InstructionBlock {
                    role: Role::System,
                    content,
                });
            }
        }
        if let Some(messages) = body.get("messages").and_then(Value::as_array) {
            for (index, message) in messages.iter().enumerate() {
                let message =
                    message
                        .as_object()
                        .ok_or_else(|| TranslationError::InvalidValue {
                            path: format!("$.messages[{index}]"),
                            message: "expected an object".to_string(),
                        })?;
                let role = match message.get("role").and_then(Value::as_str) {
                    Some("user") => Role::User,
                    Some("assistant") => Role::Assistant,
                    Some(other) => {
                        return Err(TranslationError::unsupported_role(
                            format!("$.messages[{index}].role"),
                            other,
                        ));
                    }
                    None => Role::User,
                };
                let content = message
                    .get("content")
                    .and_then(Value::as_array)
                    .map(|blocks| decode_content(blocks, role, &mut diagnostics, policy))
                    .transpose()?
                    .unwrap_or_default();
                request.messages.push(Message { role, content });
            }
        }
        if let Some(tool_config) = body.get("toolConfig").and_then(Value::as_object) {
            request.tools = decode_tools(tool_config.get("tools"));
            request.tool_choice = tool_config.get("toolChoice").map(decode_tool_choice);
        }
        if let Some(stop_sequences) = inference.and_then(|value| value.get("stopSequences")) {
            request
                .extensions
                .fields
                .insert("stop".to_string(), stop_sequences.clone());
        }
        if let Some(fields) = body.get("additionalModelRequestFields") {
            request
                .extensions
                .fields
                .insert("additionalModelRequestFields".to_string(), fields.clone());
        }
        request.extensions.fields.extend(provider_extensions(
            body,
            &[
                "messages",
                "system",
                "inferenceConfig",
                "toolConfig",
                "additionalModelRequestFields",
            ],
        ));

        Ok(DecodedRequest {
            request,
            diagnostics,
        })
    }

    fn encode_request(
        &self,
        request: &LlmRequest,
        policy: &TranslationPolicy,
    ) -> Result<EncodedRequest> {
        if let Some(body) =
            exact_preserved_request(&request.preservation, WireFormat::BedrockConverse, policy)
        {
            return Ok(EncodedRequest {
                body,
                diagnostics: Vec::new(),
            });
        }

        let mut diagnostics = Vec::new();
        validate_request_capabilities(request, &mut diagnostics, policy)?;
        let mut body = Map::new();
        let system = request
            .instructions
            .iter()
            .flat_map(|instruction| instruction.content.iter())
            .map(|block| encode_content_block(block, &mut diagnostics, policy))
            .collect::<Result<Vec<_>>>()?;
        if !system.is_empty() {
            body.insert("system".to_string(), Value::Array(system));
        }
        body.insert(
            "messages".to_string(),
            Value::Array(
                request
                    .messages
                    .iter()
                    .map(|message| encode_message(message, &mut diagnostics, policy))
                    .collect::<Result<Vec<_>>>()?,
            ),
        );

        let mut inference = Map::new();
        if let Some(value) = request.output.max_output_tokens {
            inference.insert("maxTokens".to_string(), Value::from(value));
        }
        if let Some(value) = request.sampling.temperature {
            inference.insert("temperature".to_string(), Value::from(value));
        }
        if let Some(value) = request.sampling.top_p {
            inference.insert("topP".to_string(), Value::from(value));
        }
        if let Some(value) = request.extensions.fields.get("stop") {
            inference.insert("stopSequences".to_string(), value.clone());
        }
        if !inference.is_empty() {
            body.insert("inferenceConfig".to_string(), Value::Object(inference));
        }
        if !request.tools.is_empty() || request.tool_choice.is_some() {
            let mut config = Map::new();
            if !request.tools.is_empty() {
                config.insert(
                    "tools".to_string(),
                    Value::Array(request.tools.iter().map(encode_tool).collect()),
                );
            }
            if let Some(choice) = &request.tool_choice {
                config.insert("toolChoice".to_string(), encode_tool_choice(choice));
            }
            body.insert("toolConfig".to_string(), Value::Object(config));
        }
        if let Some(value) = request
            .extensions
            .fields
            .get("additionalModelRequestFields")
        {
            body.insert("additionalModelRequestFields".to_string(), value.clone());
        }

        Ok(EncodedRequest {
            body: embed_preservation(Value::Object(body), &request.preservation, policy),
            diagnostics,
        })
    }

    fn decode_response(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedResponse> {
        let body = crate::util::object(body, "$")?;
        let mut diagnostics = Vec::new();
        let output = body
            .get("output")
            .and_then(Value::as_object)
            .and_then(|output| output.get("message"))
            .and_then(Value::as_object);
        let content = output
            .and_then(|message| message.get("content"))
            .and_then(Value::as_array)
            .map(|blocks| decode_content(blocks, Role::Assistant, &mut diagnostics, policy))
            .transpose()?
            .unwrap_or_default();
        let stop_reason = body
            .get("stopReason")
            .and_then(Value::as_str)
            .map(decode_stop_reason);
        let usage = decode_usage(body.get("usage"));
        let outputs = output
            .map(|message| ResponseOutput {
                role: match message.get("role").and_then(Value::as_str) {
                    Some("user") => Role::User,
                    _ => Role::Assistant,
                },
                content,
                stop_reason,
            })
            .into_iter()
            .collect();

        Ok(DecodedResponse {
            response: AggLlmResponse {
                outputs,
                usage,
                extensions: ProviderExtensions {
                    fields: provider_extensions(body, &["output", "stopReason", "usage"]),
                },
                preservation: capture_response_preservation(
                    WireFormat::BedrockConverse,
                    &Value::Object(body.clone()),
                    policy,
                ),
                ..AggLlmResponse::default()
            },
            diagnostics,
        })
    }

    fn encode_response(
        &self,
        response: &AggLlmResponse,
        policy: &TranslationPolicy,
    ) -> Result<EncodedResponse> {
        if let Some(body) =
            exact_preserved_response(&response.preservation, WireFormat::BedrockConverse, policy)
        {
            return Ok(EncodedResponse {
                body,
                diagnostics: Vec::new(),
            });
        }
        let mut diagnostics = Vec::new();
        let output = response.outputs.first();
        let message = output
            .map(|output| {
                let content = output
                    .content
                    .iter()
                    .map(|block| encode_content_block(block, &mut diagnostics, policy))
                    .collect::<Result<Vec<_>>>()?;
                Ok::<Value, TranslationError>(json!({
                    "role": encode_role(output.role),
                    "content": content
                }))
            })
            .transpose()?;
        let mut body = Map::new();
        if let Some(message) = message {
            body.insert("output".to_string(), json!({"message": message}));
        }
        if let Some(reason) = output.and_then(|output| output.stop_reason) {
            body.insert(
                "stopReason".to_string(),
                Value::String(encode_stop_reason(reason).to_string()),
            );
        }
        let usage = encode_usage(&response.usage);
        if !usage.is_empty() {
            body.insert("usage".to_string(), Value::Object(usage));
        }
        Ok(EncodedResponse {
            body: embed_preservation(Value::Object(body), &response.preservation, policy),
            diagnostics,
        })
    }
}

fn decode_content(
    blocks: &[Value],
    role: Role,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<ContentBlock>> {
    blocks
        .iter()
        .enumerate()
        .map(|(index, block)| decode_content_block(block, role, index, diagnostics, policy))
        .collect()
}

fn decode_content_block(
    block: &Value,
    role: Role,
    index: usize,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<ContentBlock> {
    let object = block
        .as_object()
        .ok_or_else(|| TranslationError::InvalidValue {
            path: format!("$.content[{index}]"),
            message: "expected an object".to_string(),
        })?;
    if let Some(text) = object.get("text").and_then(Value::as_str) {
        return Ok(ContentBlock::Text {
            text: text.to_string(),
        });
    }
    if let Some(tool) = object.get("toolUse").and_then(Value::as_object) {
        return Ok(ContentBlock::ToolCall(ToolCall {
            id: tool
                .get("toolUseId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            name: tool
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            arguments: tool.get("input").cloned().unwrap_or_else(|| json!({})),
        }));
    }
    if let Some(tool) = object.get("toolResult").and_then(Value::as_object) {
        let content = tool
            .get("content")
            .and_then(Value::as_array)
            .map(|blocks| decode_tool_result_content(blocks))
            .unwrap_or_default();
        return Ok(ContentBlock::ToolResult(ToolResult {
            tool_call_id: tool
                .get("toolUseId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            content,
            is_error: tool
                .get("status")
                .and_then(Value::as_str)
                .map(|status| status == "error"),
        }));
    }
    if let Some(reasoning) = object.get("reasoningContent").and_then(Value::as_object) {
        let reasoning = reasoning
            .get("reasoningText")
            .and_then(Value::as_object)
            .unwrap_or(reasoning);
        return Ok(ContentBlock::Reasoning {
            text: reasoning
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            signature: reasoning
                .get("signature")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
        });
    }
    if let Some(image) = object.get("image").and_then(Value::as_object) {
        let format = image.get("format").and_then(Value::as_str);
        let bytes = image
            .get("source")
            .and_then(Value::as_object)
            .and_then(|source| source.get("bytes"))
            .and_then(Value::as_str);
        if let Some(data) = bytes {
            return Ok(ContentBlock::Image {
                source: ImageSource::Base64 {
                    media_type: format.map(|format| format!("image/{format}")),
                    data: data.to_string(),
                },
            });
        }
    }
    push_lossy(
        diagnostics,
        policy,
        format!("unsupported Bedrock Converse content block for {role:?}"),
    )?;
    Ok(ContentBlock::Unknown {
        provider: WireFormat::BedrockConverse.into(),
        raw: block.clone(),
    })
}

fn decode_tool_result_content(blocks: &[Value]) -> Vec<ContentBlock> {
    blocks
        .iter()
        .map(|block| {
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                ContentBlock::Text {
                    text: text.to_string(),
                }
            } else if let Some(value) = block.get("json") {
                ContentBlock::Text {
                    text: value.to_string(),
                }
            } else {
                ContentBlock::Unknown {
                    provider: WireFormat::BedrockConverse.into(),
                    raw: block.clone(),
                }
            }
        })
        .collect()
}

fn encode_message(
    message: &Message,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Value> {
    Ok(json!({
        "role": encode_role(message.role),
        "content": message
            .content
            .iter()
            .map(|block| encode_content_block(block, diagnostics, policy))
            .collect::<Result<Vec<_>>>()?,
    }))
}

fn encode_role(role: Role) -> &'static str {
    match role {
        Role::Assistant => "assistant",
        Role::User | Role::Tool | Role::System | Role::Developer => "user",
    }
}

fn encode_content_block(
    block: &ContentBlock,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Value> {
    Ok(match block {
        ContentBlock::Text { text } | ContentBlock::Refusal { text } => json!({"text": text}),
        ContentBlock::Reasoning { text, signature } => json!({
            "reasoningContent": {"reasoningText": {"text": text, "signature": signature}}
        }),
        ContentBlock::ToolCall(call) => json!({
            "toolUse": {"toolUseId": call.id, "name": call.name, "input": call.arguments}
        }),
        ContentBlock::ToolResult(result) => json!({
            "toolResult": {
                "toolUseId": result.tool_call_id,
                "content": result.content.iter().map(encode_tool_result_block).collect::<Vec<_>>(),
                "status": if result.is_error == Some(true) { "error" } else { "success" }
            }
        }),
        ContentBlock::Image {
            source: ImageSource::Base64 { media_type, data },
        } => {
            let format = media_type
                .as_deref()
                .and_then(|value| value.strip_prefix("image/"))
                .unwrap_or("png");
            json!({"image": {"format": format, "source": {"bytes": data}}})
        }
        ContentBlock::Unknown { provider, raw }
            if provider.as_str() == WireFormat::BedrockConverse.as_str() =>
        {
            raw.clone()
        }
        unsupported => {
            push_lossy(
                diagnostics,
                policy,
                format!("unsupported content block encoded for Bedrock Converse: {unsupported:?}"),
            )?;
            json!({"text": unsupported_text(unsupported)})
        }
    })
}

fn unsupported_text(block: &ContentBlock) -> String {
    match block {
        ContentBlock::Audio { .. } => "[audio]",
        ContentBlock::Video { .. } => "[video]",
        ContentBlock::File { .. } => "[file]",
        ContentBlock::Image { .. } => "[image]",
        _ => "[unsupported content]",
    }
    .to_string()
}

fn encode_tool_result_block(block: &ContentBlock) -> Value {
    match block {
        ContentBlock::Text { text } | ContentBlock::Refusal { text } => json!({"text": text}),
        ContentBlock::Unknown { raw, .. } => json!({"json": raw}),
        other => json!({"text": unsupported_text(other)}),
    }
}

fn decode_tools(value: Option<&Value>) -> Vec<ToolDefinition> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| tool.get("toolSpec").and_then(Value::as_object))
        .filter_map(|tool| {
            let name = tool.get("name")?.as_str()?.to_string();
            Some(ToolDefinition {
                name,
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                parameters: tool
                    .get("inputSchema")
                    .and_then(|schema| schema.get("json"))
                    .cloned()
                    .unwrap_or_else(|| json!({})),
                strict: None,
            })
        })
        .collect()
}

fn encode_tool(tool: &ToolDefinition) -> Value {
    json!({
        "toolSpec": {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": {"json": tool.parameters},
        }
    })
}

fn decode_tool_choice(value: &Value) -> ToolChoice {
    if value.get("auto").is_some() {
        ToolChoice::Auto
    } else if value.get("any").is_some() {
        ToolChoice::Required
    } else if let Some(name) = value
        .get("tool")
        .and_then(Value::as_object)
        .and_then(|tool| tool.get("name"))
        .and_then(Value::as_str)
    {
        ToolChoice::Tool {
            name: name.to_string(),
        }
    } else {
        ToolChoice::Raw(value.clone())
    }
}

fn encode_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({"auto": {}}),
        ToolChoice::Required => json!({"any": {}}),
        ToolChoice::Tool { name } => json!({"tool": {"name": name}}),
        ToolChoice::None => json!({"none": {}}),
        ToolChoice::Raw(value) => value.clone(),
    }
}

fn decode_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "stop_sequence" => StopReason::EndTurn,
        "max_tokens" => StopReason::MaxTokens,
        "tool_use" => StopReason::ToolUse,
        "content_filtered" | "guardrail_intervened" => StopReason::ContentFilter,
        _ => StopReason::Unknown,
    }
}

fn encode_stop_reason(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndTurn => "end_turn",
        StopReason::MaxTokens => "max_tokens",
        StopReason::ToolUse => "tool_use",
        StopReason::ContentFilter => "content_filtered",
        StopReason::Error => "error",
        StopReason::Unknown => "unknown",
    }
}

fn decode_usage(value: Option<&Value>) -> Usage {
    let value = value.and_then(Value::as_object);
    let mut usage = Usage {
        input_tokens: value
            .and_then(|value| value.get("inputTokens"))
            .and_then(Value::as_u64),
        output_tokens: value
            .and_then(|value| value.get("outputTokens"))
            .and_then(Value::as_u64),
        total_tokens: value
            .and_then(|value| value.get("totalTokens"))
            .and_then(Value::as_u64),
        ..Usage::default()
    };
    if let Some(value) = value
        .and_then(|value| value.get("cacheReadInputTokens"))
        .and_then(Value::as_u64)
    {
        usage.set_cached_input_tokens(value);
    }
    if let Some(value) = value
        .and_then(|value| value.get("cacheWriteInputTokens"))
        .and_then(Value::as_u64)
    {
        usage.set_cache_creation_input_tokens(value);
    }
    usage
}

fn encode_usage(usage: &Usage) -> Map<String, Value> {
    let mut value = Map::new();
    if let Some(tokens) = usage.input_tokens {
        value.insert("inputTokens".to_string(), Value::from(tokens));
    }
    if let Some(tokens) = usage.output_tokens {
        value.insert("outputTokens".to_string(), Value::from(tokens));
    }
    if let Some(tokens) = usage.total_tokens {
        value.insert("totalTokens".to_string(), Value::from(tokens));
    }
    if let Some(tokens) = usage.cached_input_tokens() {
        value.insert("cacheReadInputTokens".to_string(), Value::from(tokens));
    }
    if let Some(tokens) = usage.cache_creation_input_tokens() {
        value.insert("cacheWriteInputTokens".to_string(), Value::from(tokens));
    }
    value
}
