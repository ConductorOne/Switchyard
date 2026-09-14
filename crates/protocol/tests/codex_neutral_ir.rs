// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable serialization and exhaustive-match coverage for native provider-neutral IR.

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use switchyard_protocol::{
    AggLlmResponse, ComputerAction, ComputerEnvironment, ComputerMouseButton, ContentBlock,
    CustomToolFormat, FileSource, GrammarSyntax, HostedTool, ImageSource, LlmRequest,
    LlmResponseChunk, MAX_OPAQUE_STATE_BYTES, MediaSource, Message, OpaqueState,
    OpaqueStatePurpose, ProviderErrorClass, ResponseMetadata, ResponseOutput, ResponseTerminal,
    Role, SpecializedToolDefinition, StreamSemanticClass, TurnInputItem,
};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn decode_round_trip<T>(value: &Value) -> Result<T, serde_json::Error>
where
    T: DeserializeOwned + Serialize,
{
    let decoded = serde_json::from_value(value.clone())?;
    assert_eq!(serde_json::to_value(&decoded)?, *value);
    Ok(decoded)
}

fn content_block_kind(block: &ContentBlock) -> &'static str {
    match block {
        ContentBlock::Text { .. } => "text",
        ContentBlock::Reasoning { .. } => "reasoning",
        ContentBlock::Image { .. } => "image",
        ContentBlock::Audio { .. } => "audio",
        ContentBlock::Video { .. } => "video",
        ContentBlock::File { .. } => "file",
        ContentBlock::ToolCall(_) => "tool_call",
        ContentBlock::ToolResult(_) => "tool_result",
        ContentBlock::CustomToolCall(_) => "custom_tool_call",
        ContentBlock::CustomToolResult(_) => "custom_tool_result",
        ContentBlock::ComputerToolCall(_) => "computer_tool_call",
        ContentBlock::ComputerToolResult(_) => "computer_tool_result",
        ContentBlock::HostedToolCall(_) => "hosted_tool_call",
        ContentBlock::HostedToolResult(_) => "hosted_tool_result",
        ContentBlock::OpaqueState(_) => "opaque_state",
        ContentBlock::Compaction(_) => "compaction",
        ContentBlock::GeneratedImage(_) => "generated_image",
        ContentBlock::PauseTurn(_) => "pause_turn",
        ContentBlock::Refusal { .. } => "refusal",
        ContentBlock::Unknown { .. } => "unknown",
    }
}

fn chunk_kind(chunk: &LlmResponseChunk) -> &'static str {
    match chunk {
        LlmResponseChunk::MessageStart { .. } => "message_start",
        LlmResponseChunk::TextDelta { .. } => "text_delta",
        LlmResponseChunk::ReasoningDelta { .. } => "reasoning_delta",
        LlmResponseChunk::ReasoningDetailsDelta { .. } => "reasoning_details_delta",
        LlmResponseChunk::ToolCallDelta { .. } => "tool_call_delta",
        LlmResponseChunk::ToolCallDone { .. } => "tool_call_done",
        LlmResponseChunk::CustomToolCallDelta { .. } => "custom_tool_call_delta",
        LlmResponseChunk::CustomToolCallDone { .. } => "custom_tool_call_done",
        LlmResponseChunk::ComputerToolCallDone { .. } => "computer_tool_call_done",
        LlmResponseChunk::HostedToolCallDone { .. } => "hosted_tool_call_done",
        LlmResponseChunk::ReasoningStarted { .. } => "reasoning_started",
        LlmResponseChunk::ReasoningDone { .. } => "reasoning_done",
        LlmResponseChunk::RefusalDelta { .. } => "refusal_delta",
        LlmResponseChunk::RefusalDone { .. } => "refusal_done",
        LlmResponseChunk::TextDone { .. } => "text_done",
        LlmResponseChunk::OpaqueState(_) => "opaque_state",
        LlmResponseChunk::ResponseMetadata(_) => "response_metadata",
        LlmResponseChunk::GeneratedImageDone(_) => "generated_image_done",
        LlmResponseChunk::CompactionDone(_) => "compaction_done",
        LlmResponseChunk::PauseTurn(_) => "pause_turn",
        LlmResponseChunk::ResponseTerminal(_) => "response_terminal",
        LlmResponseChunk::Usage(_) => "usage",
        LlmResponseChunk::MessageStop { .. } => "message_stop",
        LlmResponseChunk::DecodeError { .. } => "decode_error",
        LlmResponseChunk::StreamError { .. } => "stream_error",
    }
}

fn specialized_tool_kind(tool: &SpecializedToolDefinition) -> &'static str {
    match tool {
        SpecializedToolDefinition::Custom { .. } => "custom",
        SpecializedToolDefinition::Computer { .. } => "computer",
        SpecializedToolDefinition::Hosted { .. } => "hosted",
    }
}

fn turn_input_kind(item: &TurnInputItem) -> &'static str {
    match item {
        TurnInputItem::ConfigurationUpdate(_) => "configuration_update",
        TurnInputItem::RemoteCompaction => "remote_compaction",
    }
}

fn terminal_kind(terminal: &ResponseTerminal) -> &'static str {
    match terminal {
        ResponseTerminal::Completed => "completed",
        ResponseTerminal::Incomplete { .. } => "incomplete",
        ResponseTerminal::Failed(_) => "failed",
        ResponseTerminal::Paused(_) => "paused",
    }
}

fn provider_error_kind(class: ProviderErrorClass) -> &'static str {
    match class {
        ProviderErrorClass::Authentication => "authentication",
        ProviderErrorClass::Authorization => "authorization",
        ProviderErrorClass::RateLimited => "rate_limited",
        ProviderErrorClass::InvalidRequest => "invalid_request",
        ProviderErrorClass::ContextWindow => "context_window",
        ProviderErrorClass::ContentFiltered => "content_filtered",
        ProviderErrorClass::Unavailable => "unavailable",
        ProviderErrorClass::Protocol => "protocol",
        ProviderErrorClass::Internal => "internal",
    }
}

fn computer_action_kind(action: &ComputerAction) -> &'static str {
    match action {
        ComputerAction::Click { .. } => "click",
        ComputerAction::DoubleClick { .. } => "double_click",
        ComputerAction::Drag { .. } => "drag",
        ComputerAction::KeyPress { .. } => "key_press",
        ComputerAction::Move { .. } => "move",
        ComputerAction::Screenshot => "screenshot",
        ComputerAction::Scroll { .. } => "scroll",
        ComputerAction::Type { .. } => "type",
        ComputerAction::Wait => "wait",
    }
}

fn hosted_tool_kind(tool: &HostedTool) -> &'static str {
    match tool {
        HostedTool::WebSearch { .. } => "web_search",
        HostedTool::FileSearch { .. } => "file_search",
        HostedTool::CodeInterpreter { .. } => "code_interpreter",
        HostedTool::ImageGeneration => "image_generation",
    }
}

fn custom_format_kind(format: &CustomToolFormat) -> &'static str {
    match format {
        CustomToolFormat::Text => "text",
        CustomToolFormat::Grammar {
            syntax: GrammarSyntax::Lark,
            ..
        } => "lark",
        CustomToolFormat::Grammar {
            syntax: GrammarSyntax::Regex,
            ..
        } => "regex",
    }
}

fn environment_kind(environment: ComputerEnvironment) -> &'static str {
    match environment {
        ComputerEnvironment::Browser => "browser",
        ComputerEnvironment::Desktop => "desktop",
        ComputerEnvironment::Mobile => "mobile",
    }
}

fn mouse_button_kind(button: ComputerMouseButton) -> &'static str {
    match button {
        ComputerMouseButton::Left => "left",
        ComputerMouseButton::Middle => "middle",
        ComputerMouseButton::Right => "right",
    }
}

fn opaque_state_kind(purpose: OpaqueStatePurpose) -> &'static str {
    match purpose {
        OpaqueStatePurpose::Reasoning => "reasoning",
        OpaqueStatePurpose::Conversation => "conversation",
        OpaqueStatePurpose::Compaction => "compaction",
        OpaqueStatePurpose::TurnRouting => "turn_routing",
    }
}

fn block_uses_escape_hatch(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Image {
            source: ImageSource::Raw(_),
        }
        | ContentBlock::GeneratedImage(switchyard_protocol::GeneratedImage {
            source: ImageSource::Raw(_),
            ..
        })
        | ContentBlock::ComputerToolResult(switchyard_protocol::ComputerToolResult {
            output: ImageSource::Raw(_),
            ..
        })
        | ContentBlock::File {
            source: FileSource::Raw(_),
        }
        | ContentBlock::Audio {
            source: MediaSource::Raw(_),
        }
        | ContentBlock::Video {
            source: MediaSource::Raw(_),
        }
        | ContentBlock::Unknown { .. } => true,
        ContentBlock::ToolResult(result) => result.content.iter().any(block_uses_escape_hatch),
        ContentBlock::HostedToolResult(result) => {
            result.content.iter().any(block_uses_escape_hatch)
        }
        ContentBlock::Text { .. }
        | ContentBlock::Reasoning { .. }
        | ContentBlock::Image { .. }
        | ContentBlock::Audio { .. }
        | ContentBlock::Video { .. }
        | ContentBlock::File { .. }
        | ContentBlock::ToolCall(_)
        | ContentBlock::CustomToolCall(_)
        | ContentBlock::CustomToolResult(_)
        | ContentBlock::ComputerToolCall(_)
        | ContentBlock::ComputerToolResult(_)
        | ContentBlock::HostedToolCall(_)
        | ContentBlock::OpaqueState(_)
        | ContentBlock::Compaction(_)
        | ContentBlock::GeneratedImage(_)
        | ContentBlock::PauseTurn(_)
        | ContentBlock::Refusal { .. } => false,
    }
}
#[test]
fn codex_neutral_ir_fixture_round_trips() -> TestResult {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/codex_neutral_ir.json"))?;

    let blocks: Vec<ContentBlock> = decode_round_trip(&fixture["content_blocks"])?;
    assert_eq!(
        blocks.iter().map(content_block_kind).collect::<Vec<_>>(),
        [
            "custom_tool_call",
            "custom_tool_result",
            "computer_tool_call",
            "computer_tool_result",
            "hosted_tool_call",
            "hosted_tool_result",
            "opaque_state",
            "compaction",
            "generated_image",
            "pause_turn",
        ]
    );

    let tools: Vec<SpecializedToolDefinition> = decode_round_trip(&fixture["specialized_tools"])?;
    assert_eq!(
        tools.iter().map(specialized_tool_kind).collect::<Vec<_>>(),
        [
            "custom", "custom", "custom", "computer", "computer", "computer", "hosted", "hosted",
            "hosted", "hosted",
        ]
    );

    let trailing: Vec<TurnInputItem> = decode_round_trip(&fixture["trailing_items"])?;
    assert_eq!(
        trailing.iter().map(turn_input_kind).collect::<Vec<_>>(),
        ["configuration_update", "remote_compaction"]
    );

    let _: switchyard_protocol::ResponseMetadata = decode_round_trip(&fixture["metadata"])?;
    let terminals: Vec<ResponseTerminal> = decode_round_trip(&fixture["terminals"])?;
    assert_eq!(
        terminals.iter().map(terminal_kind).collect::<Vec<_>>(),
        ["completed", "incomplete", "failed", "paused"]
    );

    let chunks: Vec<LlmResponseChunk> = decode_round_trip(&fixture["chunks"])?;
    assert_eq!(
        chunks.iter().map(chunk_kind).collect::<Vec<_>>(),
        [
            "tool_call_done",
            "custom_tool_call_delta",
            "custom_tool_call_done",
            "computer_tool_call_done",
            "hosted_tool_call_done",
            "reasoning_started",
            "reasoning_done",
            "refusal_delta",
            "refusal_done",
            "text_done",
            "opaque_state",
            "response_metadata",
            "generated_image_done",
            "compaction_done",
            "pause_turn",
            "response_terminal",
        ]
    );
    assert_eq!(
        chunks
            .iter()
            .map(LlmResponseChunk::semantic_class)
            .collect::<Vec<_>>(),
        [
            StreamSemanticClass::Semantic,
            StreamSemanticClass::NonSemantic,
            StreamSemanticClass::Semantic,
            StreamSemanticClass::Semantic,
            StreamSemanticClass::Semantic,
            StreamSemanticClass::NonSemantic,
            StreamSemanticClass::Semantic,
            StreamSemanticClass::Semantic,
            StreamSemanticClass::Semantic,
            StreamSemanticClass::Semantic,
            StreamSemanticClass::NonSemantic,
            StreamSemanticClass::NonSemantic,
            StreamSemanticClass::Semantic,
            StreamSemanticClass::Semantic,
            StreamSemanticClass::Semantic,
            StreamSemanticClass::Semantic,
        ]
    );

    Ok(())
}

#[test]
fn nested_enum_variants_remain_exhaustive() -> TestResult {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/codex_neutral_ir.json"))?;
    let blocks: Vec<ContentBlock> = serde_json::from_value(fixture["content_blocks"].clone())?;
    let ContentBlock::ComputerToolCall(call) = &blocks[2] else {
        panic!("fixture must contain the computer call at index 2");
    };
    assert_eq!(
        call.actions
            .iter()
            .map(computer_action_kind)
            .collect::<Vec<_>>(),
        [
            "click",
            "double_click",
            "click",
            "drag",
            "key_press",
            "move",
            "screenshot",
            "scroll",
            "type",
            "wait",
        ]
    );

    let tools: Vec<SpecializedToolDefinition> =
        serde_json::from_value(fixture["specialized_tools"].clone())?;
    let custom_formats = tools.iter().filter_map(|tool| match tool {
        SpecializedToolDefinition::Custom { format, .. } => Some(custom_format_kind(format)),
        SpecializedToolDefinition::Computer { .. } | SpecializedToolDefinition::Hosted { .. } => {
            None
        }
    });
    assert_eq!(
        custom_formats.collect::<Vec<_>>(),
        ["text", "lark", "regex"]
    );

    let environments = tools.iter().filter_map(|tool| match tool {
        SpecializedToolDefinition::Computer { configuration } => {
            Some(environment_kind(configuration.environment))
        }
        SpecializedToolDefinition::Custom { .. } | SpecializedToolDefinition::Hosted { .. } => None,
    });
    assert_eq!(
        environments.collect::<Vec<_>>(),
        ["browser", "desktop", "mobile"]
    );

    let hosted = tools.iter().filter_map(|tool| match tool {
        SpecializedToolDefinition::Hosted { tool } => Some(hosted_tool_kind(tool)),
        SpecializedToolDefinition::Custom { .. } | SpecializedToolDefinition::Computer { .. } => {
            None
        }
    });
    assert_eq!(
        hosted.collect::<Vec<_>>(),
        [
            "web_search",
            "file_search",
            "code_interpreter",
            "image_generation"
        ]
    );

    assert_eq!(
        [
            ComputerMouseButton::Left,
            ComputerMouseButton::Middle,
            ComputerMouseButton::Right,
        ]
        .map(mouse_button_kind),
        ["left", "middle", "right"]
    );
    assert_eq!(
        [
            OpaqueStatePurpose::Reasoning,
            OpaqueStatePurpose::Conversation,
            OpaqueStatePurpose::Compaction,
            OpaqueStatePurpose::TurnRouting,
        ]
        .map(opaque_state_kind),
        ["reasoning", "conversation", "compaction", "turn_routing"]
    );
    assert_eq!(
        [
            ProviderErrorClass::Authentication,
            ProviderErrorClass::Authorization,
            ProviderErrorClass::RateLimited,
            ProviderErrorClass::InvalidRequest,
            ProviderErrorClass::ContextWindow,
            ProviderErrorClass::ContentFiltered,
            ProviderErrorClass::Unavailable,
            ProviderErrorClass::Protocol,
            ProviderErrorClass::Internal,
        ]
        .map(provider_error_kind),
        [
            "authentication",
            "authorization",
            "rate_limited",
            "invalid_request",
            "context_window",
            "content_filtered",
            "unavailable",
            "protocol",
            "internal",
        ]
    );

    Ok(())
}

#[test]
fn new_default_fields_preserve_existing_serialized_forms() -> TestResult {
    let request = serde_json::to_value(LlmRequest::default())?;
    assert!(request.get("trailing_items").is_none());
    assert!(request.get("specialized_tools").is_none());

    let response = serde_json::to_value(switchyard_protocol::AggLlmResponse::default())?;
    assert!(response.get("metadata").is_none());
    assert!(response.get("terminal").is_none());
    Ok(())
}

#[test]
fn opaque_state_enforces_its_serialized_size_bound() -> TestResult {
    let at_limit = "x".repeat(MAX_OPAQUE_STATE_BYTES);
    let state = OpaqueState::new(OpaqueStatePurpose::Conversation, at_limit)?;
    let _: OpaqueState = serde_json::from_value(serde_json::to_value(state)?)?;

    let over_limit = "x".repeat(MAX_OPAQUE_STATE_BYTES + 1);
    assert!(OpaqueState::new(OpaqueStatePurpose::Conversation, over_limit.clone()).is_err());
    assert!(
        serde_json::from_value::<OpaqueState>(serde_json::json!({
            "purpose": "conversation",
            "data": over_limit,
        }))
        .is_err()
    );
    Ok(())
}

#[test]
fn codex_parity_fixture_needs_no_escape_hatches() -> TestResult {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/codex_neutral_ir.json"))?;
    let blocks: Vec<ContentBlock> = serde_json::from_value(fixture["content_blocks"].clone())?;
    let specialized_tools: Vec<SpecializedToolDefinition> =
        serde_json::from_value(fixture["specialized_tools"].clone())?;
    let trailing_items: Vec<TurnInputItem> =
        serde_json::from_value(fixture["trailing_items"].clone())?;
    let metadata: ResponseMetadata = serde_json::from_value(fixture["metadata"].clone())?;
    let terminals: Vec<ResponseTerminal> = serde_json::from_value(fixture["terminals"].clone())?;

    assert!(!blocks.iter().any(block_uses_escape_hatch));
    let request = LlmRequest {
        messages: vec![Message {
            role: Role::User,
            content: blocks.clone(),
        }],
        trailing_items,
        specialized_tools,
        ..LlmRequest::default()
    };
    assert!(request.extensions.fields.is_empty());
    assert!(request.preservation.requests.is_empty());
    assert!(request.reasoning.raw.is_none());
    let request: LlmRequest = serde_json::from_value(serde_json::to_value(request)?)?;
    assert!(request.extensions.fields.is_empty());

    let response = AggLlmResponse {
        outputs: vec![ResponseOutput {
            role: Role::Assistant,
            content: blocks,
            stop_reason: None,
        }],
        metadata,
        terminal: terminals.into_iter().next(),
        ..AggLlmResponse::default()
    };
    assert!(response.extensions.fields.is_empty());
    assert!(response.preservation.responses.is_empty());
    let response: AggLlmResponse = serde_json::from_value(serde_json::to_value(response)?)?;
    assert!(response.extensions.fields.is_empty());
    assert!(
        !response.outputs[0]
            .content
            .iter()
            .any(block_uses_escape_hatch)
    );
    Ok(())
}
