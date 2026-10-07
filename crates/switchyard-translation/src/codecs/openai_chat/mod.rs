// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Chat Completions buffered and streaming codecs.

mod buffered;
mod stream;

pub use buffered::OpenAiChatCodec;
pub use stream::OpenAiChatStreamCodec;

pub(crate) use stream::prepare_google_tool_replay;

pub(crate) use buffered::{decode_file_source, decode_image_source};

// Google issues a complete opaque signature on the originating call, not a reasoning block.
fn decode_google_thought_signature(
    tool_call: &serde_json::Map<String, serde_json::Value>,
) -> crate::error::Result<Option<&str>> {
    let Some(signature) = tool_call
        .get("extra_content")
        .and_then(|extra| extra.get("google"))
        .and_then(|google| google.get("thought_signature"))
    else {
        return Ok(None);
    };
    signature
        .as_str()
        .map(Some)
        .ok_or_else(|| crate::error::TranslationError::InvalidType {
            path: "tool_calls[].extra_content.google.thought_signature".to_string(),
            expected: "string",
        })
}

fn encode_google_thought_signature(tool_call: &mut serde_json::Value, signature: Option<&str>) {
    if let Some(signature) = signature {
        tool_call["extra_content"] =
            serde_json::json!({"google": {"thought_signature": signature}});
    }
}
