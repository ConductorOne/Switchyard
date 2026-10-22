// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered codecs for decision APIs, which answer typed questions about shared input.
//!
//! Both formats map to [`LlmRequest::decision`] and
//! [`AggLlmResponse::decisions`](crate::llm::AggLlmResponse::decisions), so they translate to
//! each other. Neither translates to a conversation format, because a chat model
//! returns text, not typed answers.

mod openai;
mod typesafe;

pub use openai::OpenAiDecisionsCodec;
pub use typesafe::TypesafeSystemOneCodec;

use serde_json::{Map, Value, json};

use crate::error::{Result, TranslationError};
use crate::llm::{LlmRequest, Usage};

fn invalid(path: &str, message: impl Into<String>) -> TranslationError {
    TranslationError::InvalidValue {
        path: path.to_string(),
        message: message.into(),
    }
}

fn required_str<'a>(object: &'a Map<String, Value>, key: &str, path: &str) -> Result<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| TranslationError::InvalidType {
            path: format!("{path}.{key}"),
            expected: "string",
        })
}

fn optional_str(object: &Map<String, Value>, key: &str) -> Option<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn required_array<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<&'a Vec<Value>> {
    object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| TranslationError::InvalidType {
            path: format!("{path}.{key}"),
            expected: "array",
        })
}

fn required_f64(object: &Map<String, Value>, key: &str, path: &str) -> Result<f64> {
    object
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| TranslationError::InvalidType {
            path: format!("{path}.{key}"),
            expected: "number",
        })
}

// Wire token counts are not named the same across providers, so each codec reads its own.
fn decode_usage(usage: Option<&Value>) -> Usage {
    let field = |key: &str| {
        usage
            .and_then(|usage| usage.get(key))
            .and_then(Value::as_u64)
    };
    Usage {
        input_tokens: field("input_tokens"),
        output_tokens: field("output_tokens"),
        total_tokens: field("total_tokens"),
        ..Usage::default()
    }
}

fn encode_usage(usage: &Usage, with_total: bool) -> Option<Value> {
    if usage.input_tokens.is_none() && usage.output_tokens.is_none() {
        return None;
    }
    let mut body = json!({
        "input_tokens": usage.input_tokens.unwrap_or(0),
        "output_tokens": usage.output_tokens.unwrap_or(0),
    });
    if with_total {
        let total = usage
            .total_tokens
            .unwrap_or(usage.input_tokens.unwrap_or(0) + usage.output_tokens.unwrap_or(0));
        body["total_tokens"] = json!(total);
    }
    Some(body)
}

// Copies only fields the target wire documents. Another format's fields could be rejected upstream.
fn copy_extensions(
    body: &mut Map<String, Value>,
    extensions: &Map<String, Value>,
    allowed: &[&str],
) {
    for key in allowed {
        if let Some(value) = extensions.get(*key) {
            body.entry((*key).to_string())
                .or_insert_with(|| value.clone());
        }
    }
}

// The System One wire keys answers and questions by name, so a repeated key would drop one.
fn insert_unique(
    map: &mut Map<String, Value>,
    key: String,
    value: Value,
    path: &str,
) -> Result<()> {
    if map.contains_key(&key) {
        return Err(invalid(path, format!("duplicate name {key:?}")));
    }
    map.insert(key, value);
    Ok(())
}

// A decision request must carry questions, or its codec produced the wrong IR.
fn require_decision(request: &LlmRequest, format: &str) -> Result<()> {
    if request.decision.is_none() {
        return Err(TranslationError::Other(format!(
            "{format} requests need decision questions; this request has none"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
