// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content-free descriptions of judge replies for operator logs.
//!
//! A judge reply can quote the conversation back, so nothing here keeps its text. Every
//! value is a count, a flag, a static tag, or a field name taken from the configured
//! response schema. Keys the model invented are counted, never named.

use serde_json::Value;
use switchyard_protocol::{AggLlmResponse, ContentBlock, StopReason, completion_text};

/// Most schema field names reported in one list.
const MAX_REPORTED_FIELDS: usize = 16;

/// The shape of one judge reply: how it ended and how much of it was visible.
#[derive(Debug, PartialEq)]
pub(crate) struct ReplyShape {
    pub(crate) stop_reason: &'static str,
    pub(crate) output_limit_reached: bool,
    pub(crate) input_tokens: Option<u64>,
    pub(crate) output_tokens: Option<u64>,
    pub(crate) reasoning_tokens: Option<u64>,
    pub(crate) visible_chars: usize,
    pub(crate) reasoning_chars: usize,
    pub(crate) fenced: bool,
}

impl ReplyShape {
    pub(crate) fn of(response: &AggLlmResponse) -> Self {
        let stop_reason = response
            .first_output()
            .and_then(|output| output.stop_reason);
        let text = completion_text(response);
        let reasoning_chars = response
            .first_output()
            .map(|output| {
                output
                    .content
                    .iter()
                    .map(|block| match block {
                        ContentBlock::Reasoning { text, .. } => text.chars().count(),
                        _ => 0,
                    })
                    .sum()
            })
            .unwrap_or(0);
        Self {
            stop_reason: stop_reason_tag(stop_reason),
            output_limit_reached: stop_reason == Some(StopReason::MaxTokens),
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
            reasoning_tokens: response.usage.reasoning_tokens,
            visible_chars: text.chars().count(),
            reasoning_chars,
            fenced: text.trim_start().starts_with("```"),
        }
    }
}

fn stop_reason_tag(stop_reason: Option<StopReason>) -> &'static str {
    match stop_reason {
        None => "unreported",
        Some(StopReason::EndTurn) => "end_turn",
        Some(StopReason::MaxTokens) => "max_tokens",
        Some(StopReason::ToolUse) => "tool_use",
        Some(StopReason::ContentFilter) => "content_filter",
        Some(StopReason::Error) => "error",
        Some(StopReason::Unknown) => "unknown",
    }
}

/// Why a judge reply could not become a verdict.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct ParseDiagnosis {
    /// One of `empty_completion`, `whitespace_completion`, `truncated_json`, `invalid_json`,
    /// `not_object`, or `schema_mismatch`.
    pub(crate) reason_code: &'static str,
    /// serde_json's error class for replies that are not JSON.
    pub(crate) json_category: Option<&'static str>,
    pub(crate) json_line: Option<usize>,
    pub(crate) json_column: Option<usize>,
    pub(crate) top_level_keys: Option<usize>,
    /// Required schema fields the reply left out.
    pub(crate) missing_fields: String,
    /// Schema fields whose JSON type differs from the schema's declared type.
    pub(crate) wrong_type_fields: String,
    /// Keys the schema does not declare. Counted, never named.
    pub(crate) unknown_key_count: usize,
}

/// Explains a reply the judge's decoder rejected. `schema` is the configured inner JSON
/// Schema; with no `properties` or `required` only the JSON-level checks apply.
pub(crate) fn diagnose_parse(reply: &str, schema: Option<&Value>) -> ParseDiagnosis {
    if reply.is_empty() {
        return ParseDiagnosis {
            reason_code: "empty_completion",
            ..ParseDiagnosis::default()
        };
    }
    // Constrained JSON decoding can spend the whole budget on whitespace.
    let body = super::llm_judge::strip_json_fence(reply.trim());
    if body.trim().is_empty() {
        return ParseDiagnosis {
            reason_code: "whitespace_completion",
            ..ParseDiagnosis::default()
        };
    }
    let value = match serde_json::from_str::<Value>(body) {
        Ok(value) => value,
        Err(error) => {
            let (reason_code, json_category) = match error.classify() {
                serde_json::error::Category::Eof => ("truncated_json", "eof"),
                serde_json::error::Category::Syntax => ("invalid_json", "syntax"),
                serde_json::error::Category::Data => ("invalid_json", "data"),
                serde_json::error::Category::Io => ("invalid_json", "io"),
            };
            return ParseDiagnosis {
                reason_code,
                json_category: Some(json_category),
                json_line: Some(error.line()),
                json_column: Some(error.column()),
                ..ParseDiagnosis::default()
            };
        }
    };
    let Value::Object(object) = value else {
        return ParseDiagnosis {
            reason_code: "not_object",
            ..ParseDiagnosis::default()
        };
    };
    let properties = schema
        .and_then(|schema| schema.get("properties"))
        .and_then(Value::as_object);
    let missing_fields = schema
        .and_then(|schema| schema.get("required"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|field| !object.contains_key(*field));
    let wrong_type_fields = properties
        .into_iter()
        .flatten()
        .filter_map(|(field, rule)| {
            let value = object.get(field)?;
            let expected = rule.get("type")?;
            (!type_matches(value, expected)).then_some(field.as_str())
        });
    let unknown_key_count = properties.map_or(0, |properties| {
        object
            .keys()
            .filter(|key| !properties.contains_key(*key))
            .count()
    });
    ParseDiagnosis {
        reason_code: "schema_mismatch",
        top_level_keys: Some(object.len()),
        missing_fields: field_list(missing_fields),
        wrong_type_fields: field_list(wrong_type_fields),
        unknown_key_count,
        ..ParseDiagnosis::default()
    }
}

fn field_list<'a>(fields: impl Iterator<Item = &'a str>) -> String {
    fields
        .take(MAX_REPORTED_FIELDS)
        .collect::<Vec<_>>()
        .join(",")
}

fn type_matches(value: &Value, expected: &Value) -> bool {
    match expected {
        Value::String(name) => is_json_type(value, name),
        Value::Array(names) => names
            .iter()
            .filter_map(Value::as_str)
            .any(|name| is_json_type(value, name)),
        _ => true,
    }
}

fn is_json_type(value: &Value, name: &str) -> bool {
    match name {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "null" => value.is_null(),
        _ => true,
    }
}

/// Collects formatted log lines from the current thread for assertions.
#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct LogCapture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

#[cfg(test)]
impl LogCapture {
    /// Routes this thread's INFO-and-above events into the capture until the guard drops.
    ///
    /// A callsite another test thread is registering at the same moment can be cached as
    /// uninteresting to this subscriber. Callers run their scenario once, then [`Self::rearm`],
    /// then run it again and assert on that second run.
    pub(crate) fn install(&self) -> tracing::subscriber::DefaultGuard {
        tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_writer(self.clone())
                .with_ansi(false)
                .without_time()
                .with_max_level(tracing::Level::INFO)
                .finish(),
        )
    }

    /// Recomputes every registered callsite's interest and discards what was captured so far.
    pub(crate) fn rearm(&self) {
        tracing::callsite::rebuild_interest_cache();
        self.0.lock().expect("log capture lock").clear();
    }

    pub(crate) fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().expect("log capture lock")).into_owned()
    }
}

#[cfg(test)]
impl std::io::Write for LogCapture {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .expect("log capture lock")
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogCapture {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use switchyard_protocol::{Usage, text_response};

    fn schema() -> Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["crux", "primary_rule", "capability_boundary", "p_solve"],
            "properties": {
                "crux": {"type": "string"},
                "primary_rule": {"type": "string"},
                "capability_boundary": {"type": "string"},
                "p_solve": {"type": "number"}
            }
        })
    }

    #[test]
    fn a_reply_is_sorted_into_one_bounded_reason() {
        let schema = schema();
        let cases = [
            ("", "empty_completion"),
            ("\n\n  \n", "whitespace_completion"),
            ("```json\n```", "whitespace_completion"),
            (
                r#"{"crux":"needs a parser","primary_rule":"SU"#,
                "truncated_json",
            ),
            ("Sure! Here is the routing decision", "invalid_json"),
            (r#"{"crux":"x"} trailing"#, "invalid_json"),
            ("[1,2]", "not_object"),
            (r#"{"crux":"x"}"#, "schema_mismatch"),
        ];
        for (reply, expected) in cases {
            assert_eq!(
                diagnose_parse(reply, Some(&schema)).reason_code,
                expected,
                "{reply:?}"
            );
        }
    }

    #[test]
    fn a_truncated_reply_reports_where_the_json_stopped() {
        let diagnosis = diagnose_parse("{\n  \"crux\": \"needs a par", Some(&schema()));
        assert_eq!(diagnosis.json_category, Some("eof"));
        assert_eq!(diagnosis.json_line, Some(2));
        assert_eq!(diagnosis.json_column, Some(22));
    }

    #[test]
    fn schema_flags_name_only_schema_fields_and_count_invented_keys() {
        let diagnosis = diagnose_parse(
            r#"{"crux":"x","p_solve":"high","secret_user_field":"leak me"}"#,
            Some(&schema()),
        );
        assert_eq!(
            diagnosis,
            ParseDiagnosis {
                reason_code: "schema_mismatch",
                top_level_keys: Some(3),
                missing_fields: "primary_rule,capability_boundary".to_string(),
                wrong_type_fields: "p_solve".to_string(),
                unknown_key_count: 1,
                ..ParseDiagnosis::default()
            }
        );
        let rendered = format!("{diagnosis:?}");
        assert!(!rendered.contains("secret_user_field"));
        assert!(!rendered.contains("leak me"));
        assert!(!rendered.contains("high"));
    }

    #[test]
    fn the_shape_counts_visible_and_reasoning_output_and_flags_the_limit() {
        let mut response = text_response(None, r#"{"crux":"#);
        response.usage = Usage {
            input_tokens: Some(959),
            output_tokens: Some(128),
            ..Usage::default()
        };
        let output = &mut response.outputs[0];
        output.stop_reason = Some(StopReason::MaxTokens);
        output.content.insert(
            0,
            ContentBlock::Reasoning {
                text: "think".to_string(),
                signature: None,
                details: Vec::new(),
                openai_chat_field: Default::default(),
            },
        );
        assert_eq!(
            ReplyShape::of(&response),
            ReplyShape {
                stop_reason: "max_tokens",
                output_limit_reached: true,
                input_tokens: Some(959),
                output_tokens: Some(128),
                reasoning_tokens: None,
                visible_chars: 8,
                reasoning_chars: 5,
                fenced: false,
            }
        );
    }
}
