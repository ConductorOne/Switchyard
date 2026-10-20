// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered codec for the OpenAI-compatible Decisions API (`POST /v1/decisions`).

use serde_json::{Map, Value, json};

use super::{
    copy_extensions, decode_usage, encode_usage, invalid, optional_str, require_decision,
    required_array, required_f64, required_str,
};
use crate::codecs::common::provider_extensions;
use crate::codecs::openai_media::image_payload;
use crate::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use crate::decision::{
    DecisionAnswer, DecisionChoice, DecisionKind, DecisionLevel, DecisionProbability,
    DecisionQuestion, DecisionRequest, NamedDecisionAnswer,
};
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::llm::{
    AggLlmResponse, ContentBlock, ImageSource, LlmRequest, Message, ProviderExtensions, Role,
};
use crate::policy::TranslationPolicy;
use crate::util::{
    capture_request_preservation, capture_response_preservation, embed_preservation,
    exact_preserved_request, exact_preserved_response, object, push_lossy,
};

/// Format codec for OpenAI-compatible Decisions API payloads.
pub struct OpenAiDecisionsCodec;

const REQUEST_FIELDS: &[&str] = &["model", "input", "questions"];
const RESPONSE_FIELDS: &[&str] = &["id", "model", "answers", "usage"];

impl FormatCodec for OpenAiDecisionsCodec {
    fn format(&self) -> FormatId {
        WireFormat::OpenAiDecisions.into()
    }

    fn decode_request(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedRequest> {
        let object = object(body, "$")?;
        let questions = required_array(object, "questions", "$")?
            .iter()
            .enumerate()
            .map(|(index, question)| decode_question(question, &format!("$.questions[{index}]")))
            .collect::<Result<Vec<_>>>()?;
        let request = LlmRequest {
            model: optional_str(object, "model").filter(|model| !model.is_empty()),
            messages: decode_input(object.get("input"))?,
            decision: Some(DecisionRequest { questions }),
            extensions: ProviderExtensions {
                fields: provider_extensions(object, REQUEST_FIELDS),
            },
            preservation: capture_request_preservation(WireFormat::OpenAiDecisions, body, policy),
            ..LlmRequest::default()
        };
        Ok(DecodedRequest {
            request,
            diagnostics: Vec::new(),
        })
    }

    fn encode_request(
        &self,
        request: &LlmRequest,
        policy: &TranslationPolicy,
    ) -> Result<EncodedRequest> {
        if let Some(body) =
            exact_preserved_request(&request.preservation, WireFormat::OpenAiDecisions, policy)
        {
            return Ok(EncodedRequest {
                body,
                diagnostics: Vec::new(),
            });
        }
        require_decision(request, "openai_decisions")?;
        let mut diagnostics = Vec::new();
        for question in request
            .decision
            .iter()
            .flat_map(|decision| &decision.questions)
        {
            if let DecisionKind::Boolean {
                true_case: Some(_), ..
            }
            | DecisionKind::Boolean {
                false_case: Some(_),
                ..
            } = &question.kind
            {
                push_lossy(
                    &mut diagnostics,
                    policy,
                    "OpenAI predicates have no criteria; the true/false definitions were dropped",
                )?;
            }
        }
        let decision = request
            .decision
            .as_ref()
            .into_iter()
            .flat_map(|decision| &decision.questions);
        let mut body = Map::new();
        if let Some(model) = &request.model {
            body.insert("model".to_string(), json!(model));
        }
        body.insert("input".to_string(), encode_input(request)?);
        body.insert(
            "questions".to_string(),
            Value::Array(decision.map(encode_question).collect()),
        );
        copy_extensions(
            &mut body,
            &request.extensions.fields,
            &["providerOptions", "safety_identifier"],
        );
        Ok(EncodedRequest {
            body: embed_preservation(Value::Object(body), &request.preservation, policy),
            diagnostics,
        })
    }

    fn decode_response(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedResponse> {
        let object = object(body, "$")?;
        let answers = required_array(object, "answers", "$")?
            .iter()
            .enumerate()
            .map(|(index, answer)| decode_answer(answer, &format!("$.answers[{index}]")))
            .collect::<Result<Vec<_>>>()?;
        let response = AggLlmResponse {
            id: optional_str(object, "id"),
            model: optional_str(object, "model"),
            decisions: Some(answers),
            usage: decode_usage(object.get("usage")),
            extensions: ProviderExtensions {
                fields: provider_extensions(object, RESPONSE_FIELDS),
            },
            preservation: capture_response_preservation(WireFormat::OpenAiDecisions, body, policy),
            ..AggLlmResponse::default()
        };
        Ok(DecodedResponse {
            response,
            diagnostics: Vec::new(),
        })
    }

    fn encode_response(
        &self,
        response: &AggLlmResponse,
        policy: &TranslationPolicy,
    ) -> Result<EncodedResponse> {
        if let Some(body) =
            exact_preserved_response(&response.preservation, WireFormat::OpenAiDecisions, policy)
        {
            return Ok(EncodedResponse {
                body,
                diagnostics: Vec::new(),
            });
        }
        let answers = response.decisions.as_ref().ok_or_else(|| {
            TranslationError::Other("openai_decisions responses need decision answers".to_string())
        })?;
        let mut body = Map::new();
        if let Some(id) = &response.id {
            body.insert("id".to_string(), json!(id));
        }
        if let Some(model) = &response.model {
            body.insert("model".to_string(), json!(model));
        }
        body.insert(
            "answers".to_string(),
            Value::Array(answers.iter().map(encode_answer).collect()),
        );
        if let Some(usage) = encode_usage(&response.usage, true) {
            body.insert("usage".to_string(), usage);
        }
        copy_extensions(
            &mut body,
            &response.extensions.fields,
            &["provider_metadata"],
        );
        Ok(EncodedResponse {
            body: embed_preservation(Value::Object(body), &response.preservation, policy),
            diagnostics: Vec::new(),
        })
    }
}

fn decode_question(question: &Value, path: &str) -> Result<DecisionQuestion> {
    let question = object(question, path)?;
    let kind = match required_str(question, "type", path)? {
        "predicate" => DecisionKind::Boolean {
            true_case: None,
            false_case: None,
        },
        "choice" => DecisionKind::Choice {
            choices: required_array(question, "choices", path)?
                .iter()
                .enumerate()
                .map(|(index, choice)| {
                    let path = format!("{path}.choices[{index}]");
                    let choice = object(choice, &path)?;
                    let value = choice
                        .get("value")
                        .filter(|value| value.is_string() || value.is_boolean())
                        .cloned()
                        .ok_or_else(|| {
                            invalid(&path, "choice value must be a string or boolean")
                        })?;
                    Ok(DecisionChoice {
                        value,
                        description: optional_str(choice, "description"),
                    })
                })
                .collect::<Result<_>>()?,
        },
        "score" => DecisionKind::Score {
            levels: required_array(question, "levels", path)?
                .iter()
                .enumerate()
                .map(|(index, level)| {
                    let path = format!("{path}.levels[{index}]");
                    let level = object(level, &path)?;
                    Ok(DecisionLevel {
                        label: required_str(level, "label", &path)?.to_string(),
                        description: optional_str(level, "description"),
                    })
                })
                .collect::<Result<_>>()?,
        },
        other => {
            return Err(invalid(
                &format!("{path}.type"),
                format!("expected one of predicate, choice, score; got {other:?}"),
            ));
        }
    };
    Ok(DecisionQuestion {
        name: optional_str(question, "name"),
        instructions: required_str(question, "instructions", path)?.to_string(),
        kind,
    })
}

fn encode_question(question: &DecisionQuestion) -> Value {
    let mut body = Map::new();
    body.insert(
        "type".to_string(),
        json!(match question.kind {
            DecisionKind::Boolean { .. } => "predicate",
            DecisionKind::Choice { .. } => "choice",
            DecisionKind::Score { .. } => "score",
        }),
    );
    if let Some(name) = &question.name {
        body.insert("name".to_string(), json!(name));
    }
    body.insert("instructions".to_string(), json!(question.instructions));
    match &question.kind {
        DecisionKind::Boolean { .. } => {}
        DecisionKind::Choice { choices } => {
            let choices = choices.iter().map(|choice| {
                let mut item = json!({"value": choice.value});
                if let Some(description) = &choice.description {
                    item["description"] = json!(description);
                }
                item
            });
            body.insert("choices".to_string(), Value::Array(choices.collect()));
        }
        DecisionKind::Score { levels } => {
            let levels = levels.iter().map(|level| {
                let mut item = json!({"label": level.label});
                if let Some(description) = &level.description {
                    item["description"] = json!(description);
                }
                item
            });
            body.insert("levels".to_string(), Value::Array(levels.collect()));
        }
    }
    Value::Object(body)
}

// `input` is a string, or an array of user messages whose content is a string or ordered parts.
fn decode_input(input: Option<&Value>) -> Result<Vec<Message>> {
    match input {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(text)) => Ok(vec![Message::text(Role::User, text.clone())]),
        Some(Value::Array(messages)) => messages
            .iter()
            .enumerate()
            .map(|(index, message)| {
                let path = format!("$.input[{index}]");
                let message = object(message, &path)?;
                let content = match message.get("content") {
                    Some(Value::String(text)) => vec![ContentBlock::Text { text: text.clone() }],
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .enumerate()
                        .map(|(part, value)| decode_part(value, &format!("{path}.content[{part}]")))
                        .collect::<Result<_>>()?,
                    _ => {
                        return Err(TranslationError::InvalidType {
                            path: format!("{path}.content"),
                            expected: "string or array",
                        });
                    }
                };
                Ok(Message {
                    role: Role::User,
                    content,
                })
            })
            .collect(),
        Some(_) => Err(TranslationError::InvalidType {
            path: "$.input".to_string(),
            expected: "string or array",
        }),
    }
}

fn decode_part(part: &Value, path: &str) -> Result<ContentBlock> {
    let part = object(part, path)?;
    match required_str(part, "type", path)? {
        "input_text" => Ok(ContentBlock::Text {
            text: required_str(part, "text", path)?.to_string(),
        }),
        "input_image" => Ok(ContentBlock::Image {
            source: ImageSource::Url {
                url: required_str(part, "image_url", path)?.to_string(),
                detail: optional_str(part, "detail"),
            },
        }),
        other => Err(invalid(
            &format!("{path}.type"),
            format!("expected input_text or input_image; got {other:?}"),
        )),
    }
}

fn encode_input(request: &LlmRequest) -> Result<Value> {
    if let [message] = request.messages.as_slice()
        && message.role == Role::User
        && let [ContentBlock::Text { text }] = message.content.as_slice()
    {
        return Ok(json!(text));
    }
    let messages = request
        .messages
        .iter()
        .map(|message| {
            let parts = message
                .content
                .iter()
                .map(|block| match block {
                    ContentBlock::Text { text } => Ok(json!({"type": "input_text", "text": text})),
                    ContentBlock::Image { source } => {
                        let payload = image_payload(source).ok_or_else(|| {
                            TranslationError::Other(
                                "decision images need a URL or a typed base64 payload".to_string(),
                            )
                        })?;
                        let mut part = json!({"type": "input_image", "image_url": payload.url});
                        if let Some(detail) = payload.detail {
                            part["detail"] = json!(detail);
                        }
                        Ok(part)
                    }
                    _ => Err(TranslationError::Other(
                        "decision input supports only text and images".to_string(),
                    )),
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(json!({"role": "user", "content": parts}))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Value::Array(messages))
}

fn decode_probabilities(
    answer: &Map<String, Value>,
    path: &str,
) -> Result<Vec<DecisionProbability>> {
    let Some(items) = answer.get("probabilities") else {
        return Ok(Vec::new());
    };
    let items = items
        .as_array()
        .ok_or_else(|| TranslationError::InvalidType {
            path: format!("{path}.probabilities"),
            expected: "array",
        })?;
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let path = format!("{path}.probabilities[{index}]");
            let item = object(item, &path)?;
            Ok(DecisionProbability {
                value: item
                    .get("value")
                    .cloned()
                    .ok_or_else(|| invalid(&path, "missing value"))?,
                label: optional_str(item, "label"),
                probability: required_f64(item, "probability", &path)?,
            })
        })
        .collect()
}

fn decode_answer(answer: &Value, path: &str) -> Result<NamedDecisionAnswer> {
    let answer = object(answer, path)?;
    let name = optional_str(answer, "name");
    let confidence = answer.get("confidence").and_then(Value::as_f64);
    let decoded = match required_str(answer, "type", path)? {
        "predicate" => DecisionAnswer::Boolean {
            probability: required_f64(answer, "probability", path)?,
        },
        "choice" => DecisionAnswer::Choice {
            choice: answer
                .get("choice")
                .cloned()
                .ok_or_else(|| invalid(path, "missing choice"))?,
            confidence,
            probabilities: decode_probabilities(answer, path)?,
        },
        "score" => DecisionAnswer::Score {
            score: required_f64(answer, "score", path)?,
            confidence,
            probabilities: decode_probabilities(answer, path)?,
        },
        "refusal" => DecisionAnswer::Refusal,
        other => {
            return Err(invalid(
                &format!("{path}.type"),
                format!("expected one of predicate, choice, score, refusal; got {other:?}"),
            ));
        }
    };
    Ok(NamedDecisionAnswer {
        name,
        answer: decoded,
    })
}

fn encode_probabilities(probabilities: &[DecisionProbability]) -> Value {
    Value::Array(
        probabilities
            .iter()
            .map(|item| {
                let mut body = json!({"value": item.value});
                if let Some(label) = &item.label {
                    body["label"] = json!(label);
                }
                body["probability"] = json!(item.probability);
                body
            })
            .collect(),
    )
}

fn encode_answer(answer: &NamedDecisionAnswer) -> Value {
    let mut body = Map::new();
    let kind = match &answer.answer {
        DecisionAnswer::Boolean { .. } => "predicate",
        DecisionAnswer::Choice { .. } => "choice",
        DecisionAnswer::Score { .. } => "score",
        DecisionAnswer::Refusal => "refusal",
    };
    body.insert("type".to_string(), json!(kind));
    // An unnamed question comes back as `name: null`.
    body.insert("name".to_string(), json!(answer.name));
    match &answer.answer {
        DecisionAnswer::Boolean { probability } => {
            body.insert("probability".to_string(), json!(probability));
        }
        DecisionAnswer::Choice {
            choice,
            confidence,
            probabilities,
        } => {
            body.insert("choice".to_string(), choice.clone());
            if let Some(confidence) = confidence {
                body.insert("confidence".to_string(), json!(confidence));
            }
            if !probabilities.is_empty() {
                body.insert(
                    "probabilities".to_string(),
                    encode_probabilities(probabilities),
                );
            }
        }
        DecisionAnswer::Score {
            score,
            confidence,
            probabilities,
        } => {
            body.insert("score".to_string(), json!(score));
            if let Some(confidence) = confidence {
                body.insert("confidence".to_string(), json!(confidence));
            }
            if !probabilities.is_empty() {
                body.insert(
                    "probabilities".to_string(),
                    encode_probabilities(probabilities),
                );
            }
        }
        DecisionAnswer::Refusal => {}
    }
    Value::Object(body)
}
