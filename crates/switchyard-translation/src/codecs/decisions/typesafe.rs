// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered codec for the TypeSafe System One API (`POST /v1/systemone`).
//!
//! Requests carry the input as `state` and the questions as a map keyed by name. A boolean
//! question is called `noul`. Choice criteria are a map of value to description, and score
//! criteria are an ordered list of labels.

use serde_json::{Map, Value, json};

use super::{
    copy_extensions, decode_usage, encode_usage, insert_unique, invalid, optional_str,
    require_decision, required_f64, required_str,
};
use crate::codecs::common::provider_extensions;
use crate::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use crate::decision::{
    DecisionAnswer, DecisionChoice, DecisionKind, DecisionLevel, DecisionProbability,
    DecisionQuestion, DecisionRequest, NamedDecisionAnswer,
};
use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::llm::{AggLlmResponse, ContentBlock, LlmRequest, Message, ProviderExtensions, Role};
use crate::policy::TranslationPolicy;
use crate::util::{
    capture_request_preservation, capture_response_preservation, embed_preservation,
    exact_preserved_request, exact_preserved_response, object, push_lossy,
};

/// Format codec for TypeSafe System One payloads.
pub struct TypesafeSystemOneCodec;

const REQUEST_FIELDS: &[&str] = &["model", "state", "questions"];
const RESPONSE_FIELDS: &[&str] = &["model", "answers", "usage"];

impl FormatCodec for TypesafeSystemOneCodec {
    fn format(&self) -> FormatId {
        WireFormat::TypesafeSystemOne.into()
    }

    fn decode_request(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedRequest> {
        let object = object(body, "$")?;
        let questions = object
            .get("questions")
            .and_then(Value::as_object)
            .ok_or_else(|| TranslationError::InvalidType {
                path: "$.questions".to_string(),
                expected: "object",
            })?
            .iter()
            .map(|(name, question)| decode_question(name, question))
            .collect::<Result<Vec<_>>>()?;
        let messages = match object.get("state") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::String(text)) => vec![Message::text(Role::User, text.clone())],
            // Structured state is shared as its JSON text, which is also how a model reads it.
            Some(state) => vec![Message::text(Role::User, state.to_string())],
        };
        let request = LlmRequest {
            model: optional_str(object, "model").filter(|model| !model.is_empty()),
            messages,
            decision: Some(DecisionRequest { questions }),
            extensions: ProviderExtensions {
                fields: provider_extensions(object, REQUEST_FIELDS),
            },
            preservation: capture_request_preservation(WireFormat::TypesafeSystemOne, body, policy),
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
            exact_preserved_request(&request.preservation, WireFormat::TypesafeSystemOne, policy)
        {
            return Ok(EncodedRequest {
                body,
                diagnostics: Vec::new(),
            });
        }
        require_decision(request, "typesafe_systemone")?;
        let mut diagnostics = Vec::new();
        warn_lossy_request(request, policy, &mut diagnostics)?;
        let mut body = Map::new();
        if let Some(model) = &request.model {
            body.insert("model".to_string(), json!(model));
        }
        body.insert("state".to_string(), json!(encode_state(request)?));
        let mut questions = Map::new();
        for (index, question) in request
            .decision
            .iter()
            .flat_map(|decision| &decision.questions)
            .enumerate()
        {
            // The wire map needs a key, so an unnamed question is keyed by its position.
            let name = question.name.clone().unwrap_or_else(|| index.to_string());
            insert_unique(
                &mut questions,
                name,
                encode_question(question),
                "$.questions",
            )?;
        }
        body.insert("questions".to_string(), Value::Object(questions));
        copy_extensions(&mut body, &request.extensions.fields, &["providerOptions"]);
        Ok(EncodedRequest {
            body: embed_preservation(Value::Object(body), &request.preservation, policy),
            diagnostics,
        })
    }

    fn decode_response(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedResponse> {
        let object = object(body, "$")?;
        let answers = object
            .get("answers")
            .and_then(Value::as_object)
            .ok_or_else(|| TranslationError::InvalidType {
                path: "$.answers".to_string(),
                expected: "object",
            })?
            .iter()
            .map(|(name, answer)| decode_answer(name, answer))
            .collect::<Result<Vec<_>>>()?;
        let response = AggLlmResponse {
            model: optional_str(object, "model"),
            decisions: Some(answers),
            usage: decode_usage(object.get("usage")),
            extensions: ProviderExtensions {
                fields: provider_extensions(object, RESPONSE_FIELDS),
            },
            preservation: capture_response_preservation(
                WireFormat::TypesafeSystemOne,
                body,
                policy,
            ),
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
        if let Some(body) = exact_preserved_response(
            &response.preservation,
            WireFormat::TypesafeSystemOne,
            policy,
        ) {
            return Ok(EncodedResponse {
                body,
                diagnostics: Vec::new(),
            });
        }
        let answers = response.decisions.as_ref().ok_or_else(|| {
            TranslationError::Other(
                "typesafe_systemone responses need decision answers".to_string(),
            )
        })?;
        let mut body = Map::new();
        if let Some(model) = &response.model {
            body.insert("model".to_string(), json!(model));
        }
        let mut named = Map::new();
        for (index, answer) in answers.iter().enumerate() {
            let name = answer.name.clone().unwrap_or_else(|| index.to_string());
            insert_unique(&mut named, name, encode_answer(&answer.answer), "$.answers")?;
        }
        body.insert("answers".to_string(), Value::Object(named));
        if let Some(usage) = encode_usage(&response.usage, false) {
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

// Reports what the System One wire cannot express, so a strict policy can refuse the request.
fn warn_lossy_request(
    request: &LlmRequest,
    policy: &TranslationPolicy,
    diagnostics: &mut Vec<TranslationDiagnostic>,
) -> Result<()> {
    if request.messages.len() > 1 {
        push_lossy(
            diagnostics,
            policy,
            "System One takes one `state`; the input messages were joined with newlines",
        )?;
    }
    for question in request
        .decision
        .iter()
        .flat_map(|decision| &decision.questions)
    {
        match &question.kind {
            DecisionKind::Choice { choices }
                if choices.iter().any(|choice| !choice.value.is_string()) =>
            {
                push_lossy(
                    diagnostics,
                    policy,
                    "System One choice values are object keys; non-string values became strings",
                )?;
            }
            DecisionKind::Score { levels }
                if levels.iter().any(|level| level.description.is_some()) =>
            {
                push_lossy(
                    diagnostics,
                    policy,
                    "System One levels are single strings; descriptions were joined to their labels",
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn decode_question(name: &str, question: &Value) -> Result<DecisionQuestion> {
    let path = format!("$.questions.{name}");
    let question = object(question, &path)?;
    let criteria = question.get("criteria");
    let kind = match required_str(question, "type", &path)? {
        "noul" => DecisionKind::Boolean {
            true_case: criteria
                .and_then(|criteria| criteria.get("true"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            false_case: criteria
                .and_then(|criteria| criteria.get("false"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
        },
        "choice" => DecisionKind::Choice {
            choices: criteria
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    invalid(
                        &format!("{path}.criteria"),
                        "choice needs a criteria object",
                    )
                })?
                .iter()
                .map(|(value, description)| DecisionChoice {
                    value: json!(value),
                    description: description.as_str().map(ToOwned::to_owned),
                })
                .collect(),
        },
        "score" => DecisionKind::Score {
            levels: criteria
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    invalid(&format!("{path}.criteria"), "score needs a criteria array")
                })?
                .iter()
                .map(|label| {
                    label
                        .as_str()
                        .map(|label| DecisionLevel {
                            label: label.to_string(),
                            description: None,
                        })
                        .ok_or_else(|| {
                            invalid(&format!("{path}.criteria"), "levels must be strings")
                        })
                })
                .collect::<Result<_>>()?,
        },
        other => {
            return Err(invalid(
                &format!("{path}.type"),
                format!("expected one of 'noul', 'choice', 'score'; got {other:?}"),
            ));
        }
    };
    Ok(DecisionQuestion {
        name: Some(name.to_string()),
        instructions: required_str(question, "instructions", &path)?.to_string(),
        kind,
    })
}

fn encode_question(question: &DecisionQuestion) -> Value {
    let mut body = Map::new();
    body.insert(
        "type".to_string(),
        json!(match question.kind {
            DecisionKind::Boolean { .. } => "noul",
            DecisionKind::Choice { .. } => "choice",
            DecisionKind::Score { .. } => "score",
        }),
    );
    body.insert("instructions".to_string(), json!(question.instructions));
    match &question.kind {
        DecisionKind::Boolean {
            true_case,
            false_case,
        } => {
            let mut criteria = Map::new();
            if let Some(true_case) = true_case {
                criteria.insert("true".to_string(), json!(true_case));
            }
            if let Some(false_case) = false_case {
                criteria.insert("false".to_string(), json!(false_case));
            }
            if !criteria.is_empty() {
                body.insert("criteria".to_string(), Value::Object(criteria));
            }
        }
        DecisionKind::Choice { choices } => {
            let criteria = choices
                .iter()
                .map(|choice| {
                    // System One needs a description per choice, so a bare value describes itself.
                    let key = key_text(&choice.value);
                    let description = choice.description.clone().unwrap_or_else(|| key.clone());
                    (key, json!(description))
                })
                .collect();
            body.insert("criteria".to_string(), Value::Object(criteria));
        }
        DecisionKind::Score { levels } => {
            // TypeSafe levels are single strings, so a description joins its label.
            let criteria = levels.iter().map(|level| match &level.description {
                Some(description) => json!(format!("{}: {description}", level.label)),
                None => json!(level.label),
            });
            body.insert("criteria".to_string(), Value::Array(criteria.collect()));
        }
    }
    Value::Object(body)
}

// System One takes one `state` value, so conversation roles and non-text parts cannot be kept.
fn encode_state(request: &LlmRequest) -> Result<String> {
    let mut parts = Vec::new();
    for message in &request.messages {
        for block in &message.content {
            match block {
                ContentBlock::Text { text } => parts.push(text.as_str()),
                _ => {
                    return Err(TranslationError::Other(
                        "typesafe_systemone state supports only text".to_string(),
                    ));
                }
            }
        }
    }
    Ok(parts.join("\n"))
}

// Probability maps are keyed by choice value, or by level index for a score.
fn key_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn decode_probabilities(
    answer: &Map<String, Value>,
    path: &str,
    indexed: bool,
) -> Result<Vec<DecisionProbability>> {
    let Some(map) = answer.get("probabilities") else {
        return Ok(Vec::new());
    };
    map.as_object()
        .ok_or_else(|| TranslationError::InvalidType {
            path: format!("{path}.probabilities"),
            expected: "object",
        })?
        .iter()
        .map(|(key, probability)| {
            let value = match key.parse::<u64>() {
                Ok(index) if indexed => json!(index),
                _ => json!(key),
            };
            // A score's legend maps each probability key to its level text.
            let label = answer
                .get("legend")
                .and_then(|legend| legend.get(key))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            Ok(DecisionProbability {
                value,
                label,
                probability: probability
                    .as_f64()
                    .ok_or_else(|| TranslationError::InvalidType {
                        path: format!("{path}.probabilities.{key}"),
                        expected: "number",
                    })?,
            })
        })
        .collect()
}

fn decode_answer(name: &str, answer: &Value) -> Result<NamedDecisionAnswer> {
    let path = format!("$.answers.{name}");
    let answer = object(answer, &path)?;
    let confidence = answer.get("confidence").and_then(Value::as_f64);
    // A refused noul is reported as `refused: true` with `noul: 0`, not as its own type.
    let refused = answer.get("refused").and_then(Value::as_bool) == Some(true);
    let decoded = match required_str(answer, "type", &path)? {
        _ if refused => DecisionAnswer::Refusal,
        "noul" => DecisionAnswer::Boolean {
            probability: required_f64(answer, "noul", &path)?,
        },
        "choice" => DecisionAnswer::Choice {
            choice: answer
                .get("choice")
                .cloned()
                .ok_or_else(|| invalid(&path, "missing choice"))?,
            confidence,
            probabilities: decode_probabilities(answer, &path, false)?,
        },
        "score" => DecisionAnswer::Score {
            score: required_f64(answer, "score", &path)?,
            confidence,
            probabilities: decode_probabilities(answer, &path, true)?,
        },
        "refusal" => DecisionAnswer::Refusal,
        other => {
            return Err(invalid(
                &format!("{path}.type"),
                format!("expected one of noul, choice, score, refusal; got {other:?}"),
            ));
        }
    };
    Ok(NamedDecisionAnswer {
        name: Some(name.to_string()),
        answer: decoded,
    })
}

fn encode_probabilities(probabilities: &[DecisionProbability]) -> Value {
    Value::Object(
        probabilities
            .iter()
            .map(|item| (key_text(&item.value), json!(item.probability)))
            .collect(),
    )
}

fn encode_answer(answer: &DecisionAnswer) -> Value {
    match answer {
        DecisionAnswer::Boolean { probability } => json!({"type": "noul", "noul": probability}),
        DecisionAnswer::Choice {
            choice,
            confidence,
            probabilities,
        } => {
            // The wire shape requires both fields, so unavailable values use the documented
            // sentinels: confidence 0 and an empty map.
            json!({
                "type": "choice",
                "choice": choice,
                "confidence": confidence.unwrap_or(0.0),
                "probabilities": encode_probabilities(probabilities),
            })
        }
        DecisionAnswer::Score {
            score,
            confidence,
            probabilities,
        } => {
            let legend: Map<String, Value> = probabilities
                .iter()
                .filter_map(|item| Some((key_text(&item.value), json!(item.label.as_ref()?))))
                .collect();
            json!({
                "type": "score",
                "score": score,
                "confidence": confidence.unwrap_or(0.0),
                "probabilities": encode_probabilities(probabilities),
                "legend": legend,
            })
        }
        // Only a noul can be refused on this wire, as `refused: true` with `noul: 0`.
        DecisionAnswer::Refusal => json!({"type": "noul", "noul": 0, "refused": true}),
    }
}
