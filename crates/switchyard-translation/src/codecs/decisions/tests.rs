use serde_json::{Value, json};
use switchyard_protocol::ModelId;

use crate::decision::{DecisionAnswer, DecisionKind};
use crate::engine::TranslationEngine;
use crate::error::TranslationError;
use crate::format::WireFormat;
use crate::policy::{PreservationPolicy, TranslationPolicy};
use crate::util::prepare_request_for_target;

const OPENAI: WireFormat = WireFormat::OpenAiDecisions;
const TYPESAFE: WireFormat = WireFormat::TypesafeSystemOne;

// Replays nothing from the source body, so assertions see what the normalized IR re-encodes.
fn rebuilt() -> TranslationPolicy {
    TranslationPolicy {
        preservation: PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    }
}

fn openai_request() -> Value {
    json!({
        "model": "openai/gpt-6-luna-decisions",
        "input": "The package arrived with a broken screen. I want my money back.",
        "questions": [
            {"type": "predicate", "name": "damaged", "instructions": "Does the customer report a damaged item?"},
            {
                "type": "choice", "name": "queue", "instructions": "Which team should handle this ticket?",
                "choices": [
                    {"value": "billing", "description": "Refunds and payments"},
                    {"value": "shipping", "description": "Delivery and damaged items"}
                ]
            },
            {
                "type": "score", "name": "urgency", "instructions": "How urgent is this ticket?",
                "levels": [{"label": "low"}, {"label": "medium"}, {"label": "high"}]
            }
        ],
        "providerOptions": {"gateway": {"zeroDataRetention": true}}
    })
}

fn openai_response() -> Value {
    json!({
        "model": "openai/gpt-6-luna-decisions",
        "answers": [
            {"type": "predicate", "name": "damaged", "probability": 0.95},
            {
                "type": "choice", "name": "queue", "choice": "shipping", "confidence": 0.81,
                "probabilities": [
                    {"value": "billing", "probability": 0.19},
                    {"value": "shipping", "probability": 0.81}
                ]
            },
            {
                "type": "score", "name": "urgency", "score": 1.3, "confidence": 0.62,
                "probabilities": [
                    {"value": 0, "label": "low", "probability": 0.08},
                    {"value": 1, "label": "medium", "probability": 0.54},
                    {"value": 2, "label": "high", "probability": 0.38}
                ]
            }
        ],
        "usage": {"input_tokens": 96, "output_tokens": 0, "total_tokens": 96}
    })
}

fn typesafe_request() -> Value {
    json!({
        "model": "typesafe-ai/jev",
        "state": "I was charged twice for my subscription.",
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": "Is the customer asking for money back?",
                "criteria": {"true": "asks for a refund", "false": "does not"}
            },
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Charges and refunds", "technical": "Bugs and outages"}
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["low", "high"]
            }
        },
        "providerOptions": {"gateway": {"only": ["typesafe-ai"]}}
    })
}

fn typesafe_response() -> Value {
    json!({
        "model": "typesafe-ai/jev",
        "answers": {
            "refund": {"type": "noul", "noul": 0.98},
            "department": {
                "type": "choice", "choice": "billing", "confidence": 0.9,
                "probabilities": {"billing": 0.9, "technical": 0.1}
            },
            "urgency": {
                "type": "score", "score": 0.7, "confidence": 0.5,
                "probabilities": {"0": 0.3, "1": 0.7},
                "legend": {"0": "low", "1": "high"}
            }
        },
        "usage": {"input_tokens": 275, "output_tokens": 20}
    })
}

#[test]
fn openai_request_decodes_questions_and_input() {
    let engine = TranslationEngine::default();
    let decoded = engine
        .decode_request(OPENAI, &openai_request(), &TranslationPolicy::default())
        .unwrap()
        .request;

    assert_eq!(
        decoded.model.as_deref(),
        Some("openai/gpt-6-luna-decisions")
    );
    assert_eq!(
        decoded.messages[0].text_content("\n").as_deref(),
        Some("The package arrived with a broken screen. I want my money back.")
    );
    let questions = decoded.decision.unwrap().questions;
    assert_eq!(questions.len(), 3);
    assert_eq!(questions[0].name.as_deref(), Some("damaged"));
    assert!(matches!(questions[0].kind, DecisionKind::Boolean { .. }));
    assert!(matches!(&questions[1].kind, DecisionKind::Choice { choices } if choices.len() == 2));
    assert!(matches!(&questions[2].kind, DecisionKind::Score { levels } if levels.len() == 3));
}

#[test]
fn same_format_replays_the_exact_body_with_the_routed_model() {
    let engine = TranslationEngine::default();
    for (format, body) in [(OPENAI, openai_request()), (TYPESAFE, typesafe_request())] {
        let policy = TranslationPolicy::default();
        let mut request = engine
            .decode_request(format, &body, &policy)
            .unwrap()
            .request;
        prepare_request_for_target(&mut request, &ModelId::from("vendor/routed"), None);
        let encoded = engine
            .encode_request(format, &request, &policy)
            .unwrap()
            .body;

        let mut expected = body;
        expected["model"] = json!("vendor/routed");
        assert_eq!(encoded, expected, "{format}");
    }
}

#[test]
fn requests_round_trip_through_the_neutral_ir() {
    let engine = TranslationEngine::default();
    for (format, body) in [(OPENAI, openai_request()), (TYPESAFE, typesafe_request())] {
        let request = engine
            .decode_request(format, &body, &rebuilt())
            .unwrap()
            .request;
        let encoded = engine
            .encode_request(format, &request, &rebuilt())
            .unwrap()
            .body;
        assert_eq!(encoded, body, "{format}");
    }
}

#[test]
fn responses_round_trip_through_the_neutral_ir() {
    let engine = TranslationEngine::default();
    for (format, body) in [(OPENAI, openai_response()), (TYPESAFE, typesafe_response())] {
        let response = engine
            .decode_response(format, &body, &rebuilt())
            .unwrap()
            .response;
        let encoded = engine
            .encode_response(format, &response, &rebuilt())
            .unwrap()
            .body;
        // TypeSafe's wire fills confidence and legend, so compare answers rather than usage totals.
        assert_eq!(encoded["answers"], body["answers"], "{format}");
        assert_eq!(encoded["model"], body["model"], "{format}");
    }
}

#[test]
fn openai_request_translates_to_system_one() {
    let engine = TranslationEngine::default();
    let body = engine
        .translate_request(OPENAI, TYPESAFE, &openai_request(), &rebuilt())
        .unwrap()
        .body;

    assert_eq!(
        body["state"],
        "The package arrived with a broken screen. I want my money back."
    );
    assert_eq!(body["questions"]["damaged"]["type"], "noul");
    assert_eq!(
        body["questions"]["queue"]["criteria"],
        json!({"billing": "Refunds and payments", "shipping": "Delivery and damaged items"})
    );
    assert_eq!(
        body["questions"]["urgency"]["criteria"],
        json!(["low", "medium", "high"])
    );
    // Gateway options pass through to the upstream untouched.
    assert_eq!(body["providerOptions"], openai_request()["providerOptions"]);
}

#[test]
fn system_one_request_translates_to_openai() {
    let engine = TranslationEngine::default();
    let body = engine
        .translate_request(TYPESAFE, OPENAI, &typesafe_request(), &rebuilt())
        .unwrap()
        .body;

    assert_eq!(body["input"], "I was charged twice for my subscription.");
    assert_eq!(body["questions"][0]["type"], "predicate");
    assert_eq!(body["questions"][0]["name"], "refund");
    assert_eq!(body["questions"][1]["choices"][1]["value"], "technical");
    assert_eq!(body["questions"][2]["levels"][1]["label"], "high");
}

#[test]
fn system_one_response_translates_to_openai() {
    let engine = TranslationEngine::default();
    let body = engine
        .translate_response(TYPESAFE, OPENAI, &typesafe_response(), &rebuilt())
        .unwrap()
        .body;

    assert_eq!(
        body["answers"][0],
        json!({"type": "predicate", "name": "refund", "probability": 0.98})
    );
    assert_eq!(
        body["answers"][2]["probabilities"][1],
        json!({"value": 1, "label": "high", "probability": 0.7})
    );
    assert_eq!(
        body["usage"],
        json!({"input_tokens": 275, "output_tokens": 20, "total_tokens": 295})
    );
}

#[test]
fn refused_noul_decodes_as_refusal_and_encodes_back() {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "typesafe-ai/jev",
        "answers": {"refund": {"type": "noul", "noul": 0, "refused": true}},
    });
    let response = engine
        .decode_response(TYPESAFE, &body, &rebuilt())
        .unwrap()
        .response;
    assert_eq!(
        response.decisions.as_ref().unwrap()[0].answer,
        DecisionAnswer::Refusal
    );

    let encoded = engine
        .encode_response(TYPESAFE, &response, &rebuilt())
        .unwrap()
        .body;
    assert_eq!(encoded["answers"], body["answers"]);
    let openai = engine
        .encode_response(OPENAI, &response, &rebuilt())
        .unwrap()
        .body;
    assert_eq!(
        openai["answers"][0],
        json!({"type": "refusal", "name": "refund"})
    );
}

#[test]
fn unnamed_questions_are_keyed_by_position_for_system_one() {
    let engine = TranslationEngine::default();
    let body = json!({
        "input": "hi",
        "questions": [{"type": "predicate", "instructions": "Is it a greeting?"}]
    });
    let encoded = engine
        .translate_request(OPENAI, TYPESAFE, &body, &rebuilt())
        .unwrap()
        .body;
    assert_eq!(encoded["questions"]["0"]["type"], "noul");
}

#[test]
fn unknown_question_type_is_rejected_with_the_field_path() {
    let engine = TranslationEngine::default();
    let body = json!({"state": "x", "questions": {"q": {"type": "maybe", "instructions": "?"}}});
    let error = engine
        .decode_request(TYPESAFE, &body, &rebuilt())
        .unwrap_err();
    assert!(
        matches!(&error, TranslationError::InvalidValue { path, .. } if path == "$.questions.q.type"),
        "{error}"
    );
}

#[test]
fn decision_and_conversation_formats_do_not_translate() {
    let engine = TranslationEngine::default();
    let chat = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
    for (source, target, body) in [
        (WireFormat::OpenAiChat, OPENAI, chat.clone()),
        (WireFormat::OpenAiChat, TYPESAFE, chat),
        (OPENAI, WireFormat::OpenAiChat, openai_request()),
        (TYPESAFE, WireFormat::AnthropicMessages, typesafe_request()),
    ] {
        let error = engine
            .translate_request(source, target, &body, &rebuilt())
            .unwrap_err();
        assert!(
            matches!(error, TranslationError::UnsupportedTranslation { .. }),
            "{source} -> {target}"
        );
    }

    let decoded = engine
        .decode_request(OPENAI, &openai_request(), &rebuilt())
        .unwrap()
        .request;
    let error = engine
        .encode_request(WireFormat::OpenAiChat, &decoded, &rebuilt())
        .unwrap_err();
    assert!(matches!(
        error,
        TranslationError::UnsupportedTranslation { .. }
    ));

    let response = engine
        .decode_response(OPENAI, &openai_response(), &rebuilt())
        .unwrap()
        .response;
    let error = engine
        .encode_response(WireFormat::AnthropicMessages, &response, &rebuilt())
        .unwrap_err();
    assert!(matches!(
        error,
        TranslationError::UnsupportedTranslation { .. }
    ));
}

#[test]
fn images_survive_openai_decisions_input() {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "openai/gpt-6-luna-decisions",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Inspect the product in this photo."},
                {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo=", "detail": "high"}
            ]
        }],
        "questions": [{"type": "predicate", "name": "visible_damage", "instructions": "Does the product have visible damage?"}]
    });
    let request = engine
        .decode_request(OPENAI, &body, &rebuilt())
        .unwrap()
        .request;
    let encoded = engine
        .encode_request(OPENAI, &request, &rebuilt())
        .unwrap()
        .body;
    assert_eq!(encoded["input"], body["input"]);

    // System One takes one text `state`, so an image cannot be carried.
    assert!(
        engine
            .translate_request(OPENAI, TYPESAFE, &body, &rebuilt())
            .is_err()
    );
}

#[test]
fn only_documented_gateway_fields_cross_formats() {
    let engine = TranslationEngine::default();
    let mut body = openai_request();
    body["safety_identifier"] = json!("user-1");
    body["unrelated"] = json!(true);

    let typesafe = engine
        .translate_request(OPENAI, TYPESAFE, &body, &rebuilt())
        .unwrap()
        .body;
    assert!(typesafe.get("providerOptions").is_some());
    assert!(typesafe.get("safety_identifier").is_none());
    assert!(typesafe.get("unrelated").is_none());

    let openai = engine
        .translate_request(OPENAI, OPENAI, &body, &rebuilt())
        .unwrap()
        .body;
    assert_eq!(openai["safety_identifier"], "user-1");
}

#[test]
fn lossy_conversions_are_reported_and_a_strict_policy_refuses_them() {
    let engine = TranslationEngine::default();
    let strict = TranslationPolicy {
        lossy_conversion_policy: crate::policy::LossyConversionPolicy::Reject,
        ..rebuilt()
    };

    // The criteria of a noul have no OpenAI predicate equivalent.
    let output = engine
        .translate_request(TYPESAFE, OPENAI, &typesafe_request(), &rebuilt())
        .unwrap();
    assert!(!output.diagnostics.is_empty());
    assert!(matches!(
        engine.translate_request(TYPESAFE, OPENAI, &typesafe_request(), &strict),
        Err(TranslationError::LossyConversion(_))
    ));

    let described = json!({
        "input": "x",
        "questions": [{"type": "score", "name": "s", "instructions": "?", "levels": [{"label": "a", "description": "b"}, {"label": "c"}]}]
    });
    let output = engine
        .translate_request(OPENAI, TYPESAFE, &described, &rebuilt())
        .unwrap();
    assert_eq!(
        output.body["questions"]["s"]["criteria"],
        json!(["a: b", "c"])
    );
    assert_eq!(output.diagnostics.len(), 1);
}

#[test]
fn duplicate_system_one_keys_are_rejected_instead_of_overwritten() {
    let engine = TranslationEngine::default();
    let body = json!({
        "input": "x",
        "questions": [
            {"type": "predicate", "instructions": "first, unnamed so keyed 0"},
            {"type": "predicate", "name": "0", "instructions": "second, named 0"}
        ]
    });
    let error = engine
        .translate_request(OPENAI, TYPESAFE, &body, &rebuilt())
        .unwrap_err();
    assert!(
        matches!(error, TranslationError::InvalidValue { .. }),
        "{error}"
    );
}
