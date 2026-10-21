// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered, wire-faithful decision types for the OpenAI Decisions and TypeSafe System One codecs.
//!
//! These carry what the two wire formats express and the `decision` module's routing IR does
//! not: question order, refusals, and per-level labels. They ride on [`LlmRequest::decision`]
//! and [`AggLlmResponse::decisions`] so decision requests flow through the same host pipeline
//! as conversations.
//!
//! [`LlmRequest::decision`]: crate::LlmRequest::decision
//! [`AggLlmResponse::decisions`]: crate::AggLlmResponse::decisions

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One selectable value of a choice question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WireDecisionChoice {
    /// Choice value. Providers accept strings and booleans.
    pub value: Value,
    /// What the choice means.
    pub description: Option<String>,
}

/// One rung of a score question, ordered from lowest to highest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WireDecisionLevel {
    /// Level label.
    pub label: String,
    /// What the level means.
    pub description: Option<String>,
}

/// The answer shape a question asks for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireDecisionKind {
    /// Yes or no, answered as the probability of yes.
    Boolean {
        /// What a yes means, when the caller defines it.
        true_case: Option<String>,
        /// What a no means, when the caller defines it.
        false_case: Option<String>,
    },
    /// One value out of a fixed set.
    Choice {
        /// Allowed values.
        choices: Vec<WireDecisionChoice>,
    },
    /// A position on an ordered scale.
    Score {
        /// Ordered levels. A score of `n` sits between level `n` and `n + 1`.
        levels: Vec<WireDecisionLevel>,
    },
}

/// One typed question asked about the request input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WireDecisionQuestion {
    /// Question name. Answers are matched to questions by name or by position.
    pub name: Option<String>,
    /// What the model should decide.
    pub instructions: String,
    /// Answer shape.
    pub kind: WireDecisionKind,
}

/// Questions carried by a decision request. The input itself is in `LlmRequest::messages`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WireDecisionRequest {
    /// Questions in request order.
    pub questions: Vec<WireDecisionQuestion>,
}

/// Probability of one choice value or score level.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WireDecisionProbability {
    /// Choice value, or the level index for a score.
    pub value: Value,
    /// Level label, when the provider reports one.
    pub label: Option<String>,
    /// Probability in `0..=1`.
    pub probability: f64,
}

/// A model's answer to one question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireDecisionAnswer {
    /// Probability that the answer is yes.
    Boolean {
        /// Probability in `0..=1`.
        probability: f64,
    },
    /// Selected choice.
    Choice {
        /// Selected value.
        choice: Value,
        /// Concentration of the probability distribution, when reported.
        confidence: Option<f64>,
        /// Probability of each choice, when reported.
        probabilities: Vec<WireDecisionProbability>,
    },
    /// Probability-weighted level index.
    Score {
        /// Interpolated level index, starting at 0.
        score: f64,
        /// Concentration of the probability distribution, when reported.
        confidence: Option<f64>,
        /// Probability of each level, when reported.
        probabilities: Vec<WireDecisionProbability>,
    },
    /// The model declined to answer.
    Refusal,
}

/// An answer tied to the question it answers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NamedWireDecisionAnswer {
    /// Name of the answered question.
    pub name: Option<String>,
    /// The answer.
    pub answer: WireDecisionAnswer,
}
