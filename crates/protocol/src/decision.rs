// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral types for decision APIs, which answer typed questions about shared input.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One selectable value of a choice question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionChoice {
    /// Choice value. Providers accept strings and booleans.
    pub value: Value,
    /// What the choice means.
    pub description: Option<String>,
}

/// One rung of a score question, ordered from lowest to highest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionLevel {
    /// Level label.
    pub label: String,
    /// What the level means.
    pub description: Option<String>,
}

/// The answer shape a question asks for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionKind {
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
        choices: Vec<DecisionChoice>,
    },
    /// A position on an ordered scale.
    Score {
        /// Ordered levels. A score of `n` sits between level `n` and `n + 1`.
        levels: Vec<DecisionLevel>,
    },
}

/// One typed question asked about the request input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionQuestion {
    /// Question name. Answers are matched to questions by name or by position.
    pub name: Option<String>,
    /// What the model should decide.
    pub instructions: String,
    /// Answer shape.
    pub kind: DecisionKind,
}

/// Questions carried by a decision request. The input itself is in `LlmRequest::messages`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DecisionRequest {
    /// Questions in request order.
    pub questions: Vec<DecisionQuestion>,
}

/// Probability of one choice value or score level.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionProbability {
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
pub enum DecisionAnswer {
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
        probabilities: Vec<DecisionProbability>,
    },
    /// Probability-weighted level index.
    Score {
        /// Interpolated level index, starting at 0.
        score: f64,
        /// Concentration of the probability distribution, when reported.
        confidence: Option<f64>,
        /// Probability of each level, when reported.
        probabilities: Vec<DecisionProbability>,
    },
    /// The model declined to answer.
    Refusal,
}

/// An answer tied to the question it answers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NamedDecisionAnswer {
    /// Name of the answered question.
    pub name: Option<String>,
    /// The answer.
    pub answer: DecisionAnswer,
}
