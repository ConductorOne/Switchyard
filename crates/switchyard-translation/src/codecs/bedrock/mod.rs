// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Amazon Bedrock Converse wire-format codecs.

mod buffered;
pub mod stream;

pub use buffered::BedrockConverseCodec;
