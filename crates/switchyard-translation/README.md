# switchyard-translation

Pure Rust translation between OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages
request, response, and streaming formats.

The crate translates through provider-neutral LLM types from `switchyard-protocol` and does not
depend on provider SDKs, HTTP servers, Python, or FFI bindings.

## Google-compatible Chat tool signatures

`ToolCall.google_thought_signature` holds the exact opaque string from
`tool_calls[].extra_content.google.thought_signature`. It belongs to that call.
Unsigned calls keep `None`. It is not reasoning text or an Anthropic signature.
Chat request, response, and SSE codecs rebuild the Google envelope from this field.
Clearing raw request or response preservation does not clear the field.

`LlmResponseChunk::ToolCallDelta.google_thought_signature` is a complete snapshot,
not a string fragment. A late metadata-only update is allowed. The Chat stream
decoder and encoder ignore repeated identical snapshots and reject conflicting
or non-string signatures without including their values in errors. Argument and
name fragments remain separate. Accumulation and synthetic streaming retain the
signature on its call index.

Same-format replay removes repeated Google snapshots, including repeats in one
event. It also removes repeated complete call ids on the first signature update
and on later updates to that signed call, even without a signature. Metadata-only
calls get an empty `function` object so OpenAI SDK stream accumulation accepts
them. Other raw fields and argument fragments remain unchanged. Signed streams
require explicit tool indices and one Chat choice. Multiple choices fail even
with omitted or duplicate indices, before, alongside, or after a signature.
Unsigned Chat streams keep their existing behavior.
Distinct complete call ids at one index also make a later signature ambiguous
and fail. A signature may arrive before the first id; identical id updates remain
valid.

The host must keep this state only for the selected Google provider and model.
Before sending history to a different provider, clear each call's
`google_thought_signature` and raw request preservation. Other wire codecs do not
emit Google signatures. The Chat codec cannot identify a provider from its wire
format alone. Do not remove other providers' reasoning fields or signatures.

Clients must return the full assistant tool calls, including `extra_content`, in
the next request. Clients that rebuild calls from only id, name, and arguments
lose required state. This supports Google-compatible Chat tools, not native
Google APIs or Google Responses.

See Google's [thought signature contract](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures)
and [OpenAI compatibility guide](https://ai.google.dev/gemini-api/docs/openai).

## License

Licensed under the Apache License, Version 2.0. See the
[Switchyard repository](https://github.com/NVIDIA-NeMo/Switchyard) for details.
