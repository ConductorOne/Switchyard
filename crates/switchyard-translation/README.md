# switchyard-translation

Pure Rust translation between OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages
request, response, and streaming formats.

The crate translates through provider-neutral LLM types from `switchyard-protocol` and does not
depend on provider SDKs, HTTP servers, Python, or FFI bindings.

## Checked native Anthropic request edits

`NativeAnthropicRequest` keeps the original native JSON outside the mutable IR.
Use it when routing to a compatible native Anthropic target:

```rust
use switchyard_translation::{NativeAnthropicRequest, TranslationPolicy};

# fn example(body: &serde_json::Value) -> switchyard_translation::Result<()> {
let policy = TranslationPolicy::default();
let native = NativeAnthropicRequest::decode(body, &policy)?;
let mut edited = native.request().clone();
edited.model = Some("target-model".to_string());
edited.reasoning.effort = Some("high".to_string());
edited.preservation.requests.clear();
let outgoing = native.encode(&edited, &policy)?;
# Ok(())
# }
```

Run routing and policy checks on the cloned request before encoding. Keep the
native value from ingress until those checks finish. Encoding ignores the edited
request's preservation map. It applies supported changes and keeps all other
native JSON values, including unknown fields, system arrays, tool metadata,
message roles, content blocks and tool IDs.

Supported edits are model, stream, sampling controls, maximum output tokens,
raw thinking, effort, and removal of existing native top-level extensions.
Effort changes only `output_config.effort`; they keep other output and thinking
settings. The host must choose a combination the target model accepts. Raw
thinking replacement is a whole-field edit. Unknown native extensions remain
unchanged as caller data, or can be removed by policy. Changing extension values
or adding keys fails; host or algorithm metadata never becomes provider authority
through a generic extension merge. Changes to normalized `parallel_tool_calls`
also fail because the native control belongs to `tool_choice`.

Changes to instructions, message history, tools, tool choice, output schema or
schema enforcement, or typed `decision` data return an error. Do not retry that error by replaying the
original body. Such changes need an explicitly supported native mapping or a
separate route that allows reconstruction. Required prompts and policy edits
must stay effective.

Embedded `_switchyard_translation` metadata and native top-level internal
preservation, provenance and Codex bookkeeping keys are rejected. `Debug` output
is redacted, including when this value is nested in a derived `Debug` context.
The type has no serialization or deserialization support. Capability checks reject
incompatible normalized content even if lossy conversion is allowed. Opaque
content also fails when a content capability is disabled, since its compatibility
cannot be checked. Native tools obey tool, parallel-tool and code-execution
restrictions even when they lack a normalized tool definition. These checks do
not prove that a model supports an opaque future feature.

Only use this encoder for a native Anthropic target. For another provider or
format, use normal IR translation with the target's restrictions. The native
value has no foreign-format encoder and its decoded IR has no preserved body.
This API does not change response or stream preservation, HTTP headers, or the
standalone client's request policies.

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
require an identified call and one Chat choice. Google's Chat endpoint omits
tool indices and sends each call whole, so a call without an index is keyed by
its id. Once calls are keyed by id, a fragment with neither index nor id, or a
second named call with a known id, fails. Multiple choices fail even
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
