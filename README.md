# ComfyUI OpenRouter Simple

One OpenRouter node for multimodal context in and text only out. It keeps the useful inference controls, filters the model list to the exact combination of connected inputs, compresses media locally before submission, and treats timeout/Stop as real cancellation rather than a background request that keeps running.

## What the node exposes

| Input | Behavior |
| --- | --- |
| `system_prompt` | Optional system instruction. |
| `user_prompt` | The user message sent with any connected media. |
| `api_key` | Optional OpenRouter key. Enter it here or convert the widget to an input and connect a STRING/API Key box. Blank uses the environment fallback. |
| `model` | Live OpenRouter text-output models, filtered by connected modalities. |
| `reasoning_effort` | `auto`, `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`. Unsupported explicit reasoning fails before submission. |
| `timeout_seconds` | One deadline for catalog refresh, preprocessing, and the chat request. Credits use only time still left, capped at 3 seconds. |
| `temperature` | Sent only when supported by the selected model. |
| `max_tokens` | One total output budget shared by hidden reasoning and visible completion. It maps to `max_completion_tokens` when available, then legacy `max_tokens`. |
| `response_format` | `text` or `json_object`. JSON fails before submission when the model does not advertise structured output support. |
| `zdr` | Off by default. When enabled, sends only `provider.zdr: true`. |
| `regenerate` | On by default. `true` makes a fresh request each queue; `false` reuses ComfyUI's cached outputs while this node and all upstream inputs are unchanged. |
| `image*` / `video*` / `audio*` | Optional context inputs that progressively grow from one visible socket to three per modality, nine total. They never become generation outputs. |

Outputs are `text`, compact JSON `info`, and `credits`. There are no IMAGE, VIDEO, or AUDIO outputs and the request hardcodes `modalities: ["text"]`.

`regenerate=false` uses ComfyUI's normal in-memory execution cache; it does not create a separate persistent cache. Editing a prompt or setting, changing connected media, or changing another upstream input invalidates the cache and runs the node once to produce the new output. Restarting ComfyUI may also clear the reusable output.

## Install

Requires ComfyUI v0.32.0 or newer, Python 3.10+, and `ffmpeg`/`ffprobe` for video or audio inputs.

Install from the Comfy Registry:

```bash
comfy node install comfyui-openrouter-simple
```

Or install directly from GitHub:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/theshubzworld/ComfyUI-OpenRouter-API.git
python -m pip install -r ComfyUI-OpenRouter-API/requirements.txt
```

Restart ComfyUI, then add **Shubz → OpenRouter → Shubz - OpenRouter Simple (Text Output)** (or search for **Shubz** or **OpenRouter**).

Enter a generation key in `api_key`, or convert that widget to an input and connect a STRING/API Key box. The explicit key takes precedence. Keys entered in a widget or connected box can appear in saved workflows, generated-media metadata and ComfyUI error reports. Clear them before sharing those files; the environment fallback keeps keys out of graph inputs.

You can also leave `api_key` blank and set the key in ComfyUI's environment:

```bash
export OPENROUTER_API_KEY="your-generation-key"
```

Environment fallback checks `OPENROUTER_API_KEY`, then legacy `LLM_KEY`, ignoring blank values. Existing workflows without `api_key` keep working with this fallback. An optional `OPENROUTER_MANAGEMENT_KEY` enables true account-credit lookup. Without it, the `credits` output reports the generation key's `limit_remaining` from `/key` and labels it as an API-key limit—not account credit.

## Media preparation

The limits are decimal bytes measured before base64:

- IMAGE: WebP, at most 1,000,000 bytes. Quality is reduced before dimensions; spatial fallback preserves aspect ratio with Pillow Lanczos.
- VIDEO: MP4/H.264/AAC, at most 10,000,000 bytes. An untrimmed compatible MP4 already below the cap is preserved byte-for-byte. Required conversion or trimming uses a two-pass budget derived from duration and capped at the smaller of 10 MB and the source size, so preparation never enlarges an under-limit video. Spatial fallback uses ffmpeg Lanczos and only lowers frame rate when the available bitrate is unusually small.
- AUDIO: MP3, at most 1,000,000 bytes. Bitrate is derived from duration; SoXr is preferred, with high-precision FFmpeg SWR fallback when SoXr is unavailable.

Up to three items per modality are prepared concurrently in temporary directories. The chat POST does not begin until every connected input is below its cap. A media item that cannot safely reach its cap fails locally, cancels its sibling preparations, and prevents the paid request.

## Model filtering

The backend fetches OpenRouter's model metadata, keeps only models whose output modalities include text, and caches the normalized snapshot. The browser computes this intersection whenever a media cable changes:

```text
required = text + every connected kind among image/video/audio
eligible = required ⊆ model.input_modalities
```

The current selection is retained only if it remains eligible. Otherwise the node returns to **choose a compatible OpenRouter model**; it never silently changes to another paid model. Additional sockets of the same kind do not change the modality intersection. The backend repeats the same compatibility check before preprocessing.

The Models API does not publish a universal per-request attachment-count limit. OpenRouter documents that file counts vary by provider and model, so the node's three-per-kind cap is a UI and resource bound—not a promise that every compatible model accepts all nine items. Provider rejections are surfaced without dropping attachments.

If live metadata is unavailable, a prior disk cache is used and marked stale. With no live or cached catalog, execution stops clearly rather than guessing that a media model is compatible.

## Cancellation and failures

- ComfyUI Stop is polled at 100 ms or less during HTTP and encoder work.
- Active HTTP tasks are cancelled and their sessions closed.
- Active ffmpeg/ffprobe children receive terminate, then kill after a bounded grace period, and are reaped.
- Temporary directories are removed on success, rejection, timeout, and Stop.
- The paid chat POST is never automatically retried, avoiding ambiguous double spend.
- Error bodies are capped and secrets, bearer tokens, and URL query strings are redacted.
- Provider rejections surface sanitized nested provider diagnostics and opt-in router metadata when OpenRouter supplies them; inline media remains redacted.
- Credits failure is non-fatal after a successful text completion.

OpenRouter rejections remain normal ComfyUI execution errors. Stop uses ComfyUI's native interruption exception, so it is shown as an interrupted workflow rather than a fabricated API failure.

## Deliberately excluded

This node does not expose seed, tools/functions, web search, PDFs/files, chat history, more than three attachments per modality, streaming partial text, JSON Schema, top-p/top-k/min-p, penalties, logprobs, provider ordering, price/latency routing, fallback models, transforms, plugins, or media generation.

## OpenRouter contracts used

- [Chat Completions request](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion)
- [Reasoning tokens and effort](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [Model metadata](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties)
- [Provider routing and ZDR](https://openrouter.ai/docs/guides/routing/provider-selection)
- [Router metadata for diagnostics](https://openrouter.ai/docs/guides/features/router-metadata)
- [Image inputs](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding)
- [Video inputs](https://openrouter.ai/docs/guides/overview/multimodal/videos)
- [Audio inputs](https://openrouter.ai/docs/guides/overview/multimodal/audio)
- [Account credits](https://openrouter.ai/docs/api/api-reference/credits/get-remaining-credits) and [current API key](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-key)

## Verification

```bash
./scripts/verify fast
./scripts/verify full
```

`fast` covers the public contract, model intersection, payload mapping, stalled HTTP deadline, rejection redaction, credits failure, and real child-process cancellation. `full` also generates and compresses image, video, and audio fixtures. Set `PYTHON_BIN` if ComfyUI uses a non-default Python.

No verification command makes a paid OpenRouter request. Local fake endpoints exercise the HTTP boundary.

## Architecture and operational boundary

- `node.py` owns the nine-slot ComfyUI input/output contract and request lifecycle.
- `openrouter_api/models.py` owns live metadata normalization and the stale cache.
- `openrouter_api/media.py` owns exact pre-base64 caps and encoder selection.
- `openrouter_api/cancellation.py` owns the shared deadline, Stop polling, and subprocess cleanup.
- `openrouter_api/client.py` owns bounded OpenRouter HTTP and credits lookup.
- `web/` owns display-only live filtering; it is never authoritative for execution.

This is a standalone public custom node distributed through [GitHub](https://github.com/theshubzworld/ComfyUI-OpenRouter-API) and the Comfy Registry. Installing it does not deploy it to a RunPod template. There is no project brain page yet.
