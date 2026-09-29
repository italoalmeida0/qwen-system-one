# Qwen System-One ⚡

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Runtime](https://img.shields.io/badge/Runtime-Rust%20%7C%20ONNX%20Runtime%20%7C%20Axum-orange.svg)]()
[![TypeSafe Jev](https://img.shields.io/badge/Wire%20Protocol-TypeSafe%20Jev%20Compatible-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/Model-Qwen%203.5%200.8B%20Q4%20ONNX-blueviolet.svg)](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX-OPT)

> **High-performance, self-contained System 1 decision engine based on Qwen 3.5 0.8B Q4 ONNX. Zero Python dependencies, pure Rust native binary (`qwen-serve`) with statically linked ONNX Runtime and Axum. 100% wire-compatible with TypeSafe Jev (`POST /v1/systemone`).**
>
> 📋 **Visão Geral, Arquitetura e Próximos Passos:** Veja o [**ROADMAP.md**](ROADMAP.md) para detalhes completos do projeto, objetivos de benchmark e guia de handoff.

---

## 🌟 Key Highlights

- 🔒 **100% Offline & Private:** Zero cloud calls at inference time. Everything runs entirely local on CPU.
- ⚡ **Native C++/Rust Speed:** Statically linked ONNX Runtime with zero-copy past-state preallocation (~270ms - 400ms per decision question on CPU).
- 🔄 **TypeSafe Jev Compatible:** Drop-in replacement for `/v1/systemone`.
- 🌍 **Multilingual:** Handles English, Portuguese, Spanish, German, French, Chinese, Japanese, and 100+ languages natively.
- 📦 **Compact Footprint:** INT4 block quantization (Q4) packs the full model into ~533 MB compressed.
- 💻 **8 Native Platforms Supported:**
  - `linux-x64` (glibc)
  - `linux-arm64` (glibc)
  - `linux-x64-musl` (Alpine Docker self-contained bundle)
  - `linux-arm64-musl` (Alpine Docker self-contained bundle)
  - `win32-x64`
  - `win32-arm64`
  - `darwin-arm64` (Apple Silicon)
  - `darwin-x64` (macOS Intel legacy)

---

## 🚀 Quick Start

### 1. Acquire the Model
```bash
node tools/acquire-model.js
```
Downloads `qwen-0.8b-q4.model` from GitHub Releases, verifies SHA256, and extracts ONNX weights into `models/`.

### 2. Run the Native Server
```bash
# Build & run from source (Rust 1.80+)
cargo run --release --manifest-path native/qwen-serve/Cargo.toml -- --port 8093

# Tune concurrency: N workers x M intra-op threads (defaults: auto = min(cpus,4) workers, cpus/workers threads)
cargo run --release --manifest-path native/qwen-serve/Cargo.toml -- --port 8093 --workers 4 --threads 2

# Or execute a prebuilt binary directly
./qwen-serve --port 8093
```

> **Concurrency model.** Each `--workers` slot owns an independent embed+decoder ONNX session pair.
> Requests are distributed round-robin; when all workers are busy, extra requests queue on the
> worker's mutex (OS-parked, no spin). Total ORT threads ≈ CPU cores, so `--workers 4 --threads 2`
> on an 8-core box saturates the CPU without oversubscription. `/health` reports the active
> `workers` / `threads_per_session` configuration.

### Real KV cache (`--prefix-cache`, on by default)

Prompts are prefilled in chunks: the template prefix (identical for every request with the
same question def) runs once and its `present_*` states (conv/recurrent/KV) are cached;
later requests only prefill the state suffix. Measured on Snapdragon X: sustained latency
**129ms median vs ~1100ms** single-pass (~8.5×), with identical decisions and displayed
probabilities (validated case-by-case). `--prefix-cache off` restores the exact single-pass
path (bit-identical logits to the pinned reference; chunked logits differ by ~2e-5).

### Hardware: CPU only, by design

Execution providers (DirectML/CoreML/NNAPI/QNN) were implemented and measured — then removed
after the real KV cache made the CPU path unbeatable on target hardware (129ms sustained, no
thermal collapse vs 260-900ms+ on accelerators). The runtime is CPU-focused; measurements and
the removed code live in ROADMAP / commit `32498d4` if Fase 2's smaller models ever change
that calculus.

### 3. Send a System 1 Decision Request
```bash
curl -X POST http://127.0.0.1:8093/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "We were billed twice on the March invoice and want a refund.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this ticket?",
        "criteria": {
          "billing": "refunds, charges, payments and invoices",
          "tech": "bugs, crashes, downtime and technical issues",
          "sales": "upgrades, subscriptions, contracts and seat purchases"
        }
      }
    }
  }'
```

Response:
```json
{
  "model": "onnx-community/Qwen3.5-0.8B-ONNX-OPT",
  "answers": {
    "department": {
      "choice": "billing",
      "probability": 0.985
    }
  },
  "usage": {
    "input_tokens": 78,
    "output_tokens": 1
  }
}
```

---

## 🛠️ Verification & Quick Check

Run the automated 10-question triage test suite:
```bash
node tools/quick-check.js
```

Benchmark concurrent throughput (spawns the server, fires bursts at several concurrency levels):
```bash
node tools/bench-concurrent.js --workers 2 --levels 1,2,4,8 --requests 16
```

> **Exact-match cache.** Identical requests (`state` + `questions` + temperature) hit an in-memory
> LRU (default 512 entries, 10 min TTL, tunable via `--cache-size` / `--cache-ttl`) and replay in
> ~1-2ms with zero inference. `/health` reports `cache.hits` / `misses` / `entries`.

---

---

## 🧩 Question Types

### `choice` — Multiple Choice (Single Selection)
Selects one label from fixed options. Supports either a `criteria` map (label -> criterion) or a plain `options` array.

```json
{
  "type": "choice",
  "instructions": "Which department should handle this ticket?",
  "criteria": {
    "billing": "refunds, charges, payments and invoices",
    "tech": "bugs, crashes, downtime and technical issues",
    "sales": "upgrades, subscriptions, contracts and seat purchases"
  }
}
```

Response:
```json
{
  "choice": "billing",
  "probability": 0.985,
  "probabilities": { "billing": 0.985, "tech": 0.01, "sales": 0.005 },
  "confidence": 0.97
}
```

### `score` — Numeric Scoring (0-1)
Outputs a normalized score in the range `[0, 1]`. An optional `threshold` overrides the default 0.5.

```json
{
  "type": "score",
  "instructions": "Rate the urgency of this request",
  "threshold": 0.5
}
```

Response:
```json
{ "score": 0.82, "probability": 0.82, "confidence": 0.74 }
```

### `noul` — Nullability / Binary Classification (Yes/No)
Returns `true` (yes), `false` (no), or `null` (not enough information in the state). Binary case of `choice` with `null` support.

```json
{
  "type": "noul",
  "instructions": "Does this message contain a refund request?"
}
```

Response:
```json
{ "value": true, "probability": 0.93, "confidence": 0.88 }
```

---

## 🌐 HTTP API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness + model info + pool/cache stats |
| `POST` | `/v1/systemone` | TypeSafe Jev decision endpoint |

### `GET /health`
```json
{
  "status": "ok",
  "backend": "qwen-serve",
  "model": "onnx-community/Qwen3.5-0.8B-ONNX-OPT",
  "runtime": "native-rust-onnx",
  "workers": 1,
  "threads_per_session": 12,
  "cache": { "hits": 26, "misses": 8, "entries": 8 }
}
```

### `POST /v1/systemone`

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `state` | `string \| object` | ✅ | Context to evaluate (text or JSON) |
| `questions` | `map<string, QuestionDef>` | ✅ | One or more typed questions (max 255 options per `choice`) |
| `model` | `string` | ❌ | Optional model identifier echoed back |

**Response body:**

| Field | Type | Description |
|---|---|---|
| `model` | `string` | Model identifier |
| `answers` | `map<string, Answer>` | One answer per question key |
| `usage.input_tokens` | `int` | Total input tokens across all questions |
| `usage.output_tokens` | `int` | Number of questions answered |

### Error responses

| Status | Meaning |
|---|---|
| `401` | Missing or invalid API key (only when `LAYA_API_KEY` / `--api-key` is set) |
| `422` | Invalid request body (missing `state`/`questions`, malformed JSON) |
| `404` | Unknown endpoint |
| `413` | Request body too large |
| `500` | Internal inference error (tokenizer, ONNX session, tensor extraction) |

---

## ⚡ Performance

Measured on 12-core Snapdragon X (release build), prompt ~114 tokens, single question:

| Condition | Latency | Throughput |
|---|---|---|
| Cold machine, miss (new inference) | **~233ms** | **~4.3 req/s** |
| Cache hit (identical request) | **~1-2ms** | **~1000 req/s** |
| Sustained load (SoC thermal throttle) | ~1000ms | ~1 req/s |
| Real KV cache sustained (prefix cache) | **129ms median** | **~7.8 req/s** (8.5× vs single-pass) |

CI (GitHub Actions, 2-4 vCPU runners, release): 8/8 platforms green. See `tools/bench-concurrent.js` and `tools/quick-check.js` output in the Actions logs for per-platform numbers.

---

## 🔧 Server Configuration

All flags are CLI arguments (no env vars required except `LAYA_API_KEY`):

| Flag | Default | Description |
|---|---|---|
| `--model-dir` | `models` | Directory with `embed_tokens_q4.onnx`, `decoder_model_merged_q4.onnx`, `tokenizer.json` |
| `--host` | `127.0.0.1` | Bind host |
| `--port` | `8093` | Bind port (0 = pick free port) |
| `--workers` | `0` (auto = 1) | Concurrent inference workers (ONNX session pairs) |
| `--threads` | `0` (auto = cpus/workers) | Intra-op threads per worker |
| `--api-key` | unset | Bearer API key (or `LAYA_API_KEY` env var) |
| `--temperature` | `1.14` | Softmax temperature for probability calibration |
| `--cache-size` | `512` | Exact-match cache entries (0 = disabled) |
| `--cache-ttl` | `600` | Exact-match cache TTL in seconds |
| `--kv-cache-size` | `64` | KV-prefix cache entries (0 = disabled) |
| `--prefix-cache` | `on` | Real KV cache (`on`/`off`) |

---

## 🧪 Development

```bash
# 10-question triage smoke test (spawns server, validates answers)
node tools/quick-check.js

# Concurrent throughput benchmark (levels 1,2,4,8 + cache-hit phase)
node tools/bench-concurrent.js --levels 1,2,4 --requests 8

# Build native binary
cargo build --release --manifest-path native/qwen-serve/Cargo.toml

# Trigger CI builds on all 8 platforms
gh workflow run build-packages.yml
```

---

## 🙏 Credits

- **Base model:** [Qwen 3.5 0.8B](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX-OPT) by Alibaba Qwen Team (Apache 2.0).
- **ONNX export:** [onnx-community](https://huggingface.co/onnx-community) — `Qwen3.5-0.8B-ONNX-OPT` (embed_tokens + decoder_model_merged layout).
- **Training recipe:** [Mapika/decider](https://github.com/Mapika/decider) (Apache 2.0) — mixture builder, Brier loss, temperature calibration.
- **Wire protocol:** [TypeSafe Jev](https://docs.typesafe.ai/concepts/system-one) `/v1/systemone` specification.
- **Reference implementations:** [convai/laya-system-one](https://github.com/convai/laya-system-one) (ModernBERT encoder), [kirp/jpt-0.8b](https://huggingface.co/kirp/jpt-0.8b) (LoRA on Qwen 3.5).

## 📜 License

Apache 2.0. Base model weights based on [Qwen 3.5](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX-OPT) by Alibaba Qwen Team.
