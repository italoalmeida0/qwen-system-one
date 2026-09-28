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

---

## 📜 License

Apache 2.0. Base model weights based on [Qwen 3.5](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX-OPT) by Alibaba Qwen Team.
