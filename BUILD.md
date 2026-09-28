# Building qwen-serve

This document outlines how `qwen-serve` is compiled across all 8 target platforms.

## Supported Targets

| Platform Slot | Target Triple | Environment | Linkage |
|---|---|---|---|
| `linux-x64` | `x86_64-unknown-linux-gnu` | `ubuntu-latest` | Static ORT / glibc |
| `linux-arm64` | `aarch64-unknown-linux-gnu` | `ubuntu-24.04-arm` | Static ORT / glibc |
| `win32-x64` | `x86_64-pc-windows-msvc` | `windows-latest` | Static ORT / MSVC CRT |
| `win32-arm64` | `aarch64-pc-windows-msvc` | `windows-11-arm` | Static ORT / MSVC CRT |
| `darwin-arm64` | `aarch64-apple-darwin` | `macos-latest` | Static ORT (Apple Silicon) |
| `darwin-x64` | `x86_64-apple-darwin` | `macos-15-intel` | Dynamic ORT 1.23.0 dylib (`--features mac-x64-legacy`) |
| `linux-x64-musl` | `x86_64-unknown-linux-musl` | `alpine:3.22` in Docker | Dynamic musl ORT + self-extracting bundle |
| `linux-arm64-musl` | `aarch64-unknown-linux-musl` | `alpine:3.22` in Docker | Dynamic musl ORT + self-extracting bundle |

---

## Local Compilation

### Windows / Linux / macOS (ARM64)
```bash
cargo build --release --manifest-path native/qwen-serve/Cargo.toml
```

### macOS Intel (x86_64)
ORT discontinued official x86_64 macOS builds after 1.23.0. We link the official 1.23.0 dylib with api-23 compatibility:
```bash
curl -sL -o /tmp/ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-osx-x86_64-1.23.0.tgz"
mkdir -p ort-libs && tar -xzf /tmp/ort.tgz -C ort-libs --strip-components=1
export ORT_LIB_PATH="$PWD/ort-libs"
export ORT_PREFER_DYNAMIC_LINK=1
cargo build --release --no-default-features --features mac-x64-legacy --manifest-path native/qwen-serve/Cargo.toml
```

### Alpine Linux (musl)
Compiled inside an `alpine:3.22` container using Alpine edge `onnxruntime-dev`, then bundled with `tools/make-bundle.js`.
