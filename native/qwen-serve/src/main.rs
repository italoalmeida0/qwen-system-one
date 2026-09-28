//! qwen-serve: self-contained native HTTP server for Qwen 3.5 / JPT models.
//!
//! Embeds: ONNX Runtime (statically linked via `ort`), Hugging Face tokenizers
//! (pure Rust), Axum HTTP server.
//!
//! Wire protocol: 100% compatible with TypeSafe Jev (/v1/systemone).

mod prompt;
mod schema;

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::anyhow;
use axum::{extract::State, http::StatusCode, response::Json, routing::post, Router};
use clap::Parser;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{DynValue, Tensor};
use tokenizers::Tokenizer;
use tower_http::cors::CorsLayer;
use tracing::info;

use schema::*;

#[derive(Parser, Debug)]
#[command(name = "qwen-serve", about = "Self-contained Qwen 3.5 ONNX native inference server")]
struct Args {
    /// Model directory (embed_tokens_q4.onnx, decoder_model_merged_q4.onnx, tokenizer.json)
    #[arg(long, default_value = "models")]
    model_dir: PathBuf,
    /// Bind host
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Bind port (0 = pick free port, prints it for the spawner)
    #[arg(long, default_value_t = 8093)]
    port: u16,
    /// Intra-op threads per worker (0 = auto: cpus / workers)
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Number of concurrent inference workers (ONNX session pairs).
    /// 0 = auto (1: a single worker using ALL cores gives the lowest
    /// latency because this decoder scales ~linearly with threads;
    /// more workers only help past ~16 cores). Extra requests queue on
    /// the worker's mutex (round-robin) inside spawn_blocking.
    #[arg(long, default_value_t = 0)]
    workers: usize,
    /// Bearer API key (optional; also LAYA_API_KEY)
    #[arg(long)]
    api_key: Option<String>,
    /// Chunked prefill + template prefix cache ("KV real"): caches the
    /// present_* states of the template chunk and reuses them on later
    /// requests with the same question def, so only the state suffix is
    /// re-prefilled. "off" restores the exact single-pass path (bit-identical
    /// logits to the pinned reference; chunked logits differ by ~2e-5).
    #[arg(long, value_name = "on|off", default_value = "on")]
    prefix_cache: String,
    /// Temperature for softmax (default 1.140 for Qwen 3.5 / JPT)
    #[arg(long, default_value_t = 1.140)]
    temperature: f32,
    /// Exact-match cache size (responses; 0 = disabled)
    #[arg(long, default_value_t = 512)]
    cache_size: usize,
    /// Exact-match cache TTL in seconds
    #[arg(long, default_value_t = 600)]
    cache_ttl: u64,
}

/// One inference worker: an independent embed + decoder session pair.
///
/// ONNX `Session::run` takes `&mut self`, so a single session can only run
/// one inference at a time. A pool of N pairs allows N concurrent requests.
/// Requests are distributed round-robin via `AppState::next`; when more
/// requests than workers arrive, they queue naturally on the worker's
/// `Mutex` inside `spawn_blocking` threads (no busy-spin, no deadlock).
struct Worker {
    embed: std::sync::Mutex<Session>,
    decoder: std::sync::Mutex<Session>,
}

/// Exact-match response cache (idea ported from the call_me_maybe reference
/// project: identical prompt -> replay the stored answer at ~0ms).
///
/// Key = u64 hash of the canonical request bytes (state + questions +
/// temperature). Value = the full OutBody. Bounded LRU (default 512
/// entries) + TTL (default 10 min) so memory stays flat and stale answers
/// expire. Hits skip tokenization AND both ONNX sessions entirely —
/// the biggest possible saving (no FLOPs, no heat) for retries, polls
/// and repeated triage states.
struct ExactCache {
    map: HashMap<u64, (OutBody, Instant)>,
    order: VecDeque<u64>,
    capacity: usize,
    ttl_secs: u64,
    hits: u64,
    misses: u64,
}

impl ExactCache {
    fn new(capacity: usize, ttl_secs: u64) -> Self {
        Self { map: HashMap::new(), order: VecDeque::new(), capacity, ttl_secs, hits: 0, misses: 0 }
    }
    fn get(&mut self, key: u64) -> Option<OutBody> {
        if let Some((body, at)) = self.map.get(&key) {
            if at.elapsed().as_secs() < self.ttl_secs {
                self.hits += 1;
                return Some(body.clone());
            }
            self.map.remove(&key);
        }
        self.misses += 1;
        None
    }
    fn put(&mut self, key: u64, body: OutBody) {
        if self.map.contains_key(&key) {
            self.map.insert(key, (body, Instant::now()));
            return;
        }
        while self.order.len() >= self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            } else {
                break;
            }
        }
        self.order.push_back(key);
        self.map.insert(key, (body, Instant::now()));
    }
}

fn cache_key(body: &InBody, temperature: f32) -> u64 {
    // Canonical bytes: serde_json with preserve_order keeps caller key order.
    let mut buf = serde_json::to_vec(body).unwrap_or_default();
    buf.extend_from_slice(&temperature.to_le_bytes());
    let mut h = std::collections::hash_map::DefaultHasher::new();
    buf.hash(&mut h);
    h.finish()
}

struct AppState {
    workers: Vec<Worker>,
    next: AtomicUsize,
    tokenizer: Tokenizer,
    precreated_past: Vec<(&'static str, DynValue)>,
    /// Decoder accepts `num_logits_to_keep` (the OPT fused-op export does;
    /// the standard primitive-op export computes logits for every position).
    num_logits_input: bool,
    /// Chunked-prefill prefix cache: template-token prefix -> the matching
    /// past_* inputs (the template chunk's present_* outputs), keyed by the
    /// prefix ids themselves. Bounded (cleared on overflow); shared by all
    /// workers — the states are plain tensors, safe for any decoder session.
    prefix_cache: std::sync::Mutex<std::collections::HashMap<Vec<i64>, Vec<(&'static str, DynValue)>>>,
    prefix_cache_on: bool,
    temperature: f32,
    api_key: Option<String>,
    workers_count: usize,
    threads_per_session: usize,
    exact_cache: std::sync::Mutex<ExactCache>,
}

fn oe<T, E: std::fmt::Debug>(r: Result<T, E>) -> anyhow::Result<T> {
    r.map_err(|e| anyhow!("{e:?}"))
}

/// present_* output name matching a past_* input name. Validated on the
/// merged decoder graph (kv-oracle): past_key_values.N -> present.N,
/// past_conv.N -> present_conv.N, past_recurrent.N -> present_recurrent.N.
fn present_name(past: &str) -> String {
    if let Some(rest) = past.strip_prefix("past_key_values.") {
        format!("present.{rest}")
    } else if let Some(rest) = past.strip_prefix("past_conv.") {
        format!("present_conv.{rest}")
    } else if let Some(rest) = past.strip_prefix("past_recurrent.") {
        format!("present_recurrent.{rest}")
    } else {
        format!("present.{past}")
    }
}

/// Split point for chunked prefill: the token prefix of `token_ids` shared
/// with `split_prompt` (the template-only string). The divergence point
/// depends only on fixed template strings, so the split is deterministic per
/// question def and the prefix-cache key stays stable across states. None
/// when chunking does not apply (empty/too-short split).
fn prefix_split_len(tokenizer: &Tokenizer, split_prompt: &str, token_ids: &[i64]) -> Option<usize> {
    let enc = tokenizer.encode(split_prompt, false).ok()?;
    let prefix_ids = enc.get_ids();
    let mut p = 0usize;
    while p < prefix_ids.len() && p + 1 < token_ids.len() && token_ids[p] == prefix_ids[p] as i64 {
        p += 1;
    }
    if p >= 1 {
        Some(p)
    } else {
        None
    }
}

// Trimmed runtimes: tokio multi_thread with exactly 2 core threads
// (accept + dispatch only; inference runs on spawn_blocking). The ORT
// intra pool owns the other cores, so a big default tokio pool would
// oversubscribe and fight the decoder for L3 / bandwidth.
#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let api_key = args.api_key.or_else(|| std::env::var("LAYA_API_KEY").ok());

    // ---- Pool sizing: one fat worker beats many thin ones ----
    // Measured (Qwen3.5-0.8B Q4, prefill ~120 tokens): decoder latency
    // scales ~linearly with intra threads (1t=2965ms, 2t=1402ms,
    // 4t=695ms, 6t=503ms, 12t=233ms). Splitting cores across workers
    // only adds contention (2x6t concurrent = ~1000ms each), so the
    // throughput-optimal default is ONE worker with ALL cores:
    // throughput = 1/latency. More workers help only past ~16 cores.
    // threads=0 -> all cpus on the single worker.
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let workers_count = if args.workers > 0 { args.workers } else { 1 };
    let threads_per_session = if args.threads > 0 {
        args.threads
    } else {
        (cpus / workers_count).max(1)
    };
    info!("pool: {workers_count} workers x {threads_per_session} intra threads ({cpus} cpus)");

    // ---- ONNX Runtime init ----
    ort::init().with_name("qwen-serve").commit();

    let opt_level = GraphOptimizationLevel::Level3;

    let embed_path = args.model_dir.join("embed_tokens_q4.onnx");
    let decoder_path = args.model_dir.join("decoder_model_merged_q4.onnx");
    info!("loading embed session from {}", embed_path.display());
    info!("loading decoder session from {}", decoder_path.display());

    let mut workers: Vec<Worker> = Vec::with_capacity(workers_count);
    for i in 0..workers_count {
        let mut embed_builder = oe(Session::builder()?.with_optimization_level(opt_level))?;
        embed_builder = oe(embed_builder.with_intra_threads(threads_per_session))?;
        // Single-chain model: sequential execution avoids inter-op pool overhead.
        embed_builder = oe(embed_builder.with_inter_threads(1))?;
        // Same no-spin rationale as the decoder (see below): a spinning
        // embed pool alone keeps the SoC hot between requests.
        embed_builder = oe(embed_builder.with_memory_pattern(false))?;
        embed_builder = oe(embed_builder.with_intra_op_spinning(false))?;
        embed_builder = oe(embed_builder.with_inter_op_spinning(false))?;
        let embed_session = oe(embed_builder.commit_from_file(&embed_path))?;

        let mut decoder_builder = oe(Session::builder()?.with_optimization_level(opt_level))?;
        decoder_builder = oe(decoder_builder.with_intra_threads(threads_per_session))?;
        decoder_builder = oe(decoder_builder.with_inter_threads(1))?;
        // Variable seq_len per request: disable static memory-pattern reuse
        // (avoids reallocation stalls when shapes change between requests).
        decoder_builder = oe(decoder_builder.with_memory_pattern(false))?;
        // Critical on mobile SoCs (Snapdragon etc.): ORT intra threads SPIN
        // by default after each run. 12 spinning threads = constant 100%
        // load = the SoC drops clocks and never boosts again (first req at
        // full speed, all later reqs throttled). Park instead of spin so
        // cores idle between requests and boost works per request.
        decoder_builder = oe(decoder_builder.with_intra_op_spinning(false))?;
        decoder_builder = oe(decoder_builder.with_inter_op_spinning(false))?;
        let decoder_session = oe(decoder_builder.commit_from_file(&decoder_path))?;
        workers.push(Worker {
            embed: std::sync::Mutex::new(embed_session),
            decoder: std::sync::Mutex::new(decoder_session),
        });
        info!("worker {}/{workers_count} ready", i + 1);
    }

    // Log decoder I/O (defines prefix-cache viability: present_* outputs?)
    {
        let first = workers[0].decoder.lock().unwrap();
        let ins: Vec<String> = first.inputs().iter().map(|i| i.name().to_string()).collect();
        let outs: Vec<String> = first.outputs().iter().map(|o| o.name().to_string()).collect();
        info!("decoder inputs ({}): {:?}", ins.len(), ins);
        info!("decoder outputs ({}): {:?}", outs.len(), outs);
    }
    // Pre-create all static past states once (shape is identical for every
    // worker, so probe the first decoder session and share the tensors).
    // Shapes come from the graph itself: the conv cache is 3-wide in the OPT
    // export (CausalConvWithState fused op) and 4-wide in the standard
    // export (primitive ops), while recurrent/KV states match both.
    let mut precreated_past: Vec<(&'static str, DynValue)> = Vec::new();
    let mut num_logits_input = false;
    {
        let first = workers[0].decoder.lock().unwrap();
        for inp in first.inputs().iter() {
            let name = inp.name().to_string();
            if name == "num_logits_to_keep" {
                num_logits_input = true;
            } else if name.starts_with("past_key_values") {
                let data = vec![0.0f32; 0];
                let t = oe(Tensor::from_array((vec![1, 2, 0, 256], data.into_boxed_slice())))?;
                precreated_past.push((name.leak() as &'static str, DynValue::from(t)));
            } else if name.starts_with("past_conv") || name.starts_with("past_recurrent") {
                let dims: Vec<usize> = match inp.dtype() {
                    ort::value::ValueType::Tensor { shape, .. } => {
                        shape.iter().map(|&d| if d < 0 { 1 } else { d as usize }).collect()
                    }
                    _ => continue,
                };
                let n: usize = dims.iter().product();
                let data = vec![0.0f32; n];
                let t = oe(Tensor::from_array((dims, data.into_boxed_slice())))?;
                precreated_past.push((name.leak() as &'static str, DynValue::from(t)));
            }
        }
    }
    info!(
        "pre-created {} past state tensors (num_logits_to_keep input: {num_logits_input})",
        precreated_past.len()
    );

    // ---- Tokenizer ----
    let tok_path = args.model_dir.join("tokenizer.json");
    info!("loading tokenizer from {}", tok_path.display());
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("{e:?}"))?;

    let state = Arc::new(AppState {
        workers,
        next: AtomicUsize::new(0),
        tokenizer,
        precreated_past,
        num_logits_input,
        prefix_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        prefix_cache_on: args.prefix_cache.trim().to_ascii_lowercase() != "off",
        temperature: args.temperature,
        api_key,
        workers_count,
        threads_per_session,
        exact_cache: std::sync::Mutex::new(ExactCache::new(args.cache_size, args.cache_ttl)),
    });

    let app = Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/health", axum::routing::get(health))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("{}:{}", args.host, args.port)).await?;
    let addr = listener.local_addr()?;
    println!("QWEN_READY {addr}");
    info!("listening on {addr}");

    axum::serve(listener, app).await?;
    Ok(())
}

async fn health(State(st): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let (cache_hits, cache_misses, cache_len) = st
        .exact_cache
        .lock()
        .map(|c| (c.hits, c.misses, c.map.len()))
        .unwrap_or((0, 0, 0));
    Json(serde_json::json!({
        "status": "ok",
        "backend": "qwen-serve",
        "model": "onnx-community/Qwen3.5-0.8B-ONNX-OPT",
        "runtime": "native-rust-onnx",
        "workers": st.workers_count,
        "threads_per_session": st.threads_per_session,
        "cache": { "hits": cache_hits, "misses": cache_misses, "entries": cache_len },
    }))
}

async fn systemone(
    State(st): State<Arc<AppState>>,
    req: axum::extract::Request,
) -> Result<Json<OutBody>, (StatusCode, Json<serde_json::Value>)> {
    if let Some(key) = &st.api_key {
        let ok = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|v| v == format!("Bearer {key}"))
            .unwrap_or(false);
        if !ok {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "Unauthorized" })),
            ));
        }
    }

    let bytes = axum::body::to_bytes(req.into_body(), 8 * 1024 * 1024)
        .await
        .map_err(|e| {
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({ "error": format!("read body: {e}") })),
            )
        })?;

    let body: InBody = serde_json::from_slice(&bytes).map_err(|e| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "error": format!("invalid body: {e}") })),
        )
    })?;

    if body.questions.is_empty() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "error": "missing state/questions" })),
        ));
    }

    // Exact-match cache: identical request bytes -> replay stored answer.
    // Hits skip tokenization + both ONNX sessions (zero FLOPs, zero heat).
    let key = cache_key(&body, st.temperature);
    if let Ok(mut cache) = st.exact_cache.lock() {
        if let Some(hit) = cache.get(key) {
            return Ok(Json(hit));
        }
    }

    let st2 = st.clone();
    // Round-robin: each request pins to one worker for its whole lifetime.
    // If all workers are busy the request waits on that worker's Mutex
    // inside spawn_blocking (OS-parked, no spin). Requests are independent,
    // so head-of-line blocking across workers cannot happen.
    let worker_idx = st.next.fetch_add(1, Ordering::Relaxed) % st.workers.len();
    let out = tokio::task::spawn_blocking(move || infer_all(&st2, worker_idx, &body))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("task: {e}") })),
            )
        })??;

    // Store exact-match answer for future identical requests.
    if let Ok(mut cache) = st.exact_cache.lock() {
        cache.put(key, out.clone());
    }

    Ok(Json(out))
}

fn infer_all(st: &AppState, worker_idx: usize, body: &InBody) -> Result<OutBody, (StatusCode, Json<serde_json::Value>)> {
    let err500 = |what: &str, e: std::fmt::Arguments| -> (StatusCode, Json<serde_json::Value>) {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("{what}: {e}") })),
        )
    };

    let model_name = body
        .model
        .clone()
        .unwrap_or_else(|| "onnx-community/Qwen3.5-0.8B-ONNX-OPT".into());

    let mut answers = serde_json::Map::new();
    let mut total_in = 0usize;

    for (qid, qdef) in &body.questions {
        let t_start = std::time::Instant::now();
        let rendered = prompt::render_question(&body.state, qdef)
            .map_err(|e| err500("render_question", format_args!("{e}")))?;

        let encoding = st
            .tokenizer
            .encode(rendered.prompt.as_str(), false)
            .map_err(|e| err500("tokenize", format_args!("{e:?}")))?;

        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let seq_len = token_ids.len();
        total_in += seq_len;

        // Chunked-prefill split (prefix cache): the template tokens are
        // identical across requests with the same qdef, so their present_*
        // states can be cached and the suffix chunk reuses them instead of
        // re-prefilling the whole prompt. Oracle-validated (chunked ==
        // single-pass within 2.2e-5): position_ids absolute on all 3 rows,
        // attention_mask over the full length. Splitting the id sequence at
        // ANY point is correct; the cache key is the prefix ids themselves.
        let split_at: Option<usize> = if st.prefix_cache_on {
            prefix_split_len(&st.tokenizer, &rendered.split_prompt, &token_ids)
        } else {
            None
        };
        let cache_key = split_at.map(|p| token_ids[..p].to_vec());

        // 1. Run embed session (full id sequence once; rows sliced per chunk)
        let t_embed_start = std::time::Instant::now();
        let t_ids = Tensor::from_array((vec![1, seq_len], token_ids.into_boxed_slice()))
            .map_err(|e| err500("input_ids", format_args!("{e:?}")))?;

        let embed_data: Vec<f32> = {
            let mut embed_session = st.workers[worker_idx].embed.lock().unwrap();
            let embed_out = embed_session
                .run(ort::inputs!["input_ids" => t_ids])
                .map_err(|e| err500("embed_run", format_args!("{e:?}")))?;
            let (_shape, data) = embed_out[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| err500("embed_extract", format_args!("{e:?}")))?;
            data.to_vec()
        };
        let t_embed_dur = t_embed_start.elapsed();

        // 2. Prepare decoder inputs
        let t_prep_start = std::time::Instant::now();
        let (t_embeds, t_attn, t_pos) = match split_at {
            // Suffix chunk: embedding rows p..seq_len, attention over the FULL
            // length (cached template positions stay visible), absolute
            // positions p..seq_len on all 3 rows (oracle convention 'a').
            Some(p) => {
                let s = seq_len - p;
                let e = embed_data[p * 1024..].to_vec();
                let t_embeds = Tensor::from_array((vec![1, s, 1024], e.into_boxed_slice()))
                    .map_err(|e| err500("embed_tensor", format_args!("{e:?}")))?;
                let t_attn =
                    Tensor::from_array((vec![1, seq_len], vec![1i64; seq_len].into_boxed_slice()))
                        .map_err(|e| err500("attn_tensor", format_args!("{e:?}")))?;
                let mut pos_ids = Vec::with_capacity(3 * s);
                for _ in 0..3 {
                    for i in p..seq_len {
                        pos_ids.push(i as i64);
                    }
                }
                let t_pos = Tensor::from_array((vec![3, 1, s], pos_ids.into_boxed_slice()))
                    .map_err(|e| err500("pos_tensor", format_args!("{e:?}")))?;
                (t_embeds, t_attn, t_pos)
            }
            None => {
                let t_embeds = Tensor::from_array((vec![1, seq_len, 1024], embed_data.clone().into_boxed_slice()))
                    .map_err(|e| err500("embed_tensor", format_args!("{e:?}")))?;

                let t_attn =
                    Tensor::from_array((vec![1, seq_len], vec![1i64; seq_len].into_boxed_slice()))
                        .map_err(|e| err500("attn_tensor", format_args!("{e:?}")))?;

                let mut pos_ids = Vec::with_capacity(3 * seq_len);
                for _ in 0..3 {
                    for i in 0..seq_len {
                        pos_ids.push(i as i64);
                    }
                }
                let t_pos = Tensor::from_array((vec![3, 1, seq_len], pos_ids.into_boxed_slice()))
                    .map_err(|e| err500("pos_tensor", format_args!("{e:?}")))?;
                (t_embeds, t_attn, t_pos)
            }
        };

        let mut inputs: Vec<(&str, DynValue)> = Vec::with_capacity(55);
        inputs.push(("inputs_embeds", DynValue::from(t_embeds)));
        inputs.push(("attention_mask", DynValue::from(t_attn)));
        inputs.push(("position_ids", DynValue::from(t_pos)));
        if st.num_logits_input {
            // OPT export: score with the last position's row only.
            let t_num_logits = Tensor::from_array((Vec::<usize>::new(), vec![1i64].into_boxed_slice()))
                .map_err(|e| err500("num_logits_tensor", format_args!("{e:?}")))?;
            inputs.push(("num_logits_to_keep", DynValue::from(t_num_logits)));
        }

        let t_prep_dur = t_prep_start.elapsed();

        // 4. Run decoder session
        let t_dec_start = std::time::Instant::now();
        let (target_logits, run_desc): (Vec<f32>, String) = {
            let mut decoder_session = st.workers[worker_idx].decoder.lock().unwrap();

            // 3. Past states: cached template-chunk presents when chunking
            // (KV real), pre-created zeros for the single-pass path.
            let (pasts, run_desc) = match (split_at, cache_key) {
                (Some(p), Some(key)) => {
                    let hit = st.prefix_cache.lock().unwrap().get(&key).cloned();
                    match hit {
                        Some(cached) => (cached, format!("{p}/{seq_len} hit")),
                        None => {
                            // Cache miss: prefill the template chunk alone and
                            // keep its present_* states as the past inputs.
                            let e1 = embed_data[..p * 1024].to_vec();
                            let t_e1 = Tensor::from_array((vec![1, p, 1024], e1.into_boxed_slice()))
                                .map_err(|e| err500("embed_tensor", format_args!("{e:?}")))?;
                            let t_a1 =
                                Tensor::from_array((vec![1, p], vec![1i64; p].into_boxed_slice()))
                                    .map_err(|e| err500("attn_tensor", format_args!("{e:?}")))?;
                            let mut pos1 = Vec::with_capacity(3 * p);
                            for _ in 0..3 {
                                for i in 0..p {
                                    pos1.push(i as i64);
                                }
                            }
                            let t_p1 = Tensor::from_array((vec![3, 1, p], pos1.into_boxed_slice()))
                                .map_err(|e| err500("pos_tensor", format_args!("{e:?}")))?;
                            let mut in1: Vec<(&str, DynValue)> = Vec::with_capacity(55);
                            in1.push(("inputs_embeds", DynValue::from(t_e1)));
                            in1.push(("attention_mask", DynValue::from(t_a1)));
                            in1.push(("position_ids", DynValue::from(t_p1)));
                            if st.num_logits_input {
                                let t_n1 = Tensor::from_array((
                                    Vec::<usize>::new(),
                                    vec![1i64].into_boxed_slice(),
                                ))
                                .map_err(|e| err500("num_logits_tensor", format_args!("{e:?}")))?;
                                in1.push(("num_logits_to_keep", DynValue::from(t_n1)));
                            }
                            for (name, val) in &st.precreated_past {
                                in1.push((*name, val.clone()));
                            }
                            let out1 = decoder_session
                                .run(in1)
                                .map_err(|e| err500("decoder_run_prefix", format_args!("{e:?}")))?;
                            let mut cached: Vec<(&'static str, DynValue)> =
                                Vec::with_capacity(st.precreated_past.len());
                            for (name, _) in &st.precreated_past {
                                let pn = present_name(name);
                                let pv = out1.get(&pn).ok_or_else(|| {
                                    err500("prefix_present_missing", format_args!("{pn}"))
                                })?;
                                cached.push((*name, pv.clone()));
                            }
                            {
                                let mut cache = st.prefix_cache.lock().unwrap();
                                if cache.len() >= 16 {
                                    cache.clear();
                                }
                                cache.insert(key, cached.clone());
                            }
                            (cached, format!("{p}/{seq_len} miss"))
                        }
                    }
                }
                _ => (st.precreated_past.clone(), format!("{seq_len} full")),
            };
            for (name, val) in &pasts {
                inputs.push((*name, val.clone()));
            }
            let decoder_out = decoder_session
                .run(inputs)
                .map_err(|e| err500("decoder_run", format_args!("{e:?}")))?;

            let (logits_shape, logits_data) = decoder_out["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| err500("logits_extract", format_args!("{e:?}")))?;

            // Row selection is export-dependent: num_logits_to_keep=1 yields a
            // single logits row (OPT), while the standard export emits one row
            // per position and the next-token distribution lives in the last.
            let dims: &[i64] = logits_shape;
            let vocab = dims.last().copied().unwrap_or(0).max(0) as usize;
            let rows = if dims.len() >= 3 {
                (dims[dims.len() - 2]).max(1) as usize
            } else {
                1
            };
            let row_off = (rows - 1) * vocab;

            let mut targets = Vec::with_capacity(rendered.label_token_ids.len());
            for &tid in &rendered.label_token_ids {
                let idx = row_off + tid as usize;
                let logit = logits_data.get(idx).copied().unwrap_or(f32::NEG_INFINITY);
                targets.push(logit);
            }
            (targets, run_desc)
        };
        let t_dec_dur = t_dec_start.elapsed();
        let t_total = t_start.elapsed();

        eprintln!(
            "[w{worker_idx}:{qid}] total={:?} (embed={:?}, prep={:?}, decoder={:?}, tokens={seq_len}, chunk={run_desc})",
            t_total, t_embed_dur, t_prep_dur, t_dec_dur
        );

        // 5. Softmax & Decode
        let probs = prompt::softmax(&target_logits, st.temperature);
        let ans = prompt::decode(qdef, &rendered.keys, &probs);
        answers.insert(qid.clone(), ans);
    }

    Ok(OutBody {
        model: model_name,
        answers: serde_json::Value::Object(answers),
        usage: Usage {
            input_tokens: total_in,
            output_tokens: body.questions.len(),
        },
    })
}
