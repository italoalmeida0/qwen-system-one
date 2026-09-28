//! qwen-serve: self-contained native HTTP server for Qwen 3.5 / JPT models.
//!
//! Embeds: ONNX Runtime (statically linked via `ort`), Hugging Face tokenizers
//! (pure Rust), Axum HTTP server.
//!
//! Wire protocol: 100% compatible with TypeSafe Jev (/v1/systemone).

mod prompt;
mod schema;

use std::path::PathBuf;
use std::sync::Arc;

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
    /// Intra-op threads (0 = ORT default)
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Bearer API key (optional; also LAYA_API_KEY)
    #[arg(long)]
    api_key: Option<String>,
    /// Temperature for softmax (default 1.140 for Qwen 3.5 / JPT)
    #[arg(long, default_value_t = 1.140)]
    temperature: f32,
}

struct AppState {
    embed_session: std::sync::Mutex<Session>,
    decoder_session: std::sync::Mutex<Session>,
    tokenizer: Tokenizer,
    precreated_past: Vec<(&'static str, DynValue)>,
    temperature: f32,
    api_key: Option<String>,
}

fn oe<T, E: std::fmt::Debug>(r: Result<T, E>) -> anyhow::Result<T> {
    r.map_err(|e| anyhow!("{e:?}"))
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let api_key = args.api_key.or_else(|| std::env::var("LAYA_API_KEY").ok());

    // ---- ONNX Runtime init ----
    ort::init().with_name("qwen-serve").commit();

    let opt_level = GraphOptimizationLevel::Level3;

    let embed_path = args.model_dir.join("embed_tokens_q4.onnx");
    info!("loading embed session from {}", embed_path.display());
    let mut embed_builder = oe(Session::builder()?.with_optimization_level(opt_level))?;
    if args.threads > 0 {
        embed_builder = oe(embed_builder.with_intra_threads(args.threads))?;
    }
    let embed_session = oe(embed_builder.commit_from_file(&embed_path))?;

    let decoder_path = args.model_dir.join("decoder_model_merged_q4.onnx");
    info!("loading decoder session from {}", decoder_path.display());
    let mut decoder_builder = oe(Session::builder()?.with_optimization_level(opt_level))?;
    if args.threads > 0 {
        decoder_builder = oe(decoder_builder.with_intra_threads(args.threads))?;
    }
    let decoder_session = oe(decoder_builder.commit_from_file(&decoder_path))?;

    // Pre-create all static past states once
    let mut precreated_past: Vec<(&'static str, DynValue)> = Vec::new();
    for inp in decoder_session.inputs().iter() {
        let name = inp.name().to_string();
        if name.starts_with("past_conv") {
            let data = vec![0.0f32; 1 * 6144 * 3];
            let t = oe(Tensor::from_array((vec![1, 6144, 3], data.into_boxed_slice())))?;
            precreated_past.push((name.leak() as &'static str, DynValue::from(t)));
        } else if name.starts_with("past_recurrent") {
            let data = vec![0.0f32; 1 * 16 * 128 * 128];
            let t = oe(Tensor::from_array((vec![1, 16, 128, 128], data.into_boxed_slice())))?;
            precreated_past.push((name.leak() as &'static str, DynValue::from(t)));
        } else if name.starts_with("past_key_values") {
            let data = vec![0.0f32; 0];
            let t = oe(Tensor::from_array((vec![1, 2, 0, 256], data.into_boxed_slice())))?;
            precreated_past.push((name.leak() as &'static str, DynValue::from(t)));
        }
    }
    info!("pre-created {} past state tensors", precreated_past.len());

    // ---- Tokenizer ----
    let tok_path = args.model_dir.join("tokenizer.json");
    info!("loading tokenizer from {}", tok_path.display());
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("{e:?}"))?;

    let state = Arc::new(AppState {
        embed_session: std::sync::Mutex::new(embed_session),
        decoder_session: std::sync::Mutex::new(decoder_session),
        tokenizer,
        precreated_past,
        temperature: args.temperature,
        api_key,
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

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "backend": "qwen-serve",
        "model": "onnx-community/Qwen3.5-0.8B-ONNX-OPT",
        "runtime": "native-rust-onnx"
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

    let st2 = st.clone();
    let out = tokio::task::spawn_blocking(move || infer_all(&st2, &body))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("task: {e}") })),
            )
        })??;

    Ok(Json(out))
}

fn infer_all(st: &AppState, body: &InBody) -> Result<OutBody, (StatusCode, Json<serde_json::Value>)> {
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

        // 1. Run embed session
        let t_embed_start = std::time::Instant::now();
        let t_ids = Tensor::from_array((vec![1, seq_len], token_ids.into_boxed_slice()))
            .map_err(|e| err500("input_ids", format_args!("{e:?}")))?;

        let embed_data: Vec<f32> = {
            let mut embed_session = st.embed_session.lock().unwrap();
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
        let t_embeds = Tensor::from_array((vec![1, seq_len, 1024], embed_data.into_boxed_slice()))
            .map_err(|e| err500("embed_tensor", format_args!("{e:?}")))?;

        let t_attn = Tensor::from_array((vec![1, seq_len], vec![1i64; seq_len].into_boxed_slice()))
            .map_err(|e| err500("attn_tensor", format_args!("{e:?}")))?;

        let mut pos_ids = Vec::with_capacity(3 * seq_len);
        for _ in 0..3 {
            for i in 0..seq_len {
                pos_ids.push(i as i64);
            }
        }
        let t_pos = Tensor::from_array((vec![3, 1, seq_len], pos_ids.into_boxed_slice()))
            .map_err(|e| err500("pos_tensor", format_args!("{e:?}")))?;

        let t_num_logits = Tensor::from_array((Vec::<usize>::new(), vec![1i64].into_boxed_slice()))
            .map_err(|e| err500("num_logits_tensor", format_args!("{e:?}")))?;

        let mut inputs: Vec<(&str, DynValue)> = Vec::with_capacity(55);
        inputs.push(("inputs_embeds", DynValue::from(t_embeds)));
        inputs.push(("attention_mask", DynValue::from(t_attn)));
        inputs.push(("position_ids", DynValue::from(t_pos)));
        inputs.push(("num_logits_to_keep", DynValue::from(t_num_logits)));

        // 3. Reuse pre-created past state tensors (zero cost Arc clones)
        for (name, val) in &st.precreated_past {
            inputs.push((*name, val.clone()));
        }
        let t_prep_dur = t_prep_start.elapsed();

        // 4. Run decoder session
        let t_dec_start = std::time::Instant::now();
        let target_logits: Vec<f32> = {
            let mut decoder_session = st.decoder_session.lock().unwrap();
            let decoder_out = decoder_session
                .run(inputs)
                .map_err(|e| err500("decoder_run", format_args!("{e:?}")))?;

            let (_shape, logits_data) = decoder_out["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| err500("logits_extract", format_args!("{e:?}")))?;

            let mut targets = Vec::with_capacity(rendered.label_token_ids.len());
            for &tid in &rendered.label_token_ids {
                let idx = tid as usize;
                let logit = logits_data.get(idx).copied().unwrap_or(f32::NEG_INFINITY);
                targets.push(logit);
            }
            targets
        };
        let t_dec_dur = t_dec_start.elapsed();
        let t_total = t_start.elapsed();

        eprintln!(
            "[{qid}] total={:?} (embed={:?}, prep={:?}, decoder={:?}, tokens={seq_len})",
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
