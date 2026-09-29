#!/usr/bin/env python3
"""export_onnx_q4.py — exporta o modelo treinado pro formato do runtime qwen-serve.

Caminho: `optimum-cli` (padrão HF, mesmo usado pelo onnx-community/Qwen3.5-0.8B-ONNX-OPT).

Saída (mesmo layout do repo de referência):
    embed_tokens_q4.onnx         + embed_tokens_q4.onnx_data
    decoder_model_merged_q4.onnx + decoder_model_merged_q4.onnx_data
    (vision_encoder_q4.onnx ignorado — nosso modelo é texto-only)

Uso (Colab, célula 6):
    python export_onnx_q4.py --model $OUT/model --out $GROOT/models_v1_1

O optimum-cli gera as variantes (fp32, fp16, q4, q4f16) automaticamente com
`--optimize O2`. Aqui mantemos só as q4 (formato do runtime).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="pasta do modelo treinado (safetensors)")
    ap.add_argument("--out", required=True, help="pasta de saída dos .onnx")
    ap.add_argument("--task", default="image-text-to-text", help="task do optimum (mesma do repo de referência)")
    ap.add_argument("--optimize", default="O2", help="nível de otimização/quant do optimum")
    args = ap.parse_args()

    model = Path(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Exporta com optimum-cli (gera embed_tokens + decoder_model_merged + vision_encoder) ----
    cmd = [
        sys.executable, "-m", "optimum", "onnxruntime", "export", "onnx",
        "--model", str(model),
        "--task", args.task,
        "--optimize", args.optimize,
        "--trust-remote-code",
        str(out),
    ]
    print(f"[export] rodando: {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("[export] optimum-cli falhou (arquitetura híbrida DeltaNet pode exigir opset extra).")
        print("[export] fallback: usar o exportador do onnx-community/Qwen3.5-0.8B-ONNX-OPT como referência.")
        sys.exit(1)

    # ---- 2. Mantém só as variantes q4 (formato do runtime) ----
    keep = {"embed_tokens_q4.onnx", "embed_tokens_q4.onnx_data",
            "decoder_model_merged_q4.onnx", "decoder_model_merged_q4.onnx_data",
            "tokenizer.json", "tokenizer_config.json", "config.json"}
    for p in sorted(out.iterdir()):
        if p.name not in keep:
            # remove fp32/fp16/q4f16/quantized + vision_encoder (nosso modelo é texto)
            if p.name.startswith(("embed_tokens", "decoder_model_merged", "vision_encoder")) or p.suffix in (".onnx", ".onnx_data"):
                if p.name not in keep:
                    print(f"  removendo {p.name}")
                    p.unlink()

    # ---- 3. Verifica os 2 arquivos do runtime ----
    ok = True
    for f in ["embed_tokens_q4.onnx", "embed_tokens_q4.onnx_data",
              "decoder_model_merged_q4.onnx", "decoder_model_merged_q4.onnx_data"]:
        p = out / f
        if p.exists():
            print(f"  OK  {f} ({p.stat().st_size/1024/1024:.1f}MB)")
        else:
            print(f"  FALTA {f}")
            ok = False

    if ok:
        print(f"[export] pronto em {out}")
        print("Próximo: copiar os 4 arquivos pro runtime (models/) e rodar quick-check + bench")
    else:
        print("[export] incompleto — verifique acima")
        sys.exit(1)


if __name__ == "__main__":
    main()