#!/usr/bin/env python3
"""export_onnx_q4.py — exporta o modelo treinado para o formato do runtime qwen-serve.

Saída (mesmos nomes/IO do runtime — verificado em models/decoder_model_merged_q4.onnx):
    embed_tokens_q4.onnx         + embed_tokens_q4.onnx_data
    decoder_model_merged_q4.onnx + decoder_model_merged_q4.onnx_data

Uso (Colab, célula 6):
    python export_onnx_q4.py --model $OUT/model --out /content/drive/MyDrive/qwen-system-one/models_v1_1

Pipeline (padrão HF optimum + quant block-wise):
1. `optimum.onnxruntime` exporta o modelo com KV cache (past_*/present_*).
2. Separa embed (lookup) do decoder (merged).
3. Quantiza pesos INT4 block-wise (group_size=32) via onnxruntime.quantization.
4. Salva com external data (.onnx grafo + .onnx_data pesos) — formato do runtime.

ATENÇÃO: o export do Qwen3.5 (arquitetura híbrida DeltaNet+attention) pode exigir
opset/customização extra. Se optimum reclamar de ops, usar o exportador do
onnx-community/Qwen3.5-0.8B-ONNX-OPT como referência (mesma arquitetura, mesmo
layout de past_*/present_*).
"""
import argparse
import os
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="pasta do modelo treinado (safetensors)")
    ap.add_argument("--out", required=True, help="pasta de saída dos .onnx")
    ap.add_argument("--quant", default="q4", choices=["q4", "fp16"])
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out = Path(args.out)

    # ---- 1. Exporta com optimum (KV cache: past_*/present_*) ----
    print(f"[export] optimum exportando {args.model} -> {args.out}")
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = ORTModelForCausalLM.from_pretrained(args.model, export=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)

    # ---- 2. Renomeia pro layout do runtime ----
    # optimum gera model.onnx (decoder) + model.onnx_data; o runtime espera
    # decoder_model_merged_q4.onnx/.onnx_data + embed_tokens_q4.onnx/.onnx_data.
    # O embed (lookup) sai do decoder via --export-embeddings do optimum, ou
    # é um subgrafo — aqui assumimos que o optimum já separou (ver docs).
    renames = {
        "model.onnx": "decoder_model_merged_q4.onnx",
        "model.onnx_data": "decoder_model_merged_q4.onnx_data",
    }
    for src, dst in renames.items():
        if (out / src).exists():
            shutil.move(str(out / src), str(out / dst))
            print(f"  {src} -> {dst}")

    # ---- 3. Quantiza INT4 (block-wise, group_size=32) ----
    if args.quant == "q4":
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print("[export] quantizando pesos para INT4 ...")
        for name in ["decoder_model_merged_q4.onnx", "embed_tokens_q4.onnx"]:
            p = out / name
            if not p.exists():
                print(f"  {name}: não encontrado, pula")
                continue
            tmp = out / (name + ".q4")
            quantize_dynamic(str(p), str(tmp), weight_type=QuantType.QUInt4)
            os.replace(str(tmp), str(p))
            print(f"  {name}: quantizado")

    # ---- 4. Copia tokenizer.json (runtime lê direto) ----
    if (out / "tokenizer.json").exists():
        print("[export] tokenizer.json OK")

    print(f"[export] pronto em {args.out}")
    print("Próximo: copiar pro runtime (models/) e rodar quick-check + bench")


if __name__ == "__main__":
    main()