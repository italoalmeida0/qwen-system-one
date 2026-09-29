#!/usr/bin/env python3
"""update_onnx_weights.py — Injeta pesos fine-tunados no grafo ONNX Q4 existente.

Como a arquitetura Qwen 3.5 0.8B possui operadores customizados (LinearAttention com
Gated DeltaNet, CausalConvWithState, GatherBlockQuantized, MatMulNBits com INT4 block-32),
a exportação via Optimum/PyTorch tracing falha por falta de kernel oficial.

Este script utiliza o grafo ONNX base já validado pelo runtime Rust (qwen-serve) e
injeta diretamente os novos pesos do checkpoint PyTorch/safetensors:
  1. embed_tokens_q4.onnx + embed_tokens_q4.onnx_data
  2. decoder_model_merged_q4.onnx + decoder_model_merged_q4.onnx_data
  3. Copia tokenizer.json e configs

Execução:
    python tools/update_onnx_weights.py \
        --base-onnx-dir models \
        --safetensors /content/drive/MyDrive/qwen-system-one/runs/decider08_full/model \
        --out /content/drive/MyDrive/qwen-system-one/models_v1_1
"""
import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

# Compatibilidade ml_dtypes para bfloat16
try:
    import ml_dtypes
except ImportError:
    pass

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors import safe_open

# Tenta carregar o kernel C++ de quantização INT4 do ONNX Runtime
HAS_C_QUANT = False
try:
    from onnxruntime.capi._pybind_state import quantize_matmul_4bits
    HAS_C_QUANT = True
except Exception:
    pass


def quantize_block_int4_numpy(data_2d: np.ndarray, block_size: int = 32):
    """Fallback puro em NumPy para quantização assimétrica INT4 em blocos.
    
    data_2d: matriz float32 com shape [rows, cols] onde rows = K (dimensão de redução), cols = N.
    Retorna:
        packed: uint8 com shape [cols, k_blocks, 16]
        scales: float32 com shape [cols, k_blocks]
        zero_points: uint8 com shape [cols, (k_blocks + 1) // 2]
    """
    rows, cols = data_2d.shape
    k_blocks = (rows + block_size - 1) // block_size
    pad_len = k_blocks * block_size - rows
    if pad_len > 0:
        data_2d = np.pad(data_2d, ((0, pad_len), (0, 0)), mode="constant")

    # Reshape para [cols, k_blocks, block_size]
    # data_2d é [K, N], transpondo para [N, K] e reorganizando em blocos
    t = data_2d.T.reshape(cols, k_blocks, block_size)
    min_val = np.minimum(t.min(axis=2), 0.0)
    max_val = np.maximum(t.max(axis=2), 0.0)

    range_val = max_val - min_val
    scales = np.where(range_val == 0.0, 1.0, range_val / 15.0).astype(np.float32)
    
    # zp = round(-min_val / scale) clip(0, 15)
    zp_unpacked = np.where(range_val == 0.0, 8, np.round(-min_val / scales)).clip(0, 15).astype(np.uint8)
    
    # quant = round((x / scale) + zp) clip(0, 15)
    scales_exp = np.expand_dims(scales, axis=2)
    zp_exp = np.expand_dims(zp_unpacked, axis=2)
    q_unpacked = np.where(scales_exp == 0.0, 8, np.round(t / scales_exp) + zp_exp).clip(0, 15).astype(np.uint8)

    # Empacota quant [cols, k_blocks, 32] -> [cols, k_blocks, 16] (nibble baixo = índice par, nibble alto = ímpar)
    low_nibble = q_unpacked[:, :, 0::2] & 0x0F
    high_nibble = (q_unpacked[:, :, 1::2] & 0x0F) << 4
    packed = (low_nibble | high_nibble).astype(np.uint8)

    # Empacota zero points [cols, k_blocks] -> [cols, (k_blocks + 1) // 2]
    zp_cols = (k_blocks + 1) // 2
    zp_padded = zp_unpacked
    if k_blocks % 2 != 0:
        zp_padded = np.pad(zp_unpacked, ((0, 0), (0, 1)), mode="constant")
    zp_low = zp_padded[:, 0::2] & 0x0F
    zp_high = (zp_padded[:, 1::2] & 0x0F) << 4
    zero_points = (zp_low | zp_high).astype(np.uint8)[:, :zp_cols]

    return packed, scales, zero_points


def quantize_weight(w_fp32: np.ndarray, block_size: int = 32):
    """Quantiza matriz de pesos para formato MatMulNBits (N, K // 32, 16).
    
    w_fp32: shape [N, K] (saída x entrada, convenção PyTorch)
    Retorna: packed [N, K // 32, 16], scales [N, K // 32], zero_points [N, (K // 32 + 1) // 2]
    """
    # MatMul B em ONNX é [K, N]
    B = w_fp32.T
    rows, cols = B.shape
    k_blocks = (rows + block_size - 1) // block_size

    if HAS_C_QUANT:
        packed = np.zeros((cols, k_blocks, 16), dtype=np.uint8)
        zero_point = np.zeros((cols, (k_blocks + 1) // 2), dtype=np.uint8)
        scales = np.zeros((cols, k_blocks), dtype=np.float32)
        quantize_matmul_4bits(packed, B, scales, zero_point, block_size, cols, rows, False)
        return packed, scales, zero_point
    else:
        return quantize_block_int4_numpy(B, block_size)


class WeightReader:
    def __init__(self, safetensors_dir: Path):
        self.safetensors_dir = safetensors_dir
        self.files = list(safetensors_dir.glob("*.safetensors"))
        if not self.files:
            raise FileNotFoundError(f"Nenhum arquivo .safetensors encontrado em {safetensors_dir}")
        self.handles = [safe_open(str(f), framework="numpy") for f in self.files]
        self.key_map = {}
        for idx, h in enumerate(self.handles):
            for k in h.keys():
                self.key_map[k] = idx
        print(f"[safetensors] {len(self.key_map)} tensores carregados de {len(self.files)} arquivo(s).")

    def get_tensor(self, name: str) -> np.ndarray:
        # Tenta variações de prefixo
        candidates = [
            name,
            f"model.language_model.{name}",
            name.replace("model.", "model.language_model."),
            name.replace("model.language_model.", "model."),
            f"model.{name}",
        ]
        for c in candidates:
            if c in self.key_map:
                arr = self.handles[self.key_map[c]].get_tensor(c)
                if hasattr(arr, "astype"):
                    # Converte bfloat16 para float32
                    if arr.dtype == np.dtype("bfloat16") or str(arr.dtype) == "bfloat16":
                        arr = arr.astype(np.float32)
                return arr
        raise KeyError(f"Tensor não encontrado no safetensors: {name} (tentados: {candidates[:3]})")

    def has_tensor(self, name: str) -> bool:
        candidates = [
            name,
            f"model.language_model.{name}",
            name.replace("model.", "model.language_model."),
            name.replace("model.language_model.", "model."),
            f"model.{name}",
        ]
        return any(c in self.key_map for c in candidates)


def update_embed_tokens(base_onnx: Path, out_onnx: Path, reader: WeightReader):
    """Atualiza embed_tokens_q4.onnx com novos pesos de embedding."""
    print(f"\n[embed_tokens] Carregando {base_onnx.name}...")
    m = onnx.load(str(base_onnx), load_external_data=False)
    
    # Obtém novos pesos de embedding
    w_emb = reader.get_tensor("model.language_model.embed_tokens.weight")
    vocab_size, hidden_dim = w_emb.shape
    print(f"[embed_tokens] Novos pesos de embedding: shape {w_emb.shape}, dtype {w_emb.dtype}")

    t0 = time.time()
    packed, scales, zp = quantize_weight(w_emb, block_size=32)
    # embed_tokens quant tem shape [vocab_size, 512] (packed [N, K_blocks, 16] reshaped para 2D)
    packed_2d = packed.reshape(vocab_size, -1)
    print(f"[embed_tokens] Quantização 4-bit concluída em {time.time() - t0:.2f}s.")

    # Substitui os initializers
    init_map = {init.name: init for init in m.graph.initializer}
    
    new_inits = []
    for init in m.graph.initializer:
        if init.name == "model_embed_tokens_weight_quant":
            new_inits.append(numpy_helper.from_array(packed_2d, name=init.name))
        elif init.name == "model_embed_tokens_weight_scales":
            new_inits.append(numpy_helper.from_array(scales, name=init.name))
        elif init.name == "model_embed_tokens_weight_zp":
            new_inits.append(numpy_helper.from_array(zp, name=init.name))
        else:
            new_inits.append(init)

    m.graph.ClearField("initializer")
    m.graph.initializer.extend(new_inits)

    out_data = out_onnx.parent / f"{out_onnx.name}_data"
    if out_data.exists():
        out_data.unlink()
    if out_onnx.exists():
        out_onnx.unlink()

    print(f"[embed_tokens] Salvando {out_onnx.name} (+ {out_data.name})...")
    onnx.save_model(
        m,
        str(out_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{out_onnx.name}_data",
        size_threshold=1024,
    )
    print(f"[embed_tokens] OK! onnx: {out_onnx.stat().st_size / 1024:.1f} KB, data: {out_data.stat().st_size / 1024 / 1024:.1f} MB")


def update_decoder(base_onnx: Path, out_onnx: Path, reader: WeightReader):
    """Atualiza decoder_model_merged_q4.onnx com novos pesos de camadas e MatMuls."""
    print(f"\n[decoder] Carregando {base_onnx.name}...")
    m = onnx.load(str(base_onnx), load_external_data=False)

    # Cache de quantização para não re-quantizar pesos repetidos (ex: lm_head se tied com embed_tokens)
    quant_cache = {}

    def get_quantized(weight_name: str):
        if weight_name in quant_cache:
            return quant_cache[weight_name]
        w = reader.get_tensor(weight_name)
        packed, scales, zp = quantize_weight(w, block_size=32)
        quant_cache[weight_name] = (packed, scales, zp)
        return quant_cache[weight_name]

    total_inits = len(m.graph.initializer)
    updated_count = 0
    t0 = time.time()

    new_inits = []
    for idx, init in enumerate(m.graph.initializer):
        name = init.name
        
        # 1. Constantes auxiliares de grafo: mantém inalteradas
        if "constants" in name or name == "model.inv_freq":
            new_inits.append(init)
            continue

        # 2. Final norm
        if name == "model.layers.24.final_norm_layernorm.weight":
            w = reader.get_tensor("model.language_model.norm.weight")
            new_inits.append(numpy_helper.from_array(w.astype(np.float32), name=name))
            updated_count += 1
            continue

        # 3. Layernorms de entrada e pós-atenção
        m_ln = re.match(r"model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight", name)
        if m_ln:
            layer_idx, ln_type = m_ln.group(1), m_ln.group(2)
            st_key = f"model.language_model.layers.{layer_idx}.{ln_type}.weight"
            w = reader.get_tensor(st_key)
            new_inits.append(numpy_helper.from_array(w.astype(np.float32), name=name))
            updated_count += 1
            continue

        # 4. Pesos contínuos GDN (conv1d, A_neg_exp, dt_bias, norm)
        m_gdn = re.match(r"model\.layers\.(\d+)\.gdn\.(conv1d\.weight_3d|A_neg_exp|dt_bias|norm\.weight)", name)
        if m_gdn:
            layer_idx, field = m_gdn.group(1), m_gdn.group(2)
            if field == "conv1d.weight_3d":
                w = reader.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.conv1d.weight")
                new_inits.append(numpy_helper.from_array(w.astype(np.float32), name=name))
            elif field == "A_neg_exp":
                a_log = reader.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.A_log")
                a_neg_exp = -np.exp(a_log.astype(np.float32))
                new_inits.append(numpy_helper.from_array(a_neg_exp, name=name))
            elif field == "dt_bias":
                w = reader.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.dt_bias")
                new_inits.append(numpy_helper.from_array(w.astype(np.float32), name=name))
            elif field == "norm.weight":
                w = reader.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.norm.weight")
                new_inits.append(numpy_helper.from_array(w.astype(np.float32), name=name))
            updated_count += 1
            continue

        # 5. Layernorms de atenção padrão (q_norm, k_norm)
        m_attn_norm = re.match(r"model\.layers\.(\d+)\.attn\.(q_norm|k_norm)\.layernorm\.weight", name)
        if m_attn_norm:
            layer_idx, norm_type = m_attn_norm.group(1), m_attn_norm.group(2)
            st_key = f"model.language_model.layers.{layer_idx}.self_attn.{norm_type}.weight"
            w = reader.get_tensor(st_key)
            new_inits.append(numpy_helper.from_array(w.astype(np.float32), name=name))
            updated_count += 1
            continue

        # 6. MatMulNBits: GDN in_proj / out_proj
        m_mm_gdn = re.match(r"model_layers_(\d+)_gdn_(in_proj_[abqkvz]+|out_proj)_MatMul_(weight_quant|weight_scales|weight_zp)", name)
        if m_mm_gdn:
            layer_idx, proj_name, suffix = m_mm_gdn.group(1), m_mm_gdn.group(2), m_mm_gdn.group(3)
            st_key = f"model.language_model.layers.{layer_idx}.linear_attn.{proj_name}.weight"
            packed, scales, zp = get_quantized(st_key)
            if suffix == "weight_quant":
                new_inits.append(numpy_helper.from_array(packed, name=name))
            elif suffix == "weight_scales":
                new_inits.append(numpy_helper.from_array(scales, name=name))
            elif suffix == "weight_zp":
                new_inits.append(numpy_helper.from_array(zp, name=name))
            updated_count += 1
            continue

        # 7. MatMulNBits: Self-attention q, k, v, o proj
        m_mm_attn = re.match(r"model_layers_(\d+)_attn_([qkvo]_proj)_MatMul_(weight_quant|weight_scales|weight_zp)", name)
        if m_mm_attn:
            layer_idx, proj_name, suffix = m_mm_attn.group(1), m_mm_attn.group(2), m_mm_attn.group(3)
            st_key = f"model.language_model.layers.{layer_idx}.self_attn.{proj_name}.weight"
            packed, scales, zp = get_quantized(st_key)
            if suffix == "weight_quant":
                new_inits.append(numpy_helper.from_array(packed, name=name))
            elif suffix == "weight_scales":
                new_inits.append(numpy_helper.from_array(scales, name=name))
            elif suffix == "weight_zp":
                new_inits.append(numpy_helper.from_array(zp, name=name))
            updated_count += 1
            continue

        # 8. MatMulNBits: MLP gate, up, down proj
        m_mm_mlp = re.match(r"model_layers_(\d+)_mlp_(gate|up|down)_proj_MatMul_(weight_quant|weight_scales|weight_zp)", name)
        if m_mm_mlp:
            layer_idx, proj_name, suffix = m_mm_mlp.group(1), m_mm_mlp.group(2), m_mm_mlp.group(3)
            st_key = f"model.language_model.layers.{layer_idx}.mlp.{proj_name}_proj.weight"
            packed, scales, zp = get_quantized(st_key)
            if suffix == "weight_quant":
                new_inits.append(numpy_helper.from_array(packed, name=name))
            elif suffix == "weight_scales":
                new_inits.append(numpy_helper.from_array(scales, name=name))
            elif suffix == "weight_zp":
                new_inits.append(numpy_helper.from_array(zp, name=name))
            updated_count += 1
            continue

        # 9. MatMulNBits: lm_head
        if name.startswith("lm_head_MatMul_"):
            suffix = name.split("lm_head_MatMul_")[-1]
            if reader.has_tensor("lm_head.weight"):
                st_key = "lm_head.weight"
            else:
                st_key = "model.language_model.embed_tokens.weight"
            packed, scales, zp = get_quantized(st_key)
            if suffix == "weight_quant":
                new_inits.append(numpy_helper.from_array(packed, name=name))
            elif suffix == "weight_scales":
                new_inits.append(numpy_helper.from_array(scales, name=name))
            elif suffix == "weight_zp":
                new_inits.append(numpy_helper.from_array(zp, name=name))
            updated_count += 1
            continue

        # Se não casou com nenhuma regra conhecida, mantém o original com aviso
        print(f"[decoder] AVISO: initializer não mapeado mantido original: {name}")
        new_inits.append(init)

    print(f"[decoder] {updated_count}/{total_inits} initializers atualizados em {time.time() - t0:.2f}s.")

    m.graph.ClearField("initializer")
    m.graph.initializer.extend(new_inits)

    out_data = out_onnx.parent / f"{out_onnx.name}_data"
    if out_data.exists():
        out_data.unlink()
    if out_onnx.exists():
        out_onnx.unlink()

    print(f"[decoder] Salvando {out_onnx.name} (+ {out_data.name})...")
    onnx.save_model(
        m,
        str(out_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{out_onnx.name}_data",
        size_threshold=1024,
    )
    print(f"[decoder] OK! onnx: {out_onnx.stat().st_size / 1024:.1f} KB, data: {out_data.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Injeta pesos safetensors no ONNX Q4")
    parser.add_argument("--base-onnx-dir", default="models", help="Pasta com os modelos ONNX base de referência")
    parser.add_argument("--safetensors", required=True, help="Pasta contendo model.safetensors treinado")
    parser.add_argument("--out", required=True, help="Pasta de destino para o modelo ONNX Q4 final")
    args = parser.parse_args()

    base_dir = Path(args.base_onnx_dir)
    st_dir = Path(args.safetensors)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== INÍCIO DA ATUALIZAÇÃO ONNX Q4 ===")
    print(f"Base ONNX:       {base_dir.resolve()}")
    print(f"Safetensors:     {st_dir.resolve()}")
    print(f"Saída:           {out_dir.resolve()}")
    print(f"Aceleração C++:  {'Ativa (onnxruntime)' if HAS_C_QUANT else 'NumPy Vectorized'}")

    reader = WeightReader(st_dir)

    # 1. embed_tokens_q4
    base_embed = base_dir / "embed_tokens_q4.onnx"
    out_embed = out_dir / "embed_tokens_q4.onnx"
    if not base_embed.exists():
        raise FileNotFoundError(f"Arquivo base não encontrado: {base_embed}")
    update_embed_tokens(base_embed, out_embed, reader)

    # 2. decoder_model_merged_q4
    base_decoder = base_dir / "decoder_model_merged_q4.onnx"
    out_decoder = out_dir / "decoder_model_merged_q4.onnx"
    if not base_decoder.exists():
        raise FileNotFoundError(f"Arquivo base não encontrado: {base_decoder}")
    update_decoder(base_decoder, out_decoder, reader)

    # 3. Tokenizer e arquivos de configuração
    print(f"\n[copia] Copiando tokenizer.json e configurações...")
    for f_name in ["tokenizer.json", "tokenizer_config.json", "config.json", "chat_template.jinja"]:
        src = st_dir / f_name
        if not src.exists():
            src = base_dir / f_name
        if src.exists():
            dst = out_dir / f_name
            shutil.copy2(src, dst)
            print(f"[copia] {f_name} copiado ({dst.stat().st_size / 1024:.1f} KB)")

    print(f"\n[validacao] Verificando arquivos gerados...")
    required = [
        "embed_tokens_q4.onnx",
        "embed_tokens_q4.onnx_data",
        "decoder_model_merged_q4.onnx",
        "decoder_model_merged_q4.onnx_data",
        "tokenizer.json",
    ]
    for r in required:
        p = out_dir / r
        if not p.exists():
            raise RuntimeError(f"ERRO: Arquivo obrigatório não foi gerado: {p}")
        print(f"  [OK] {r} ({p.stat().st_size / 1024 / 1024:.2f} MB)")

    print(f"\n=== SUCESSO! Modelo ONNX Q4 pronto para o runtime Rust em: {out_dir} ===")


if __name__ == "__main__":
    main()
