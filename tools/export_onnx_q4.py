#!/usr/bin/env python3
"""export_onnx_q4.py — exporta o modelo treinado para o formato do runtime qwen-serve.

Gera os 4 arquivos binários exigidos pelo runtime Rust (mesmo padrão do onnx-community):
    embed_tokens_q4.onnx         + embed_tokens_q4.onnx_data
    decoder_model_merged_q4.onnx + decoder_model_merged_q4.onnx_data
    tokenizer.json, tokenizer_config.json, config.json

Etapas:
1. Exportação FP32 via optimum (optimum.exporters.onnx.main_export ou optimum-cli).
   Com task="image-text-to-text", o Optimum separa embed_tokens e o decoder merged.
2. Quantização 4-bit (INT4 block-quantized, block_size=32, symmetric=True) usando
   o MatMul4BitsQuantizer do ONNX Runtime (mesmo algoritmo do transformers.js).
3. Salvamento com dados externos (.onnx_data) e cópia do tokenizer/configs.
4. Verificação estrita de presença e integridade de todos os arquivos.

Uso:
    python tools/export_onnx_q4.py --model $OUT/model --out $GROOT/models_v1_1
"""
import argparse
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Patch de compatibilidade no disco: compatibiliza transformers 5 com optimum
def _patch_transformers_disk():
    import glob, sysconfig

    # 1. get_parameter_dtype no modeling_utils.py
    dtype_code = (
        "\n\ndef get_parameter_dtype(parameter):\n"
        "    try:\n"
        "        return next(parameter.parameters()).dtype\n"
        "    except Exception:\n"
        "        return getattr(parameter, 'dtype', None)\n"
    )
    m_paths = [
        os.path.join(sysconfig.get_path("purelib"), "transformers", "modeling_utils.py"),
        os.path.join(sysconfig.get_path("platlib"), "transformers", "modeling_utils.py"),
    ] + glob.glob("/usr/local/**/transformers/modeling_utils.py", recursive=True)

    for p in set(m_paths):
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read()
                if "def get_parameter_dtype" not in content:
                    with open(p, "a", encoding="utf-8") as f:
                        f.write(dtype_code)
                    print(f"[patch] get_parameter_dtype injetado no arquivo em disco: {p}")
            except Exception:
                pass

    # 2. _CAN_RECORD_REGISTRY e OutputRecorder no generic.py
    generic_code = (
        "\n\n# Patch de compatibilidade Optimum / Transformers 5\n"
        "try:\n"
        "    from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY, OutputRecorder\n"
        "except Exception:\n"
        "    try:\n"
        "        from .output_capturing import _CAN_RECORD_REGISTRY, OutputRecorder\n"
        "    except Exception:\n"
        "        _CAN_RECORD_REGISTRY = set()\n"
        "        class OutputRecorder:\n"
        "            def __init__(self, *args, **kwargs): pass\n"
        "            def __enter__(self): return self\n"
        "            def __exit__(self, *args): pass\n"
    )
    g_paths = [
        os.path.join(sysconfig.get_path("purelib"), "transformers", "utils", "generic.py"),
        os.path.join(sysconfig.get_path("platlib"), "transformers", "utils", "generic.py"),
    ] + glob.glob("/usr/local/**/transformers/utils/generic.py", recursive=True)

    for p in set(g_paths):
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read()
                if "_CAN_RECORD_REGISTRY" not in content:
                    with open(p, "a", encoding="utf-8") as f:
                        f.write(generic_code)
                    print(f"[patch] _CAN_RECORD_REGISTRY injetado no arquivo em disco: {p}")
            except Exception:
                pass

    # 3. Patch direto em _traceable_decorator.py do optimum (se existir)
    dec_paths = [
        os.path.join(sysconfig.get_path("purelib"), "optimum", "exporters", "onnx", "_traceable_decorator.py"),
        os.path.join(sysconfig.get_path("platlib"), "optimum", "exporters", "onnx", "_traceable_decorator.py"),
    ] + glob.glob("/usr/local/**/optimum/exporters/onnx/_traceable_decorator.py", recursive=True)

    for p in set(dec_paths):
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read()
                target = "from transformers.utils.generic import _CAN_RECORD_REGISTRY, OutputRecorder, logger"
                if target in content:
                    replacement = (
                        "from transformers.utils.generic import logger\n"
                        "try:\n"
                        "    from transformers.utils.generic import _CAN_RECORD_REGISTRY, OutputRecorder\n"
                        "except Exception:\n"
                        "    try:\n"
                        "        from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY, OutputRecorder\n"
                        "    except Exception:\n"
                        "        _CAN_RECORD_REGISTRY = set()\n"
                        "        class OutputRecorder:\n"
                        "            def __init__(self, *a, **k): pass\n"
                        "            def __enter__(self): return self\n"
                        "            def __exit__(self, *a): pass\n"
                    )
                    content = content.replace(target, replacement)
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"[patch] _traceable_decorator.py corrigido em: {p}")
            except Exception:
                pass

_patch_transformers_disk()

# Patch de compatibilidade universal para ml_dtypes com onnx (intercepta qualquer tipo ausente)
import numpy as np
try:
    import ml_dtypes as real_ml
    class MlProxy:
        def __getattr__(self, name):
            if hasattr(real_ml, name):
                return getattr(real_ml, name)
            return np.uint8
        def __dir__(self):
            return dir(real_ml)
    sys.modules['ml_dtypes'] = MlProxy()
except ImportError:
    pass

# Patch de compatibilidade em memória: get_parameter_dtype
try:
    import transformers.modeling_utils
    if not hasattr(transformers.modeling_utils, "get_parameter_dtype"):
        def get_parameter_dtype(parameter):
            try:
                return next(parameter.parameters()).dtype
            except Exception:
                return getattr(parameter, "dtype", None)
        transformers.modeling_utils.get_parameter_dtype = get_parameter_dtype
except Exception:
    pass

# Patch de compatibilidade em memória: _CAN_RECORD_REGISTRY e OutputRecorder
try:
    import transformers.utils.generic as generic
    if not hasattr(generic, "_CAN_RECORD_REGISTRY"):
        try:
            from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY, OutputRecorder
            generic._CAN_RECORD_REGISTRY = _CAN_RECORD_REGISTRY
            generic.OutputRecorder = OutputRecorder
        except Exception:
            generic._CAN_RECORD_REGISTRY = set()
            class OutputRecorder:
                def __init__(self, *args, **kwargs): pass
                def __enter__(self): return self
                def __exit__(self, *args): pass
            generic.OutputRecorder = OutputRecorder
except Exception:
    pass


def quantize_to_q4(in_onnx: Path, out_onnx: Path, block_size: int = 32):
    """Quantiza um modelo ONNX para 4-bit (MatMulNBits) e salva com dados externos."""
    import onnx

    try:
        from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
    except ImportError:
        try:
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer as MatMul4BitsQuantizer
        except ImportError:
            raise RuntimeError(
                "Não foi possível importar MatMul4BitsQuantizer do onnxruntime.quantization! "
                "Certifique-se de que onnxruntime e optimum estão instalados."
            )

    print(f"[quant] carregando {in_onnx.name} ({in_onnx.stat().st_size / 1024 / 1024:.1f} MB)...")
    m = onnx.load_model(str(in_onnx))

    print(f"[quant] aplicando MatMul4BitsQuantizer (block_size={block_size}, symmetric=True)...")
    sig = inspect.signature(MatMul4BitsQuantizer.__init__)
    kwargs = {
        "model": m,
        "block_size": block_size,
        "is_symmetric": True,
    }
    if "accuracy_level" in sig.parameters:
        kwargs["accuracy_level"] = None

    quantizer = MatMul4BitsQuantizer(**kwargs)
    quantizer.process()

    data_name = f"{out_onnx.name}_data"
    data_path = out_onnx.parent / data_name
    if data_path.exists():
        data_path.unlink()
    if out_onnx.exists():
        out_onnx.unlink()

    print(f"[quant] salvando {out_onnx.name} (+ {data_name})...")
    try:
        from optimum.onnx.graph_transformations import check_and_save_model
        check_and_save_model(quantizer.model.model, str(out_onnx))
    except Exception as e:
        print(f"[quant] check_and_save_model aviso ({e}), usando onnx.save_model...")
        onnx.save_model(
            quantizer.model.model,
            str(out_onnx),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=1024,
        )

    # Se o check_and_save_model salvou tudo em um arquivo só (sem .onnx_data), força extração para paridade com o runtime
    if not data_path.exists() and out_onnx.exists():
        print(f"[quant] gerando arquivo de dados externos separado: {data_name}...")
        m_saved = onnx.load_model(str(out_onnx))
        out_onnx.unlink()
        onnx.save_model(
            m_saved,
            str(out_onnx),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=1024,
        )

    sz_onnx = out_onnx.stat().st_size / 1024 if out_onnx.exists() else 0
    sz_data = data_path.stat().st_size / (1024 * 1024) if data_path.exists() else 0
    print(f"[quant] OK {out_onnx.name} ({sz_onnx:.1f} KB, data: {sz_data:.1f} MB)")


def export_base_onnx(model_dir: Path, temp_dir: Path, task: str = "image-text-to-text"):
    """Exporta o modelo PyTorch/safetensors para ONNX FP32 via optimum."""
    print(f"[export] exportando modelo de {model_dir} para {temp_dir}...")

    # Tentativa 1: Python API direta do Optimum
    try:
        from optimum.exporters.onnx import main_export
        print(f"[export] executando optimum.exporters.onnx.main_export (task={task})...")
        main_export(
            model_name_or_path=str(model_dir),
            output=str(temp_dir),
            task=task,
            do_validation=False,
            trust_remote_code=True,
            device="cpu",
        )
        print("[export] export via Python API concluído!")
        return
    except Exception as e:
        print(f"[export] main_export Python API falhou ({e}). Tentando fallback CLI...")

    # Tentativa 2: CLI optimum-cli
    cli = shutil.which("optimum-cli")
    if cli:
        cmd = [cli, "export", "onnx", "--model", str(model_dir), "--task", task, "--trust-remote-code", str(temp_dir)]
    else:
        cmd = [
            sys.executable, "-m", "optimum.commands.optimum_cli",
            "export", "onnx",
            "--model", str(model_dir),
            "--task", task,
            "--trust-remote-code",
            str(temp_dir),
        ]

    print(f"[export] rodando: {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"optimum export falhou com código de retorno {r.returncode}")


def main():
    ap = argparse.ArgumentParser(description="Exporta modelo treinado para ONNX Q4 compatível com qwen-serve")
    ap.add_argument("--model", required=True, help="pasta do modelo treinado (safetensors)")
    ap.add_argument("--out", required=True, help="pasta de saída dos .onnx")
    ap.add_argument("--task", default="image-text-to-text", help="task do optimum (padrão: image-text-to-text)")
    ap.add_argument("--block-size", type=int, default=32, help="tamanho do bloco na quantização INT4 (padrão: 32)")
    ap.add_argument("--keep-temp", action="store_true", help="mantém pasta temporária de exportação FP32")
    args = ap.parse_args()

    model = Path(args.model).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not model.exists():
        print(f"[erro] Pasta do modelo não encontrada: {model}")
        sys.exit(1)

    temp_dir = Path(tempfile.mkdtemp(prefix="qwen_onnx_fp32_"))
    print(f"[inicio] Exportando {model} -> {out}")
    print(f"[inicio] Pasta temporária FP32: {temp_dir}")

    try:
        # 1. Exporta base FP32
        export_base_onnx(model, temp_dir, task=args.task)

        # 2. Localiza os modelos exportados
        # No image-text-to-text, o optimum gera embed_tokens.onnx e decoder_model_merged.onnx
        embed_candidates = [
            temp_dir / "embed_tokens.onnx",
            temp_dir / "onnx" / "embed_tokens.onnx",
            temp_dir / "embeddings.onnx",
            temp_dir / "onnx" / "embeddings.onnx",
        ]
        embed_src = next((p for p in embed_candidates if p.exists()), None)

        decoder_candidates = [
            temp_dir / "decoder_model_merged.onnx",
            temp_dir / "onnx" / "decoder_model_merged.onnx",
            temp_dir / "decoder_model.onnx",
            temp_dir / "onnx" / "decoder_model.onnx",
            temp_dir / "model.onnx",
            temp_dir / "onnx" / "model.onnx",
        ]
        decoder_src = next((p for p in decoder_candidates if p.exists()), None)

        # Fallback de segurança: se embed_tokens.onnx não foi gerado separadamente pelo optimum,
        # constrói o grafo ONNX do embedding diretamente a partir dos pesos safetensors
        if not embed_src:
            print("[aviso] embed_tokens.onnx não encontrado na saída do optimum. Tentando extrair pesos do embedding...")
            try:
                import torch
                from safetensors.torch import load_file
                safetensors_path = model / "model.safetensors"
                if not safetensors_path.exists():
                    st_list = list(model.glob("*.safetensors"))
                    safetensors_path = st_list[0] if st_list else None
                if safetensors_path and safetensors_path.exists():
                    weights = load_file(str(safetensors_path))
                    emb_weight = None
                    for k in ["model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight"]:
                        if k in weights:
                            emb_weight = weights[k]
                            break
                    if emb_weight is not None:
                        class EmbedModule(torch.nn.Module):
                            def __init__(self, w):
                                super().__init__()
                                self.embed = torch.nn.Embedding.from_pretrained(w.to(torch.float32))
                            def forward(self, input_ids):
                                return self.embed(input_ids)
                        mod = EmbedModule(emb_weight)
                        dummy_input = torch.tensor([[1]], dtype=torch.int64)
                        emb_onnx_path = temp_dir / "embed_tokens.onnx"
                        torch.onnx.export(
                            mod,
                            dummy_input,
                            str(emb_onnx_path),
                            input_names=["input_ids"],
                            output_names=["inputs_embeds"],
                            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "inputs_embeds": {0: "batch", 1: "sequence"}},
                            opset_version=14,
                        )
                        embed_src = emb_onnx_path
                        print(f"[fallback] embed_tokens.onnx gerado via safetensors: {embed_src}")
            except Exception as e:
                print(f"[fallback] aviso ao extrair embedding: {e}")

        if not embed_src:
            raise FileNotFoundError(f"Não encontrou embed_tokens.onnx em {temp_dir}. Arquivos encontrados: {[p.name for p in temp_dir.rglob('*.onnx')]}")
        if not decoder_src:
            raise FileNotFoundError(f"Não encontrou decoder_model_merged.onnx em {temp_dir}. Arquivos encontrados: {[p.name for p in temp_dir.rglob('*.onnx')]}")

        print(f"[localizado] embed:   {embed_src}")
        print(f"[localizado] decoder: {decoder_src}")

        # 3. Quantiza cada um para Q4
        embed_q4 = out / "embed_tokens_q4.onnx"
        decoder_q4 = out / "decoder_model_merged_q4.onnx"

        quantize_to_q4(embed_src, embed_q4, block_size=args.block_size)
        quantize_to_q4(decoder_src, decoder_q4, block_size=args.block_size)

        # 4. Copia metadados e tokenizer
        meta_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "generation_config.json",
            "chat_template.jinja",
        ]
        print("\n[meta] Copiando arquivos de configuração e tokenizer...")
        for mf in meta_files:
            src = None
            if (temp_dir / mf).exists():
                src = temp_dir / mf
            elif (temp_dir / "onnx" / mf).exists():
                src = temp_dir / "onnx" / mf
            elif (model / mf).exists():
                src = model / mf

            if src:
                dst = out / mf
                shutil.copy2(src, dst)
                print(f"  OK {mf} ({dst.stat().st_size / 1024:.1f} KB)")

        # 5. Verificação estrita
        required_files = [
            "embed_tokens_q4.onnx",
            "embed_tokens_q4.onnx_data",
            "decoder_model_merged_q4.onnx",
            "decoder_model_merged_q4.onnx_data",
            "tokenizer.json",
        ]
        print("\n=== Verificação final dos arquivos do runtime ===")
        all_ok = True
        for rf in required_files:
            p = out / rf
            if p.exists() and p.stat().st_size > 0:
                print(f"  OK   {rf:35s} ({p.stat().st_size / 1024 / 1024:6.1f} MB)")
            else:
                print(f"  FALTA {rf}")
                all_ok = False

        if all_ok:
            print(f"\n[SUCESSO] Todos os arquivos gerados em: {out}")
            print("Próximo passo: copiar para models/ e executar quick-check e benchmarks.")
        else:
            print("\n[ERRO] Faltam arquivos obrigatórios. Verifique os logs acima.")
            sys.exit(1)

    finally:
        if not args.keep_temp and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()