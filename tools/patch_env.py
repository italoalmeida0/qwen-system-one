#!/usr/bin/env python3
"""patch_env.py — aplica patches de compatibilidade no Transformers 5 e Optimum."""
import glob
import os
import sysconfig


def main():
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
                    print(f"[patch] get_parameter_dtype injetado em: {p}")
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
                    print(f"[patch] _CAN_RECORD_REGISTRY injetado em: {p}")
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

    print("[patch] Ambiente compatibilizado com sucesso!")


if __name__ == "__main__":
    main()
