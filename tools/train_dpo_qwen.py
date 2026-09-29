#!/usr/bin/env python3
"""
train_dpo_qwen.py - Treinamento DPO / SFT ultrarrápido para Qwen System-One v1.2.0 na A100.
Destila o conhecimento do MiMo-V2.6-Pro-RL (1T MoE) utilizando LoRA em bfloat16 + SDPA.
Garante o limite estrito de 8.192 tokens (8k) e exporta o modelo mesclado pronto para servir.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

def load_jsonl_dataset(path):
    prompts = []
    chosens = []
    rejecteds = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line)
                    prompts.append(rec["prompt"])
                    chosens.append(rec["chosen"])
                    rejecteds.append(rec["rejected"])
                except Exception:
                    pass
                    
    print(f"[*] Carregados {len(prompts):,} pares de treinamento do dataset.")
    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds
    })

def main():
    parser = argparse.ArgumentParser(description="Train Qwen System-One v1.2.0 via DPO on A100")
    parser.add_argument("--model-path", default="/content/drive/MyDrive/qwen-system-one/runs/decider08_full/model", help="Base model path")
    parser.add_argument("--data-path", default="/content/drive/MyDrive/qwen-system-one/data/distill_mimo_v1_2.jsonl", help="Distilled DPO data path")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/qwen-system-one/runs/qwen-system-one-v1.2/model", help="Final merged model output")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature beta")
    parser.add_argument("--batch-size", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=8192, help="Strict context max length (8k)")
    parser.add_argument("--max-prompt-length", type=int, default=8180, help="Max prompt length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    args = parser.parse_args()

    print("=" * 70)
    print("   QWEN-SYSTEM-ONE v1.2.0 : DPO TRAINING NA NVIDIA A100   ")
    print(f"   Modelo Base  : {args.model_path}")
    print(f"   Dataset      : {args.data_path}")
    print(f"   Max Contexto : {args.max_length} tokens (8k)")
    print(f"   Output Final : {args.output_dir}")
    print("=" * 70)

    if not os.path.exists(args.model_path):
        print(f"[ERRO] Modelo base nao encontrado em: {args.model_path}")
        sys.exit(1)

    if not os.path.exists(args.data_path):
        print(f"[ERRO] Dataset de destilacao nao encontrado em: {args.data_path}")
        sys.exit(1)

    # 1. Carregar Tokenizer
    print("[*] Carregando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 2. Carregar Dataset
    dataset = load_jsonl_dataset(args.data_path)

    # 3. Carregar Modelo com bfloat16 e FlashAttention/SDPA
    print("[*] Carregando Modelo em bfloat16 na GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()

    # 4. Configurar LoRA Adapter
    print(f"[*] Configurando LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 5. Configurar DPO Trainer
    temp_ckpt_dir = "/content/tmp_dpo_ckpt"
    dpo_config = DPOConfig(
        output_dir=temp_ckpt_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        gradient_checkpointing=True,
        remove_unused_columns=False
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None, # TRL usa o modelo base congelado automaticamente quando ref_model=None com PEFT
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config
    )

    print("[*] Iniciando treinamento DPO...")
    trainer.train()
    print("[✓] Treinamento DPO concluído com sucesso!")

    # 6. Mesclar pesos LoRA no modelo base
    print("[*] Mesclando pesos LoRA ao modelo base...")
    merged_model = trainer.model.merge_and_unload()

    # 7. Salvar modelo final autônomo
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Salvando modelo v1.2.0 final em: {out_dir}...")
    merged_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print("=" * 70)
    print(f"[✓] Qwen System-One v1.2.0 exportado com sucesso em: {out_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
