#!/bin/bash
# ==============================================================================
# run_phase3_distill_train.sh
# Pipeline Completo: Destilação MiMo-V2.6-Pro-RL -> Treino DPO A100 -> Benchmark
# ==============================================================================
set -e

echo "======================================================================"
echo "   QWEN-SYSTEM-ONE : FASE 3 - DESTILAÇÃO MIMO (1T MIT) & TREINO DPO    "
echo "======================================================================"

if [ -z "$MIMO_API_KEY" ]; then
    echo "[ERRO] Variavel MIMO_API_KEY nao definida! Execute:"
    echo "       export MIMO_API_KEY=\"seu-token-bearer\""
    exit 1
fi

BASE_DIR="/content/qwen-system-one"
DRIVE_DIR="/content/drive/MyDrive/qwen-system-one"
DATA_DIR="$DRIVE_DIR/data"
mkdir -p "$DATA_DIR"

cd "$BASE_DIR"
git pull origin main || true

echo "[*] Instalando bibliotecas necessarias para DPO na A100..."
pip install -q trl peft accelerate transformers

# 1. Executar a destilacao com MiMo-V2.6-Pro-RL
echo ""
echo "[*] Passo 1: Executando destilacao assincrona com MiMo-V2.6-Pro-RL..."
python3 tools/distill_mimo.py \
    --api-key "$MIMO_API_KEY" \
    --api-base "https://api.primalabs.ai/v1" \
    --model "primalabs-ai/MiMo-V2.6-Pro-RL" \
    --suite-rows "/content/suite-work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz" \
    --results "$DRIVE_DIR/results_benchmark_final.jsonl" \
    --out "$DATA_DIR/distill_mimo_v1_2.jsonl" \
    --concurrency 16 \
    --limit-samples 15000 \
    --max-context 8192

# 2. Executar o treino DPO na A100
echo ""
echo "[*] Passo 2: Executando treinamento DPO com LoRA na GPU A100..."
python3 tools/train_dpo_qwen.py \
    --model-path "$DRIVE_DIR/runs/decider08_full/model" \
    --data-path "$DATA_DIR/distill_mimo_v1_2.jsonl" \
    --output-dir "$DRIVE_DIR/runs/qwen-system-one-v1.2/model" \
    --epochs 2 \
    --batch-size 4 \
    --grad-accum 4 \
    --max-length 8192

# 3. Subir o servidor de inferência com o novo modelo v1.2
echo ""
echo "[*] Passo 3: Reiniciando servidor GPU com qwen-system-one-v1.2..."
fuser -k 8093/tcp 2>/dev/null || true
sleep 2

export MODEL_PATH="$DRIVE_DIR/runs/qwen-system-one-v1.2/model"
nohup /usr/bin/python3 tools/qwen_gpu_serve.py > /content/gpu_serve_v1_2.log 2>&1 &
SERVER_PID=$!
echo "  -> Servidor GPU iniciado (PID: $SERVER_PID). Aguardando warmup..."

for i in {1..30}; do
    if curl -s http://127.0.0.1:8093/health | grep -q "ok"; then
        echo "[+] Servidor GPU v1.2 online e pronto!"
        break
    fi
    sleep 2
done

# 4. Rodar o Benchmark Oficial (Fast Concurrent Runner)
echo ""
echo "[*] Passo 4: Executando Benchmark Oficial Decision Index (0.2.1)..."
RUN_DIR="/content/decision_benchmark/runs/qwen-system-one-v1.2"
mkdir -p "$RUN_DIR"

python3 tools/fast_decision_runner.py \
    --base-url "http://127.0.0.1:8093" \
    --rows "/content/suite-work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz" \
    --out "$RUN_DIR" \
    --concurrency 32 \
    --model "qwen-system-one"

# 5. Calcular a pontuação oficial
echo ""
echo "[*] Passo 5: Calculando score oficial do Decision Index..."
python3 tools/compute_scores.py

echo ""
echo "======================================================================"
echo "   FASE 3 FINALIZADA COM SUCESSO! MODELO v1.2.0 AVALIADO!            "
echo "======================================================================"
