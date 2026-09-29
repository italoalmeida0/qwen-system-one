#!/bin/bash
set -e

# ==============================================================================
# QWEN-SYSTEM-ONE v1.1.0: HIGH-SPEED JEV DECISION INDEX BENCHMARK (A100)
# Executa as 120.000 requisições a ~250-450 req/s em ~5 a 8 minutos!
# Concorrência otimizada com 32 workers e retomada automática sem perda de dados.
# ==============================================================================

echo "================================================================="
echo "   QWEN-SYSTEM-ONE v1.1.0 : SUPER FAST JEV BENCHMARK NA A100     "
echo "================================================================="

WORKDIR="/content/decision_benchmark_a100"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 1. Dependências do servidor e do kit de benchmark
echo "[1/4] Instalando dependencias de alta performance..."
pip install -q fastapi uvicorn urllib3

# 2. Reiniciar Servidor GPU A100 com as novas otimizacoes (logits_to_keep=1 + noul fix)
echo "[2/4] Iniciando Servidor GPU A100 otimizado (porta 8093)..."
pkill -f qwen_gpu_serve || true
sleep 1
python3 /content/drive/MyDrive/qwen-system-one/tools/qwen_gpu_serve.py > /content/gpu_serve.log 2>&1 &

# Aguardar servidor GPU responder no health check
echo "  -> Aguardando inicializacao do modelo na VRAM da A100..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8093/health | grep -q "qwen-gpu-serve"; then
        echo "  -> Servidor GPU A100 pronto e respondendo em http://127.0.0.1:8093!"
        break
    fi
    sleep 1
done

# 3. Garantir kit oficial decision-index
echo "[3/4] Verificando kit oficial decision-index..."
if [ ! -d "decision-index" ]; then
    git clone https://github.com/apolinario/decision-index.git
fi
cd decision-index
pip install -q -e ".[rebuild]"

export PYTHONUTF8=1

# Verificar suite reconstruida
SUITE_WORK="/content/suite-work"
ROWS="$SUITE_WORK/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz"
if [ ! -f "$ROWS" ]; then
    echo "  -> Reconstruindo suite oficial 0.2..."
    python3 -m decision_index suite rebuild --edition 0.2 --work "$SUITE_WORK"
else
    echo "  -> Suite oficial de 120k questoes pronta em $SUITE_WORK!"
fi

# 4. Executar o Fast Concurrent Runner com 32 threads paralelas!
OUT_DIR="/content/decision_benchmark/runs/qwen-system-one-v1.1"
mkdir -p "$OUT_DIR"

echo "[4/4] Executando benchmark acelerado (32 threads concorrentes na A100)..."
python3 /content/drive/MyDrive/qwen-system-one/tools/fast_decision_runner.py \
    --base-url "http://127.0.0.1:8093" \
    --rows "$ROWS" \
    --out "$OUT_DIR" \
    --concurrency 32 \
    --edition 0.2.1 \
    --model "qwen-system-one"

echo ""
echo "================================================================="
echo "   BENCHMARK FINALIZADO COM SUCESSO!                             "
echo "================================================================="
if [ -f "$OUT_DIR/scores.json" ]; then
    cat "$OUT_DIR/scores.json"
fi
