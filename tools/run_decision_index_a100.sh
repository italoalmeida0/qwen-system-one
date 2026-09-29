#!/bin/bash
set -e

# ==============================================================================
# QWEN-SYSTEM-ONE v1.1.0: JEV DECISION INDEX BENCHMARK NA GPU A100 (COLAB)
# Executa as 120.000 requisições a ~200-400 req/s em ~5 a 10 minutos!
# ==============================================================================

echo "================================================================="
echo "   QWEN-SYSTEM-ONE v1.1.0 : JEV DECISION INDEX NA GPU A100       "
echo "================================================================="

WORKDIR="/content/decision_benchmark_a100"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 1. Dependências do servidor e do kit de benchmark
echo "[1/4] Instalando dependencias do servidor GPU e benchmark..."
pip install -q fastapi uvicorn

# 2. Iniciar o servidor GPU nativo na A100 (porta 8093)
echo "[2/4] Iniciando Servidor GPU A100 (qwen_gpu_serve.py) na porta 8093..."
pkill -f qwen_gpu_serve || true
python3 /content/drive/MyDrive/qwen-system-one/tools/qwen_gpu_serve.py > gpu_serve.log 2>&1 &

# Aguardar servidor GPU ficar online
for i in {1..30}; do
    if curl -s http://127.0.0.1:8093/health | grep -q "qwen-gpu-serve"; then
        echo "  -> Servidor GPU A100 pronto e respondendo em http://127.0.0.1:8093!"
        break
    fi
    sleep 1
done

# 3. Preparar o kit oficial decision-index
echo "[3/4] Preparando kit oficial decision-index..."
if [ ! -d "decision-index" ]; then
    git clone https://github.com/apolinario/decision-index.git
fi
cd decision-index
pip install -q -e ".[rebuild]"

export PYTHONUTF8=1

# 4. Reconstruir a suite de 120k se necessário
SUITE_WORK="/content/suite-work"
if [ ! -f "$SUITE_WORK/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz" ]; then
    echo "  -> Reconstruindo suite oficial 0.2 (download e congelamento das 42 bases)..."
    python3 -m decision_index suite rebuild --edition 0.2 --work "$SUITE_WORK"
else
    echo "  -> Suite de 120k questoes ja disponivel em $SUITE_WORK!"
fi

# 5. Executar a avaliação oficial contra o servidor GPU A100!
echo "[4/4] Executando benchmark oficial Jev Decision Index (120k requisicoes na A100)..."
python3 -m decision_index run \
    --edition 0.2.1 \
    --engine http \
    --option base_url=http://127.0.0.1:8093 \
    --option model=qwen-system-one \
    --rows "$SUITE_WORK/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz" \
    --out /content/runs/qwen-system-one-a100-v1.1

# Calcular e exibir os scores oficiais
echo ""
echo "Calculando Decision Index score oficial..."
python3 -m decision_index score \
    --edition 0.2.1 \
    --results /content/runs/qwen-system-one-a100-v1.1/results.jsonl \
    --out /content/runs/qwen-system-one-a100-v1.1

echo ""
echo "================================================================="
echo "   RESULTADO FINAL DO JEV DECISION INDEX (A100):                 "
echo "================================================================="
cat /content/runs/qwen-system-one-a100-v1.1/scores.json
