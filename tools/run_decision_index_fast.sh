#!/bin/bash
set -e

# ==============================================================================
# QWEN-SYSTEM-ONE v1.1.0: HIGH-SPEED JEV DECISION INDEX BENCHMARK (A100)
# Executa as 120.000 requisições a alta velocidade com intra-row batching e 32 workers!
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

# 2. Obter scripts atualizados diretamente para o disco NVMe local (evita lag de cache do Drive)
TOOLS_DIR="/content/qwen_tools"
mkdir -p "$TOOLS_DIR"
echo "  -> Baixando scripts otimizados (batching v2)..."
curl -s -L https://raw.githubusercontent.com/italoalmeida0/qwen-system-one/main/tools/qwen_gpu_serve.py -o "$TOOLS_DIR/qwen_gpu_serve.py"
curl -s -L https://raw.githubusercontent.com/italoalmeida0/qwen-system-one/main/tools/fast_decision_runner.py -o "$TOOLS_DIR/fast_decision_runner.py"

# 3. Encerrar qualquer processo antigo e liberar porta 8093
echo "[2/4] Reiniciando Servidor GPU A100 otimizado (porta 8093)..."
pkill -9 -f qwen_gpu_serve || true
fuser -k 8093/tcp || true
sleep 2

python3 "$TOOLS_DIR/qwen_gpu_serve.py" > /content/gpu_serve.log 2>&1 &

# Aguardar servidor GPU responder no health check
echo "  -> Aguardando carregamento do modelo na VRAM da A100..."
for i in {1..60}; do
    if curl -s http://127.0.0.1:8093/health | grep -q "qwen-gpu-serve"; then
        echo "  -> Servidor GPU A100 online e respondendo em http://127.0.0.1:8093!"
        break
    fi
    sleep 1
done

# 4. Garantir kit oficial decision-index
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

# 5. Executar o Fast Concurrent Runner com 32 threads paralelas!
OUT_DIR="/content/decision_benchmark/runs/qwen-system-one-v1.1"
mkdir -p "$OUT_DIR"

echo "[4/4] Executando benchmark acelerado com batching na A100..."
python3 "$TOOLS_DIR/fast_decision_runner.py" \
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
