#!/bin/bash
set -e

# ==============================================================================
# QWEN-SYSTEM-ONE v1.1.0: OFICIAL JEV DECISION INDEX BENCHMARK RUNNER (COLAB)
# Garante 100% de paridade com o runtime nativo C++/Rust (qwen-serve)
# ==============================================================================

echo "================================================================="
echo "   QWEN-SYSTEM-ONE : JEV DECISION INDEX BENCHMARK (COLAB)        "
echo "================================================================="

WORKDIR="/content/decision_benchmark"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 1. Download do binário nativo do nosso runtime (Linux x64)
echo "[1/6] Baixando binário oficial nativo linux-x64 do qwen-serve..."
if [ ! -f "qwen-serve" ]; then
    wget -q --show-progress https://github.com/italoalmeida0/qwen-system-one/releases/download/v1.1.0/qwen-serve-linux-x64 -O qwen-serve
    chmod +x qwen-serve
fi

# 2. Localização dos pesos v1.1.0 treinados
echo "[2/6] Verificando pesos do modelo v1.1.0..."
MODEL_DIR="/content/models_v1_1"

if [ -d "/content/drive/MyDrive/qwen-system-one/models_v1_1" ]; then
    echo "  -> Encontrado no Google Drive: /content/drive/MyDrive/qwen-system-one/models_v1_1"
    MODEL_DIR="/content/drive/MyDrive/qwen-system-one/models_v1_1"
elif [ ! -f "$MODEL_DIR/decoder_model_merged_q4.onnx" ]; then
    echo "  -> Baixando arquivo do pacote v1.1.0 do GitHub..."
    mkdir -p "$MODEL_DIR"
    wget -q --show-progress https://github.com/italoalmeida0/qwen-system-one/releases/download/v1.1.0/qwen-0.8b-q4.model -O qwen-0.8b-q4.model
    tar -xf qwen-0.8b-q4.model -C "$MODEL_DIR"
fi

# 3. Validação dos Hashes SHA256 para garantir que é 100% o modelo treinado
echo "[3/6] Validando integridade dos pesos treinados..."
python3 -c "
import hashlib, os, sys
p = '$MODEL_DIR/decoder_model_merged_q4.onnx_data'
expected = 'ce5fa68c23aac7d6594f7a4a2b29c640c3a7aecf457b6bcabbdc0d5eb66bbf70'
h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
if h != expected:
    print(f'ERRO: Hash de {p} ({h}) nao bate com v1.1.0 ({expected})!')
    sys.exit(1)
print('  -> HASH CONFIRMADO: 100% modelo treinado decider08_full v1.1.0!')
"

# 4. Iniciar nosso servidor nativo em background
echo "[4/6] Iniciando qwen-serve nativo em http://127.0.0.1:8093..."
pkill -f qwen-serve || true
./qwen-serve \
    --model-dir "$MODEL_DIR" \
    --port 8093 \
    --cache-size 0 \
    --temperature 1.14 \
    --threads 0 \
    --prefix-cache on > qwen-serve.log 2>&1 &

SERVER_PID=$!

# Aguardar servidor ficar online
for i in {1..30}; do
    if curl -s http://127.0.0.1:8093/health | grep -q "qwen-serve"; then
        echo "  -> qwen-serve pronto e respondendo!"
        break
    fi
    sleep 1
done

# 5. Instalar suite oficial decision-index
echo "[5/6] Preparando kit oficial decision-index..."
if [ ! -d "decision-index" ]; then
    git clone https://github.com/apolinario/decision-index.git
fi
pip install -q -e "decision-index[rebuild]"

export PYTHONUTF8=1

# Reconstruir suite congelada de 120k se ainda não reconstruída
if [ ! -f "suite-work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz" ]; then
    echo "  -> Reconstruindo suite oficial 0.2 (downloads das fontes fixadas)..."
    python3 -m decision_index suite rebuild --edition 0.2 --work suite-work
fi

# 6. Rodar avaliação oficial contra o nosso runtime nativo!
echo "[6/6] Executando benchmark oficial Jev Decision Index (120k requisições)..."
python3 -m decision_index run \
    --edition 0.2.1 \
    --engine http \
    --option base_url=http://127.0.0.1:8093 \
    --option model=qwen-system-one \
    --rows suite-work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --out runs/qwen-system-one-v1.1

# Calcular pontuação oficial do índice
echo "Calculando Decision Index score..."
python3 -m decision_index score \
    --edition 0.2.1 \
    --results runs/qwen-system-one-v1.1/results.jsonl \
    --out runs/qwen-system-one-v1.1

echo "================================================================="
echo "   RESULTADO FINAL DO JEV DECISION INDEX:                        "
echo "================================================================="
cat runs/qwen-system-one-v1.1/scores.json
