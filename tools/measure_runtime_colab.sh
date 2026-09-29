#!/bin/bash
set -e

# ==============================================================================
# QWEN-SYSTEM-ONE v1.1.0: BENCHMARK DE PERFORMANCE DO RUNTIME NATIVO (COLAB)
# Executa exatamente os mesmos testes de estresse e vazão do GitHub Actions CI
# ==============================================================================

echo "================================================================="
echo "   QWEN-SYSTEM-ONE : BENCHMARK DO RUNTIME NATIVO NO GOOGLE COLAB "
echo "================================================================="

WORKDIR="/content/runtime_bench"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 1. Informações de Hardware da Instância Colab
echo "[1/5] Informações do Ambiente Colab:"
lscpu | grep "Model name\|CPU(s):\|Thread(s) per core:" || true
free -h | grep "Mem:" || true
echo ""

# 2. Obter o binário oficial nativo linux-x64
echo "[2/5] Baixando binário oficial nativo linux-x64 do qwen-serve..."
if [ ! -f "qwen-serve" ]; then
    wget -q --show-progress https://github.com/italoalmeida0/qwen-system-one/releases/download/v1.1.0/qwen-serve-linux-x64 -O qwen-serve
    chmod +x qwen-serve
fi

# 3. Localização e validação dos pesos v1.1.0 treinados
echo "[3/5] Localizando e validando pesos v1.1.0..."
MODEL_DIR="/content/models_v1_1"

if [ -d "/content/drive/MyDrive/qwen-system-one/models_v1_1" ]; then
    echo "  -> Encontrado no Google Drive: /content/drive/MyDrive/qwen-system-one/models_v1_1"
    MODEL_DIR="/content/drive/MyDrive/qwen-system-one/models_v1_1"
elif [ ! -f "$MODEL_DIR/decoder_model_merged_q4.onnx" ]; then
    echo "  -> Baixando pacote v1.1.0 do GitHub Release..."
    mkdir -p "$MODEL_DIR"
    wget -q --show-progress https://github.com/italoalmeida0/qwen-system-one/releases/download/v1.1.0/qwen-0.8b-q4.model -O qwen-0.8b-q4.model
    tar -xf qwen-0.8b-q4.model -C "$MODEL_DIR"
fi

# Validação do hash SHA256 dos pesos
python3 -c "
import hashlib, sys
p = '$MODEL_DIR/decoder_model_merged_q4.onnx_data'
expected = 'ce5fa68c23aac7d6594f7a4a2b29c640c3a7aecf457b6bcabbdc0d5eb66bbf70'
h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
if h != expected:
    print(f'ERRO: Hash de {p} ({h}) nao bate com v1.1.0 ({expected})!')
    sys.exit(1)
print('  -> HASH SHA256 OK: 100% modelo treinado decider08_full v1.1.0!')
"

# 4. Clonar scripts de benchmark oficiais do repo
if [ ! -f "quick-check.js" ]; then
    echo "  -> Baixando scripts de teste do GitHub Actions..."
    curl -fsSL https://raw.githubusercontent.com/italoalmeida0/qwen-system-one/main/tools/quick-check.js -o quick-check.js
    curl -fsSL https://raw.githubusercontent.com/italoalmeida0/qwen-system-one/main/tools/bench-concurrent.js -o bench-concurrent.js
fi

# 5. Executar os testes idênticos aos do GitHub Actions CI
echo ""
echo "================================================================="
echo "   [TESTE 1/2] TRIAGEM MULTILÍNGUE (10 CASOS: PT, EN, ES)        "
echo "================================================================="
node quick-check.js --binary ./qwen-serve --model "$MODEL_DIR" --port 8094

echo ""
echo "================================================================="
echo "   [TESTE 2/2] VAZÃO CONCORRENTE REAL (c=1, c=2, c=4, c=8)       "
echo "   (Cache-busted: mede inferência real de cada requisição)       "
echo "================================================================="
node bench-concurrent.js --binary ./qwen-serve --model "$MODEL_DIR" --port 8096 --levels 1,2,4,8 --requests 16

echo ""
echo "================================================================="
echo "   BENCHMARK CONCLUÍDO COM SUCESSO!                             "
echo "================================================================="
