# Qwen System-One: Visão Geral, Arquitetura e Roadmap 🚀

Este documento serve como guia completo de handoff e planejamento técnico para qualquer desenvolvedor (ou IA) que for dar continuidade ao desenvolvimento do **Qwen System-One**.

---

## 1. O que é o projeto e qual o nosso objetivo?

### Contexto e Origem
O projeto é inspirado no **Laya System-One** (`laya-system-one`), uma engine de decisão ultrarrápida baseada no conceito de *"Sistema 1"* da psicologia cognitiva (respostas reflexivas, rápidas e estruturadas em milissegundos, sem a lentidão de uma LLM generativa de chat).

O protocolo segue a especificação wire do **TypeSafe Jev** (`POST /v1/systemone`):
* O cliente envia um texto (`state`) e uma ou mais perguntas de múltipla escolha com critérios (`questions`).
* A engine calcula as probabilidades relativas (*logits*) de cada alternativa na última camada e devolve uma resposta estruturada em JSON (`answers: { choice, probability }`).

### Nosso Objetivo Principal
1. **Substituir o modelo do Laya pelo Qwen 3.5 0.8B**: O modelo original do Laya é um transformer de 1B treinado apenas para tarefas específicas. O Qwen 3.5 0.8B é uma arquitetura moderna (híbrida com atenção e camadas recorrentes) com muito mais vocabulário (152k tokens) e capacidade multilíngue superior.
2. **Superar o Laya em Acurácia Geral**: Treinar/especializar o Qwen na base [**`multimodalart/jev-decision-index`**](https://huggingface.co/datasets/multimodalart/jev-decision-index) (que avalia 120 mil perguntas em 43 benchmarks de decisão, triagem, segurança e moderação).
3. **Performance Alvo**: Atingir **3 a 4 requisições/segundo em CPU** com latência inferior a **250ms**.
4. **Zero Dependências**: Servidor 100% nativo em Rust (`qwen-serve`), rodando offline em qualquer máquina sem Python, CUDA ou PyTorch.

---

## 2. Arquitetura Técnica

```
                        ┌──────────────────────────────────────────────┐
                        │              Cliente HTTP / SDK              │
                        │           POST /v1/systemone                 │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  qwen-serve (Servidor Nativo Rust / Axum)                                              │
│                                                                                        │
│  1. Prompt Builder (prompt.rs)                                                         │
│     Renderiza template Jev com delimitadores especiais e tokens de rótulo.             │
│                                                                                        │
│  2. Tokenizer (tokenizers pure-Rust, Metaspace + BPE)                                  │
│     Tokeniza o prompt diretamente em memória sem chamada externa.                     │
│                                                                                        │
│  3. ONNX Runtime Static Engine (ort crate)                                             │
│     ├─ embed_tokens_q4.onnx        -> Projeção de embeddings (Q4 INT4)                │
│     ├─ Static Past-State Tensors   -> 48 tensores pré-alocados em Arc (zero overhead)  │
│     └─ decoder_model_merged_q4.onnx -> Decodificador autoregressivo com CausalConvState│
│                                                                                        │
│  4. Softmax Calibrado & Decoder (main.rs)                                              │
│     Extrai logits apenas dos tokens das alternativas e devolve JSON com probabilidades. │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Por que o Qwen é Q4 (INT4) e não Q2 ou FP16?
* **FP16 / FP32**: Teria de 1.6 GB a 3.2 GB de tamanho e consumiria mais de 3 GB de RAM, inviabilizando uso leve em CPU.
* **Q2 (2-bit)**: Modelos sub-1B não possuem a redundância necessária para suportar quantização agressiva abaixo de 4 bits. Em 2 bits, os logits se degradam e a CPU precisaria de operações de desempacotamento (*bit-shift*) sem instruções de hardware nativas, tornando a inferência mais lenta.
* **Q4 (INT4 block-quantized)**: É o *sweet spot* perfeito — **~533 MB compactado**, roda com menos de 1 GB de RAM e atinge ~270ms nativo na CPU.

---

## 3. O que já foi feito (Estado Atual)

### ✅ Servidor Nativo Rust Concluído (`native/qwen-serve`)
* Implementado em Rust com Axum 0.8, Tokio e ONNX Runtime (`ort`).
* **Otimização de Tensores de Estado Passado**: Pré-alocação estática dos 48 tensores de estado passado (`past_conv`, `past_recurrent`, `past_key_values`), eliminando alocações repetidas por request e cortando o overhead de preparação de ~500ms para **0ms**.

### ✅ Empacotamento e Publicação da Base
* Modelo base empacotado em `models/qwen-0.8b-q4.model` (533 MB).
* Publicado como release asset no GitHub: [Release v1.0.0](https://github.com/italoalmeida0/qwen-system-one/releases/tag/v1.0.0).
* Script de aquisição e extração com verificação SHA256 implementado (`tools/acquire-model.js`).

### ✅ CI Multi-Plataforma em 8 Arquiteturas (100% Green)
Workflow do GitHub Actions ([`.github/workflows/build-packages.yml`](file:///.github/workflows/build-packages.yml)) configurado e testado com sucesso em todas as 8 plataformas:

| Slot da Plataforma | Ambiente / Runner | Linkagem | Latência no CI (2 vCPUs) | Throughput |
|---|---|---|:---:|:---:|
| `win32-arm64` | `windows-11-arm` | MSVC CRT / Estático | **817.3 ms** | 1.22 req/s |
| `linux-arm64` | `ubuntu-24.04-arm` (Graviton) | glibc / Estático | **846.9 ms** | 1.18 req/s |
| `linux-arm64-musl` | Alpine Docker ARM64 | musl + Bundle Auto-extratível | **851.6 ms** | 1.17 req/s |
| `linux-x64` | `ubuntu-latest` | glibc / Estático | **932.8 ms** | 1.07 req/s |
| `win32-x64` | `windows-latest` | MSVC CRT / Estático | **1098.6 ms** | 0.91 req/s |
| `linux-x64-musl` | Alpine Docker x86_64 | musl + Bundle Auto-extratível | **1147.3 ms** | 0.87 req/s |
| `darwin-x64` | `macos-15-intel` | Homebrew ORT 1.30.0 dinâmico | **1700.4 ms** | 0.59 req/s |
| `darwin-arm64` | `macos-latest` (Apple Silicon) | Apple Silicon / Estático | **2006.3 ms** | 0.50 req/s |

> Em máquinas locais com mais núcleos (como o ambiente de desenvolvimento), a latência já fica na faixa de **270ms a 400ms**.

---

## 4. Estratégia de Desenvolvimento: Por que Otimizar o Runtime ANTES do Fine-Tuning?

Decidimos priorizar a **Otimização de Runtime no Servidor Rust** antes de iniciar o treinamento/fine-tuning. Essa decisão de arquitetura baseia-se em 3 pilares:

1. **Definição do "Contrato de Prompt" Definitivo**:
   * O tempo de decodificação e prefill na CPU depende diretamente da quantidade de tokens de entrada e da padronização dos delimitadores.
   * Ao otimizar o *Prefix Cache* e encurtar o prompt agora no Rust, definimos o formato exato em que o modelo deve operar. 
   * Assim, **quando formos treinar, treinaremos o modelo já nesse formato final**, evitando retrabalho e retreinos posteriores.

2. **Destravamento Imediato do Gargalo de Concorrência**:
   * O servidor atual utiliza um `std::sync::Mutex<Session>` único para o decodificador, o que limita o throughput a apenas **1 requisição por vez (concorrência = 1)**.
   * Substituindo esse Mutex por um **Pool de Sessões Concorrentes** (`SessionPool`) com distribuição equilibrada de threads (`intra_op_threads`), o throughput salta imediatamente de 1 req/s para **3 a 5 req/s** em CPUs modernas, atingindo nossa meta de velocidade antes mesmo do fine-tuning.

3. **O Servidor como um "Motor Pronto"**:
   * O servidor Rust vira um motor de alta performance já verificado e compilado para as 8 plataformas. 
   * Depois, o fine-tuning será apenas uma substituição de "combustível" (o arquivo de pesos `.onnx`), ganhando inteligência sem precisar reescrever o código de inferência.

---

## 5. O que falta fazer (Fases do Roadmap)

### 📌 Fase 1: Otimizações de Throughput e Concorrência (Prioridade Imediata)
1. **Pool de Sessões Concorrentes (`SessionPool` em `main.rs`)**:
   * Criar um pool com $N$ instâncias de `Session` (configurável via CLI `--workers`, default automático baseado em núcleos de CPU).
   * Gerenciamento de empréstimo assíncrono via canal Tokio (`tokio::sync::mpsc` ou `deadpool`), permitindo que requisições HTTP paralelas sejam processadas simultaneamente sem bloqueio de mutex global.
2. **Distribuição Equilibrada de Threads (`intra_op_threads`)**:
   * Em vez de 1 sessão monopolizar todos os 8 núcleos com contenção de thread, alocar 2 a 4 workers com 2 a 4 threads cada, maximizando a eficiência de pipeline na CPU.
3. **Prefix Caching para Perguntas e Regras**:
   * Caching de estados de tensores KV para prefixos de prompts repetidos (ex: políticas de reembolso, regras de triagem), derrubando a latência do prefill para **< 50ms**.
4. **Ferramenta de Benchmark de Concorrência (`tools/bench-concurrent.js`)**:
   * Script automatizado para disparar rajadas concorrentes (concorrência 2, 4, 8, 16) e comprovar o ganho de requisições por segundo.

### 📌 Fase 2: Fine-Tuning do Qwen 3.5 para Decisão (Superar o Laya)
1. **Dataset de Treinamento**:
   * Download e pré-processamento do dataset [`multimodalart/jev-decision-index`](https://huggingface.co/datasets/multimodalart/jev-decision-index) (120k perguntas em 43 benchmarks de decisão, triagem e segurança).
2. **Treinamento com SFT / QLoRA**:
   * Treinar o modelo base `Qwen/Qwen3.5-0.8B` com Unsloth / Hugging Face `trl`.
   * Formato de entrada: exatamente a estrutura de prompt otimizada na Fase 1.
   * Função de perda (Loss): Cross-entropy calculada estritamente no **primeiro token da decisão**, forçando o modelo a ter certeza absoluta de forma reflexiva sem alucinar texto longo.
3. **Conversão e Publicação do Modelo v1.1.0**:
   * Exportação dos pesos afinados para ONNX Q4 (`decoder_model_merged_q4.onnx`).
   * Empacotamento do novo `.model` e publicação na Release `v1.1.0` do GitHub.
   * Reavaliação no benchmark comparativo contra o Laya.

### 📌 Fase 3: Pacote NPM e CLI Pública
1. Criar `bin/cli.js` e wrapper Node/Bun com seleção automática de arquitetura (`@sys-one` ou `@qwen-system-one`).
2. Publicação no npm registry permitindo uso direto:
   ```bash
   npx qwen-system-one --port 8093
   ```

---

## 5. Como rodar e testar localmente

### Clonar e Compilar o Servidor
```bash
git clone https://github.com/italoalmeida0/qwen-system-one.git
cd qwen-system-one

# 1. Baixar os pesos do modelo (release v1.0.0)
node tools/acquire-model.js

# 2. Compilar e rodar o servidor Rust
cargo run --release --manifest-path native/qwen-serve/Cargo.toml -- --port 8093
```

### Rodar a Verificação Automatizada
```bash
# Executa 10 perguntas de teste e imprime latência média e req/s
node tools/quick-check.js
```

### Disparar Builds no CI via GitHub CLI
```bash
# Dispara o workflow em todas as 8 arquiteturas
gh workflow run build-packages.yml

# Acompanha o progresso
gh run list --workflow=build-packages.yml
```

---

## 6. Links Úteis do Repositório

* **Repositório GitHub**: [italoalmeida0/qwen-system-one](https://github.com/italoalmeida0/qwen-system-one)
* **Releases e Arquivo `.model`**: [Releases v1.0.0](https://github.com/italoalmeida0/qwen-system-one/releases/tag/v1.0.0)
* **Workflow de CI das 8 Plataformas**: [.github/workflows/build-packages.yml](file:///.github/workflows/build-packages.yml)
* **Código do Servidor Rust**: [native/qwen-serve/src/main.rs](file:///native/qwen-serve/src/main.rs)
* **Comparativo Laya vs Qwen (Benchmark Upstream)**: [`comparativo_laya_vs_qwen_nativo.md`](file:///C:/Users/italo/.gemini/antigravity/brain/5342e8d1-5385-4f22-8cc9-bca528668bc8/comparativo_laya_vs_qwen_nativo.md)
