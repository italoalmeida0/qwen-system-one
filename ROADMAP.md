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

## 4. O que falta fazer (Próximos Passos Prioritários)

Para dar continuidade, dividimos as tarefas em duas frentes: **Treinamento** e **Otimização de Runtime**.

### Fase 1: Fine-Tuning do Qwen 3.5 para Decisão (Superar o Laya)
O modelo atual é o Qwen base não calibrado especificamente para o prompt System 1. Para torná-lo um classificador de classe mundial:
1. **Dataset de Treinamento**:
   * Usar o benchmark [`multimodalart/jev-decision-index`](https://huggingface.co/datasets/multimodalart/jev-decision-index) ou extrair subconjuntos de triagem, moderação e roteamento de modelos.
2. **Treinamento com SFT / LoRA**:
   * Treinar o modelo base `Qwen/Qwen2.5-0.5B` ou `Qwen3.5-0.8B` usando Unsloth ou Hugging Face `trl` (`SFTTrainer`).
   * Formato de entrada: exatamente a estrutura de prompt gerada por `native/qwen-serve/src/prompt.rs`.
   * Função de perda: Cross-entropy apenas no token de decisão (o primeiro token gerado após as alternativas).
3. **Exportação para ONNX Q4**:
   * Salvar os pesos ajustados.
   * Exportar via Optimum / ONNX Runtime com quantização INT4 (`decoder_model_merged_q4.onnx`).
   * Criar um novo arquivo `.model` e publicar a release `v1.1.0`.

### Fase 2: Otimizações de Throughput no Servidor Rust (Meta: 3-4 req/s)
1. **Cache de Prefixo (KV Cache Reuse)**:
   * Em muitos fluxos de produção, as perguntas e as instruções de sistema são idênticas, mudando apenas o `state` do usuário.
   * Ao implementar reaproveitamento de tensores KV para prefixos repetidos, o tempo de decodificação cai para **menos de 50ms**.
2. **Concorrência e Sessões Paralelas**:
   * Atualmente, o servidor sincroniza chamadas ao decodificador com um `Mutex<Session>`.
   * Criando um pool de `Session` (ex: 2 a 4 instâncias em CPU multithread), requisições concorrentes serão processadas simultaneamente, elevando o throughput de 1 req/s para **3 a 5 req/s**.
3. **Ajuste de Intra-op Threads**:
   * Configurar e documentar flags de afinidade de CPU (`--threads`) otimizadas por contagem de núcleos físicos.

### Fase 3: Pacote NPM e CLI Pública
1. Criar `bin/cli.js` e wrapper Node/Bun similar ao Laya (`laya-system-one`), permitindo:
   ```bash
   npm install qwen-system-one
   npx qwen-system-one --port 8093
   ```
2. Adicionar detecção automática da plataforma para baixar o binário nativo correspondente gerado no CI (`qwen-serve-win32-x64`, `qwen-serve-linux-x64`, etc.).

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
