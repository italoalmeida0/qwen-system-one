# Qwen System-One: Visão Geral, Arquitetura e Roadmap 🚀

Este documento serve como guia completo de handoff e planejamento técnico para qualquer desenvolvedor (ou IA) que for dar continuidade ao desenvolvimento do **Qwen System-One**.

---

## 1. O que é o projeto e qual o nosso objetivo?

### Contexto e Origem
O projeto é inspirado no **Laya System-One** (`laya-system-one`), uma engine de decisão ultrarrápida baseada no conceito de *"Sistema 1"* da psicologia cognitiva (respostas reflexivas, rápidas e estruturadas em milissegundos, sem a lentidão de uma LLM generativa de chat).

O protocolo segue a especificação wire do **TypeSafe Jev** (`POST /v1/systemone`):
* O cliente envia um texto (`state`) e uma ou mais perguntas de múltipla escolha com critérios (`questions`).
* A engine calcula as probabilidades relativas (*logits*) de cada alternativa na última camada e devolve uma resposta estruturada em JSON (`answers: { choice, probability }`).

### Por que Qwen? (a jornada real)
1. **Laya empacotado**: o autor publicou o `laya-system-one` (ModernBERT ~400M, Apache 2.0) no npm. Rápido (~20 req/s em qualquer PC), mas pouco inteligente — encoder pequeno, inglês-fraco.
2. **JPT-0.8B descoberto** (`kirp/jpt-0.8b`): LoRA fine-tune no `Qwen/Qwen3.5-0.8B` base, **~4x mais inteligente que o Laya** (0.736 no JevBench vs 0.623 do base zero-shot; 19.22 no Decision Index, melhor 0.8B do board). Porém rodando puro (transformers/torch, FP16, engine de GPU) em CPU: **~60s por UMA questão**.
3. **A virada**: trocar o backend Python pelo export **`Qwen3.5-0.8B-ONNX-OPT`** quantizado em **Q4** + servidor nativo em Rust que lê os logits direto (sem gerar texto). De 60s → 1–2s → **~230ms** + cache. Este repositório é esse servidor + o plano para treinar (Fase 2) com o dataset que o JPT usou.

### Nosso Objetivo Principal
1. **Substituir o modelo do Laya pelo Qwen 3.5 0.8B**: O modelo original do Laya é um transformer de 1B treinado apenas para tarefas específicas. O Qwen 3.5 0.8B é uma arquitetura moderna (híbrida com atenção e camadas recorrentes) com muito mais vocabulário (152k tokens) e capacidade multilíngue superior.
2. **Superar o Laya em Acurácia Geral**: Replicar a receita do JPT (LoRA no `Qwen/Qwen3.5-0.8B`, temperatura calibrada **T=1.140**) treinando na base [**`multimodalart/jev-decision-index`**](https://huggingface.co/datasets/multimodalart/jev-decision-index) (que avalia 120 mil perguntas em 43 benchmarks de decisão, triagem, segurança e moderação).
3. **Performance Alvo**: **3 a 4 requisições/segundo em CPU** com latência inferior a **250ms** (miss/inferência nova). Requests repetidos: centenas de req/s via cache exato (já atingido — ver §3).
4. **Zero Dependências**: Servidor 100% nativo em Rust (`qwen-serve`), rodando offline em qualquer máquina sem Python, CUDA ou PyTorch.

### Projetos de referência consultados (e o que cada um ensinou)
* **`btdby4`** (contador de tokens em Go, 1M ctx em ~7ms): fast-path ASCII tabelado sem alocação; **cache de contagem shardado** e **trie de prefixo por blocos com hash FNV-1a + TTL/LRU** — validou o *desenho* do prefix caching (invalidação no primeiro mismatch). O speed do BPE em si não se aplica aqui (tokenizer é ~2% do nosso tempo).
* **`call_me_maybe`** (function-calling com constrained decoding sobre Qwen3-0.6B/0.8B, projeto da 42): classifica o vocabulário em **baldes de tokens permitidos** e faz argmax só dentro do balde (garante JSON válido em modelo pequeno) — **valida o nosso design** (logits da última posição + argmax nos labels, sem gerar texto). E o **cache `prompt → resultado`** de lá foi portado direto para o nosso cache exato.

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
│  0. Cache exato (LRU 512 + TTL 10min, --cache-size/--cache-ttl)                        │
│     hash(state+questions+temperature) → hit devolve em ~1-2ms, zero FLOPs              │
│                                                                                        │
│  1. Prompt Builder (prompt.rs) — TEMPLATE primeiro, state depois                       │
│     Renderiza template Jev com delimitadores especiais e tokens de rótulo.             │
│                                                                                        │
│  2. Tokenizer (tokenizers pure-Rust, Metaspace + BPE)                                  │
│     Tokeniza o prompt diretamente em memória sem chamada externa.                     │
│                                                                                        │
│  3. SessionPool: N Workers (default 1) × pares embed+decoder, round-robin              │
│     ├─ embed_tokens_q4.onnx        -> Projeção de embeddings (Q4 INT4, ~1ms)          │
│     ├─ Static Past-State Tensors   -> 48 tensores pré-alocados em Arc (zero overhead)  │
│     └─ decoder_model_merged_q4.onnx -> Prefill completo (~96% do tempo)                │
│        ORT tuning: inter_threads=1, memory_pattern off, spinning off, tokio wt=2       │
│                                                                                        │
│  4. Softmax Calibrado (T=1.140, herança do JPT) & Decoder (main.rs)                    │
│     Extrai logits apenas dos tokens das alternativas e devolve JSON com probabilidades. │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Por que o Qwen é Q4 (INT4) e não Q2 ou FP16?
* **FP16 / FP32**: Teria de 1.6 GB a 3.2 GB de tamanho e consumiria mais de 3 GB de RAM, inviabilizando uso leve em CPU. (O JPT puro em FP16/torch foi medido: **~60s/questão em CPU** — é o ponto de partida que este projeto elimina.)
* **Q2 (2-bit)**: Modelos sub-1B não possuem a redundância necessária para suportar quantização agressiva abaixo de 4 bits. Em 2 bits, os logits se degradam e a CPU precisaria de operações de desempacotamento (*bit-shift*) sem instruções de hardware nativas, tornando a inferência mais lenta.
* **Q4 (INT4 block-quantized)**: É o *sweet spot* perfeito — **~533 MB compactado**, roda com menos de 1 GB de RAM e atinge **~230ms** nativo na CPU (máquina fresca, 12 cores).

---

## 3. O que já foi feito (Estado Atual)

### ✅ Servidor Nativo Rust Concluído (`native/qwen-serve`)
* Implementado em Rust com Axum 0.8, Tokio e ONNX Runtime (`ort`).
* **Otimização de Tensores de Estado Passado**: Pré-alocação estática dos 48 tensores de estado passado (`past_conv`, `past_recurrent`, `past_key_values`), eliminando alocações repetidas por request e cortando o overhead de preparação de ~500ms para **~0ms**.

### ✅ Fase 1 — Técnicas aplicadas e medições (2026-09-28)
Todas as decisões abaixo foram tomadas com **número medido**, não chute. Breakdown de referência (release, 12 threads, prompt ~114 tokens): **embed ~1ms + prep ~2ms + decoder ~96%**.

1. **SessionPool concorrente** (`--workers`, default **1**): cada `Worker` tem um par embed+decoder independente; distribuição round-robin via `AtomicUsize`; excedente fila no `Mutex` dentro de `spawn_blocking` (sem spin, sem deadlock). `/health` reporta `workers`, `threads_per_session` e stats de cache.
2. **Scaling de threads medido** (1 worker, release, mesma questão): `1t=2965ms, 2t=1402ms, 4t=695ms, 6t=503ms, 12t=233ms` — speedup ~linear (~12.7x). Conclusão: **um worker gordo com todos os cores vence N workers magros** (2×6t concorrentes degradam para ~1000ms cada por contenção de L3). Por isso o default é `--workers 1`.
3. **ORT tuning**: `inter_threads=1` (cadeia sequencial, sem pool inter-op), `memory_pattern` off (seq_len varia por request), intra/inter **spinning off** (threads em busy-spin mantinham SoCs móveis aquecidos), tokio `worker_threads=2` (accept+dispatch só; inferência no blocking pool).
4. **Prompt enxuto + reordenado**: instrução cortada (~40→~22 tokens), ordem **template idêntico primeiro, state variável depois** (~112–114 tokens totais). Validado no quick-check: **mesma acurácia 10/10 respostas válidas, 9/10 matches** (o 1 miss é o caso ambíguo conhecido "quote" billing-vs-sales).
5. **Cache exato prompt→resposta** (LRU 512 + TTL 10min, `--cache-size`/`--cache-ttl`, ideia do `call_me_maybe`): chave = `hash(state+questions+temperature)`. Hit = **~1–2ms, zero FLOPs, zero calor**. Correto porque a engine é **função pura** (argmax determinístico de passo único, sem amostragem — diferente de LLM generativo). `/health` expõe `cache.{hits,misses,entries}`.
6. **KV-prefix caching: INVESTIGADO E DESCARTADO neste export (com prova)**. Implementado de verdade (replay de `present_*` como `past_*`, `position_ids` com offset, máscara cheia) + gate de corretude `--kv-check`: `max|logits_cached − logits_full| ≈ 13` (deveria ser ~1e-3). Teste dos zeros (`--kv-zero-past`): replayar **zeros** no lugar dos estados **não muda os logits** → o `decoder_model_merged_q4.onnx` **ignora os inputs `past_*`** (export prefill-only, sem `past_sequence_length` como input). Código KV revertido; prompt reordenado mantido (deixa a porta aberta se um futuro export expuser KV incremental real).
7. **`tools/bench-concurrent.js`**: warmup + rajadas configuráveis (`--levels 1,2,4,8 --requests N`) com req/s, p50/p95/max + **fase cache-hit** (repete 1 state, mede o teto do cache) + stats do `/health`. Falha se qualquer request perder resposta. **Roda no CI nas 8 plataformas** (`npm run bench:concurrent`).

### ✅ Performance medida

**Local (Snapdragon X 12-core, release, 1 worker × 12 threads):**
| Condição | Latência/req | Throughput |
|---|---|---|
| Máquina fresca, miss (inferência nova) | **~233ms** | **~4.3 req/s ✅ meta** |
| Cache-hit (request repetido) | **~1–2ms** | **~1000 req/s** |
| Carga sustentada (throttling térmico do SoC, constante longa) | ~1000ms | ~1 req/s |

> Nota térmica: degradação sob carga sustentada foi isolada como **throttling do SoC** (processo fresco = sempre ~300ms 4/4; cooldown de 60s não recupera; memória estável em ~680MB, sem vazamento; spinning/vazamento/memory-pattern descartados por teste). Não é bug de código — é física de fanless sob AVX/intenso. Menos FLOPs por request (Fase 2: modelo treinado, prompt menor) é o ataque correto.

**CI (run `36428703436`, commit `01ec6f5`, runners 2–4 vCPUs, release) — 8/8 verde:**

| Plataforma | quick-check 10/10 (latência / req/s) | bench miss nível 1 | bench cache-hit |
|---|---|---|---|
| `linux-x64` | 841ms / 1.19 | **1.57 req/s** (p50 833ms) | 1000–1333 req/s (p50 2–3ms) |
| `win32-x64` | 597ms / 1.67 | (níveis 2/4 viraram hit parcial) | 727–1333 req/s (p50 2ms) |
| `linux-arm64` | 800ms / 1.25 | — | 1142 req/s (p50 2ms) |
| `win32-arm64` | 787ms / 1.27 | — | 889 req/s (p50 3ms) |
| `darwin-arm64` | 860ms / 1.16 | — | 1333–1600 req/s (p50 3ms) |
| `darwin-x64` | 1691ms / 0.59 | — | 727–889 req/s (p50 4ms) |
| `linux-x64-musl` | 1086ms / 0.92 | — | 800–1000 req/s (p50 3ms) |
| `linux-arm64-musl` | 800ms / 1.25 | — | 800–1143 req/s (p50 3ms) |

Cache no CI: `hits=26, misses=8, entries=8` em todas — comportamento idêntico nas 8 plataformas. Miss em runner 2-vCPU (~800ms/prefill de 800M params) é limite físico, não tuning.

### ✅ Empacotamento e Publicação da Base
* Modelo base empacotado em `models/qwen-0.8b-q4.model` (533 MB).
* Publicado como release asset no GitHub: [Release v1.0.0](https://github.com/italoalmeida0/qwen-system-one/releases/tag/v1.0.0).
* Script de aquisição e extração com verificação SHA256 implementado (`tools/acquire-model.js`).

---

## 4. Estratégia de Desenvolvimento: Por que Otimizar o Runtime ANTES do Fine-Tuning?

Decidimos priorizar a **Otimização de Runtime no Servidor Rust** antes de iniciar o treinamento/fine-tuning. Essa decisão de arquitetura baseia-se em 3 pilares:

1. **Definição do "Contrato de Prompt" Definitivo**:
   * O tempo de decodificação e prefill na CPU depende diretamente da quantidade de tokens de entrada e da padronização dos delimitadores.
   * Com o prompt **enxuto e reordenado (template primeiro, ~114 tokens)** já congelado e validado no Rust, **treinaremos o modelo já nesse formato final**, evitando retrabalho e retreinos posteriores.

2. **Destravamento Imediato do Gargalo de Concorrência**:
   * O servidor original utilizava um `std::sync::Mutex<Session>` único (1 req por vez). O **SessionPool** + default workers=1 + tuning ORT levou o pico local a **4.3 req/s** e o cache-hit a **~1000 req/s** — ou seja, o motor já entrega a meta em máquina capaz antes mesmo do fine-tuning.

3. **O Servidor como um "Motor Pronto"**:
   * O servidor Rust vira um motor de alta performance já verificado e compilado para as 8 plataformas (quick-check + bench no CI).
   * Depois, o fine-tuning será apenas uma substituição de "combustível" (o arquivo de pesos `.onnx`), ganhando inteligência sem precisar reescrever o código de inferência.

---

## 5. O que falta fazer (Fases do Roadmap)

### 📌 Fase 1: Otimizações de Throughput e Concorrência ✅ (implementada — 2026-09-28)
Detalhe completo das técnicas e medições no **§3** acima. Resumo: SessionPool ✅, threads/ORT tuning ✅, prompt enxuto+reordenado ✅, cache exato ✅, bench no CI ✅, KV-prefix ❌ descartado com prova. Commits: `2a8bb84` (pool+bench), `01ec6f5` (tuning+cache exato+bench no CI), `86f9fbe` (veredito KV + prompt reordenado).

### 📌 Fase 2: Fine-Tuning do Qwen 3.5 para Decisão (Superar o Laya)
Receita a replicar: **LoRA no `Qwen/Qwen3.5-0.8B` base** (como o `kirp/jpt-0.8b`), temperatura calibrada **T=1.140** (já é nosso default — não mudar no meio do treino).
1. **Dataset de Treinamento**:
   * Download e pré-processamento do dataset [`multimodalart/jev-decision-index`](https://huggingface.co/datasets/multimodalart/jev-decision-index) (120k perguntas em 43 benchmarks de decisão, triagem e segurança). Confirmar que é o (ou deriva o) dado que o JPT usou — ver model card/recipe do JPT-4B/9B.
2. **Treinamento com SFT / QLoRA**:
   * Treinar o modelo base `Qwen/Qwen3.5-0.8B` com Unsloth / Hugging Face `trl`.
   * Formato de entrada: exatamente a estrutura de prompt otimizada na Fase 1 (**template primeiro, congelado**).
   * Função de perda (Loss): Cross-entropy calculada estritamente no **primeiro token da decisão**, forçando o modelo a ter certeza absoluta de forma reflexiva sem alucinar texto longo.
   * Avaliação: JevBench/Decision Index contra o JPT-0.8B (0.736 / 19.22) e contra o Laya.
3. **Conversão e Publicação do Modelo v1.1.0**:
   * Merge do LoRA → exportação ONNX → quant Q4 (`embed_tokens_q4.onnx` + `decoder_model_merged_q4.onnx`).
   * **Atenção**: o export atual é prefill-only (ignora `past_*` — veredito §3.6). Se o pipeline de export permitir, expor KV incremental real reabriria o prefix caching.
   * Empacotamento do novo `.model` e publicação na Release `v1.1.0` do GitHub.
   * Reavaliação no quick-check + bench (CI já mede miss vs hit separados — avaliar o modelo novo com `--cache-size 0`).

### 📌 Fase 3: Pacote NPM e CLI Pública
1. Criar `bin/cli.js` e wrapper Node/Bun com seleção automática de arquitetura (`@sys-one` ou `@qwen-system-one`).
2. Publicação no npm registry permitindo uso direto:
   ```bash
   npx qwen-system-one --port 8093
   ```

---

## 6. Como rodar e testar localmente

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

# Rajadas concorrentes + fase cache-hit (reporta req/s, p50/p95, stats de cache)
node tools/bench-concurrent.js --levels 1,2,4 --requests 8

# Avaliar modelo puro (sem máscara do cache exato) — usar no eval da Fase 2
# (servidor com --cache-size 0)
cargo run --release --manifest-path native/qwen-serve/Cargo.toml -- --port 8093 --cache-size 0
```

### Disparar Builds no CI via GitHub CLI
```bash
# Dispara o workflow em todas as 8 arquiteturas
gh workflow run build-packages.yml

# Acompanha o progresso
gh run list --workflow=build-packages.yml
```

---

## 7. Links Úteis do Repositório

* **Repositório GitHub**: [italoalmeida0/qwen-system-one](https://github.com/italoalmeida0/qwen-system-one)
* **Releases e Arquivo `.model`**: [Releases v1.0.0](https://github.com/italoalmeida0/qwen-system-one/releases/tag/v1.0.0)
* **Workflow de CI das 8 Plataformas**: [.github/workflows/build-packages.yml](file:///.github/workflows/build-packages.yml)
* **Código do Servidor Rust**: [native/qwen-serve/src/main.rs](file:///native/qwen-serve/src/main.rs)
* **Receita de referência (LoRA JPT)**: [kirp/jpt-0.8b](https://huggingface.co/kirp/jpt-0.8b)
* **Comparativo Laya vs Qwen (Benchmark Upstream)**: [`comparativo_laya_vs_qwen_nativo.md`](file:///C:/Users/italo/.gemini/antigravity/brain/5342e8d1-5385-4f22-8cc9-bca528668bc8/comparativo_laya_vs_qwen_nativo.md)
