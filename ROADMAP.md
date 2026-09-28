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
| Máquina fresca, miss (inferência nova) | **~233–262ms** | **~4.3 req/s ✅ meta** |
| Cache-hit (request repetido) | **~1–2ms** | **~1000 req/s** |
| Carga sustentada (limite de potência do SoC, constante longa) | ~1100ms | ~0.9 req/s |

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

> ⚠️ Os números de throughput do bench acima são históricos: na época os níveis repetiam states já respondidos e viravam **parcialmente cache-hit** ("peak 888 req/s" era cache, não inferência). Desde 2026-09-28 o `bench-concurrent.js` **estoura o cache nos níveis** (nonce por request) e mede inferência real; o cache-hit é medido numa fase separada. Também desde então o `quick-check.js` cobre request **multi-pergunta** (caminho K>1) e o workflow ignora commits de `notebooks/`, `data/`, docs e `tools/*.py` (que não afetam os binários).

**Medição real por plataforma (run `36468080454`, 2026-09-28 — níveis cache-busted, p50 por request):**

| plataforma | c=1 p50 | c=1 req/s | c=4 p50 | c=4 req/s | hit p50 | multi-pergunta |
|---|---|---|---|---|---|---|
| `darwin-arm64` | **395ms** | **2.44** | 1574ms | 2.00 | 1ms | 1080ms |
| `win32-x64` | 586ms | 1.68 | 1793ms | 1.70 | 3ms | 1641ms |
| `win32-arm64` | 776ms | 1.29 | 2388ms | 1.28 | 11ms | 2370ms |
| `linux-arm64` | 814ms | 1.23 | 2506ms | 1.19 | 4ms | 2227ms |
| `linux-arm64-musl` | 824ms | 1.21 | 2542ms | 1.20 | 4ms | 2209ms |
| `linux-x64` | 926ms | 1.08 | 2830ms | 1.07 | 4ms | 2640ms |
| `linux-x64-musl` | 913ms | 1.07 | 2885ms | 1.07 | 3ms | 2464ms |
| `darwin-x64` | 1491ms | 0.58 | 5671ms | 0.60 | 4ms | 5520ms |

Padrões que confirmam o modelo do sistema (1 worker, fila serial):
* `req/s` **plano de c=1 a c=4** e p50 **linear em c** (c=2 ≈ 2× c=1, c=4 ≈ 4× c=1): concorrência só enfileira, não paraleliza;
* cache-hit em ~1-11ms e `hits=6/misses=20` idênticos nas 8 plataformas: o contrato de cache é determinístico;
* multi-pergunta (3 perguntas num request) ≈ 2.7× o custo de 1 pergunta nos runners 2-vCPU (o gargalo amortiza menos), ~2.4× no `darwin-arm64` (runner M1).

### ✅ Profundidade máxima em CPU — onde o tempo realmente vai (2026-09-28)

Investigação para responder "o que mais dá pra otimizar no runtime?". Medições locais (Snapdragon X 12-core), sempre com cache desligado:

**1. O prefill é compute-bound (não memory-bound).** Latência vs. tokens do prompt é linear com intercepto pequeno:

| tokens | latência | ms/token |
|---|---|---|
| 125 | 1105ms | 8.8 |
| 161 | 1221ms | 7.6 |
| 244 | 1859ms | 7.6 |
| 377 | 2825ms | 7.5 |

Consequências: (a) **batchar N perguntas numa passada só não reduz FLOPs** — o ganho seria só overhead de `session.run()` (~5-10ms/pergunta) e eficiência de kernel, insuficiente para justificar padding/numerics; (b) KV-prefix caching valeria à pena **mesmo sendo compute-bound** (pularia os FLOPs do prefixo compartilhado), mas o export atual ignora `past_*` (ver "Itens mortos"); (c) o único ataque real é **menos FLOPs** (modelo/prompt menores) ou **mais FLOPs por watt** (EP de NPU/GPU).

**2. O colapso sob carga sustentada é do SoC, provado fora do runtime.** Stress de CPU puro (matmul numpy, sem qwen-serve):

| janela | GFLOPS | relativo |
|---|---|---|
| 0-5s | 330 | 1.00× |
| 10-45s | 81-98 | **0.25-0.30×** |

O runtime cai de ~262ms → ~1100ms (0.24×) — o mesmo fator. É o envelope de potência/térmico do Snapdragon X (fanless), não código, não ORT, não tuning.

**3. Threads não mudam o sustentado.** Mediana sustentada por nº de threads (10 requests seguidos, mediana do request 3-10, 2 passadas):

| threads | frio (best 2) | sustentado (mediana) |
|---|---|---|
| 12 | 262ms | 1103ms |
| 8 | 305ms | 1097ms |
| 6 | 408ms | 1156ms |
| 4 | 545ms | 1091ms |

4-12 threads convergem para ~1100ms: menos threads só perdem burst (mais clocks por core não compensam menos cores). **Default de 12 threads continua correto**; nenhum tuning de thread resolve o teto.

**4. Teto do runtime em CPU atingido.** Custo por request: decoder ~95%, embed ~2-3%, prep ~1%. Com prefill compute-bound, prompt congelado e export sem KV, o que sobra em código é ruído (<5%, abaixo da variância térmica). O caminho para os 250ms **sustentados** (não só burst) passa por:

1. **EP de NPU/GPU** (mais FLOPs por watt — ataca exatamente o gargalo medido):
   - Windows/ARM (Snapdragon X): `dml` (Adreno GPU) ou `qnn` (Hexagon NPU) no `ort`;
   - macOS: `coreml` (Neural Engine);
   - Risco: suporte a `MatMulNBits` (Q4) por EP; precisa de flag `--ep` com fallback para CPU.
   - **Implementado em 2026-09-28** (opt-in): build com `--features ep-directml|ep-coreml|ep-nnapi|ep-qnn`, runtime `--ep auto|dml|coreml|nnapi|qnn`. `auto` cai para CPU em silêncio; nome explícito falha alto. Default continua `cpu` (numéricos bit-idênticos).
   - **Medido no Snapdragon X (Adreno, `--ep dml`)**: o DML registra e leva os matmuls + `lm_head` para a GPU, MAS os ops internos do GDN (sigmoid/decay/k_flatten/q_flatten/split) são forçados de volta ao CPU pela heurística do ORT ("CPU path is deemed faster") — o grafo ping-ponga CPU↔GPU com `MemcpyToHost/FromHost` **em todas as 24 camadas** (~120 cópias por prefill). Efeito líquido medido (123 tokens, 1 worker):

     | condição | CPU (12 threads) | DML (Adreno) |
     |---|---|---|
     | frio | **~230ms** | ~1800ms |
     | sustentado (mediana) | ~900-1140ms (colapso térmico) | **~770-910ms (flat, sem colapso)** |

     GPU entrega ~20-25% sob carga contínua **e** previsibilidade (sem abismo térmico); CPU ainda vence de longe em rajada fria. A partição CPU/GPU do GDN é o que segura o ganho — próximos passos: `with_dimension_override` (shapes estáticos podem reduzir os fallbacks de Gather), QNN no Hexagon (partição pode ser melhor), ou exportar os ops GDN como um fused op aceito pela GPU. Lembrete: EP acelerado muda os últimos decimais dos logits (kernels fundidos) — manter `--ep cpu` nos benchmarks de paridade.

### ❌ QNN/Hexagon testado de ponta a ponta — não vence (2026-09-28, curiosidade)

Teste completo do EP QNN no Snapdragon X (Hexagon NPU): `--features ep-qnn` + link dinâmico contra o `onnxruntime.dll` do NuGet `Microsoft.ML.OnnxRuntime.QNN` 1.24.4 (API 24; os pré-built padrão do `ort` não trazem o EP QNN) + runtime QNN ao lado do exe (`QnnHtp.dll`, stubs/skel V73 e V81). Descobertas estruturais:

* o **export OPT atual não carrega em builds stock do ORT** — `com.microsoft:CausalConvWithState` é um fused-op custom do Qwen3.5. Para testar QNN foi preciso o export padrão (`onnx-community/Qwen3.5-0.8B-ONNX`, ops primitivos; mesmo I/O, mas sem `num_logits_to_keep` e conv-cache 4-wide). O runtime ganhou **compat dual-export** (shapes dos `past_*` e offset de logits derivados do próprio grafo);
* `QNNExecutionProvider` registra nas duas sessões e roda ponta a ponta (respostas corretas) — mas o ganho é **zero mensurável** em todos os formatos testados (123 tokens, 1 worker, mediana de 10 requests):

| config | frio | sustentado (mediana) |
|---|---|---|
| std q4 + CPU | 958ms | 2428ms |
| std q4 + **QNN** | 2143ms | **2378ms** (≈ empate com CPU) |
| QDQ int8 + CPU | 3283ms | 5180ms |
| QDQ int8 + **QNN** | 4760ms | **5262ms** (≈ empate com CPU) |
| *(referência: OPT fundido + CPU)* | *230ms* | *1100ms* |
| *(referência: OPT fundido + DML)* | *1800ms* | *862ms* |

Leituras: (a) QNN e CPU empatam em todos os formatos → o NPU só pega sobras do grafo híbrido (GDN/conv/If) e o overhead de RPC do FastRPC come qualquer ganho; (b) o export **int8 QDQ** (formato preferido do HTP) é o MAIS LENTO de todos — 2× pior que q4 MatMulNBits em CPU; (c) nenhuma configuração QNN chega perto do OPT fundido (230ms/1100ms). Conclusão: **o caminho do NPU não vence com o modelo atual**; valeria só com um re-export dedicado QNN (fused ops decompostos + quantização HTP-friendly + shapes estáticos), que é trabalho de export, não de runtime. O ranking atual continua: **OPT fundido + CPU (rajada) e OPT fundido + DML (sustentado)**.
2. **Export com KV real** (reabre prefix-cache: K perguntas = 1 prefill do state + K templates; ~2× em request multi-pergunta). Contrato continua igual — só o export muda.
   - **Feito como runtime em 2026-09-28** (melhor que o esperado): o export OPT **já tem** past_*/present_* funcionais — o oráculo provou que prefill em chunks (chunk1 → present_* → chunk2) ≡ single-pass com `max|d|=2.2e-5` (ruído fp) na convenção `a` (position_ids absolutos nas 3 linhas). Implementado como **chunked prefill + cache de prefixo do template** (`--prefix-cache on|off`, default on; off = single-pass bit-idêntico). Ver seção própria abaixo.
3. **Fase 2 (distilação/quant de ativações)**: menos params/FLOPs é a única forma de baixar latência em CPU.
4. **Redução de tokens do prompt** (10-25%) se/ quando o contrato puder mudar — hoje 114 tokens ≈ 230ms frio.
5. **Micro-batching entre requests concorrentes** (só para throughput, não latência): hoje c=2/4 apenas enfileira (mesmo ~1 req/s); batch de verdade exigiria batch-dim dinâmico + scheduler.

### ✅ Empacotamento e Publicação da Base
* Modelo base empacotado em `models/qwen-0.8b-q4.model` (533 MB).
* Publicado como release asset no GitHub: [Release v1.0.0](https://github.com/italoalmeida0/qwen-system-one/releases/tag/v1.0.0).
* Script de aquisição e extração com verificação SHA256 implementado (`tools/acquire-model.js`).

### ✅ KV real: chunked prefill + cache de prefixo do template (2026-09-28, Fase 1B)

A premissa antiga ("o export ignora `past_*`, prefix-cache exige re-export") estava **errada** — provada errada por oráculo numérico (`kv-oracle.py`): dividir o prefill em chunks e passar `present_*` → `past_*` reproduz o single-pass com `max|d|=2.2e-5` (2.2e-5 = ruído de ponto flutuante). A convenção vencedora testada contra o output single-pass real: **position_ids absolutos (`P..P+S`) nas 3 linhas** + `attention_mask` sobre o comprimento TOTAL; qualquer split da sequência de ids é correto (basta chunk1+chunk2 == full).

Implementado no runtime (`native/qwen-serve`): o prompt é dividido em **prefixo do template** (idêntico entre requests com a mesma pergunta) + sufixo (state + cauda). O prefixo roda UMA vez e seus `present_*` (conv/recurrent/KV reais) ficam num cache global (16 entradas, chave = ids do prefixo — determinístico por qdef via LCP de tokens); requests seguintes preenchem só o sufixo. `--prefix-cache off` restaura o single-pass bit-idêntico.

**Paridade validada** (quick-check, 10 casos EN/PT/ES + multi-pergunta): decisões E probabilidades exibidas **idênticas** com cache on vs off (o único "mismatch" do caso 9 é comportamento pré-existente do modelo — os dois modos concordam).

**Ganho medido (local, 1 worker × 12 threads, prompt ~114-123 tokens, template 86):**

| cenário | antes (single-pass) | depois (KV real) |
|---|---|---|
| miss (primeiro request do qdef) | ~230ms frio / ~900ms quente | 466ms (2 runs: template + sufixo) |
| **hit (requests seguintes, sustentado)** | **~850-1100ms** (colapso térmico) | **109-129ms** (mediana 129ms) |
| quick-check (10 casos + multi) | avg 515-770ms | **avg 301ms** (3.3 req/s) |
| multi-pergunta (3 qdefs, 1º request) | 2668ms | 2036ms (1 hit + 2 miss) |
| multi-pergunta (3 qdefs, aquecido) | ~2600ms | **~350-400ms** |

~8.5× no sustentado e **sem abismo térmico** (37 tokens de sufixo mal aquecem o SoC). Observações: o `embed` ainda roda a sequência inteira (barato, ~2-3%); o cache guarda `DynValue`s compartilhados entre workers (Arc clones, zero cópia); e o sufixo do state não é cacheável no formato atual (ordem template→state no prompt — reordenar para state→template habilitaria cache do state entre perguntas do MESMO request, ~2× em multi, mas muda o prompt-contract).

### ✅ CI — tempos por job e o que foi otimizado (2026-09-28)

Run `36454700663` (8 jobs em paralelo, wall total **4m18s** — o wall é definido pelo job mais lento):

| job | total | etapas dominantes |
|---|---|---|
| `darwin-x64` | **4m14s** (caminho crítico) | brew ORT 41s + build 1m55s + modelo 23s + smoke 45s |
| `linux-x64-musl` | **3m50s** | build Docker/Alpine **2m58s** (compilava tudo do zero) |
| `win32-arm64` | 2m41s | build 1m49s |
| `win32-x64` | 2m38s | build 1m45s |
| `linux-arm64-musl` | 2m50s | build Docker **2m08s** |
| `darwin-arm64` | 1m48s | build 22s (deps via rust-cache) |
| `linux-x64` / `linux-arm64` | 1m18s / 1m02s | build ~20s (rust-cache) |

Otimizações aplicadas (2026-09-28):
* `paths-ignore` no push: commits de `notebooks/`, `data/`, `call/`, `*.md` e `tools/*.py` não disparam mais os 8 builds nativos (a maior parte dos pushes do dia era só notebook/dados — hoje cada um gastava ~25 min de runner);
* cache do cargo registry + árvore de build musl (`.cargo-musl`/`.target-musl`, keyed no `Cargo.lock`): o container Alpine recompilava a árvore inteira a cada run (~130-180s);
* `HOMEBREW_NO_AUTO_UPDATE=1` no job darwin-x64 (o `brew update` custava ~30s e não agregava);
* bench com inferência real (`--requests 6` para compensar o custo maior).

Efeito medido (runs `36468080454` frio → `36468815911` cache quente): "Build in Alpine" **2m58s/2m08s → 63s/55s** (x64/arm64), jobs musl totais **3m50s/2m50s → 2m08s/1m57s**, `brew install` darwin-x64 41s → 24s. Wall total do run: 4m26s → 4m38s (estável — agora definido pelo `darwin-x64`: build 96s + brew 24s + modelo + smokes em série).

**Fase extra (2026-09-28 tarde): EP de aceleração opt-in.** Os jobs `win32-*` agora compilam com `ep-directml` e `darwin-*` com `ep-coreml`, mais um smoke `--ep auto --cases 2` em ambas (prova o caminho acelerado com fallback silencioso em runner sem GPU). Ver detalhes e medições na seção de profundidade.

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
