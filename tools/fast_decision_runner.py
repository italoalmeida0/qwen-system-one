#!/usr/bin/env python3
"""
fast_decision_runner.py - Runner concorrente de altíssima vazão para o Jev Decision Index.
Executa as 120.000 requisições com 32 threads simultâneas contra o servidor A100 (porta 8093).
Atinge de 200 a 400+ req/s, finalizando todo o benchmark em ~5 a 10 minutos!
Totalmente compatível com o formato oficial do Decision Index (0.2.1).
"""

import os
import sys
import time
import json
import gzip
import argparse
import threading
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error

# Importar validadores oficiais do kit se disponíveis
try:
    from decision_index.engines.base import CAPACITY_MARKERS, validate, Unsupported
except ImportError:
    CAPACITY_MARKERS = (
        "options per choice",
        "a choice needs at least two options",
        "a score takes 2 to 10 levels",
        "the canvas holds",
        "maximum context length",
        "maximum model length",
        "longer than the maximum model length",
        "context window",
        "too many tokens",
    )
    def validate(questions, response):
        if set(response.get("answers", {})) != set(questions):
            raise ValueError("Question keys mismatch")

def stamp():
    return datetime.now(timezone.utc).isoformat()

def read_jsonl_gz_or_plain(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    parser = argparse.ArgumentParser(description="Fast Decision Index Concurrent Runner")
    parser.add_argument("--base-url", default="http://127.0.0.1:8093", help="Base URL of GPU server")
    parser.add_argument("--rows", required=True, help="Path to selected-rows.jsonl.gz")
    parser.add_argument("--out", required=True, help="Output directory for results")
    parser.add_argument("--concurrency", type=int, default=32, help="Number of concurrent workers")
    parser.add_argument("--edition", default="0.2.1", help="Decision Index edition")
    parser.add_argument("--model", default="qwen-system-one", help="Model name")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for test")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    print("=" * 70)
    print("   QWEN-SYSTEM-ONE v1.1.0 : FAST CONCURRENT DECISION INDEX RUNNER   ")
    print(f"   Alvo: {args.base_url} | Concorrência: {args.concurrency} threads")
    print("=" * 70)

    # 1. Carregar resultados anteriores para RETOMADA sem perdas
    completed_ids = set()
    valid_lines = []
    if results_path.exists():
        print(f"[*] Verificando resultados anteriores em {results_path}...")
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("status") in ("ok", "unsupported", "abstained"):
                            completed_ids.add(rec["run_id"])
                            valid_lines.append(line.strip() + "\n")
                    except Exception:
                        pass
        # Limpar registros com erro para reprocessá-los corretamente
        with results_path.open("w", encoding="utf-8") as f:
            for vl in valid_lines:
                f.write(vl)
        print(f"[*] Encontradas {len(completed_ids):,} questoes validas ja concluidas! Pulando-as...")

    # 2. Carregar todas as questões da suite
    print(f"[*] Carregando questoes de: {args.rows}...")
    t0_load = time.perf_counter()
    rows_to_process = []
    total_in_suite = 0
    for row in read_jsonl_gz_or_plain(args.rows):
        total_in_suite += 1
        rid = row["_evaluation"]["run_id"]
        if rid not in completed_ids:
            rows_to_process.append(row)
            if args.limit and len(rows_to_process) >= args.limit:
                break

    print(f"[*] Total na Suite: {total_in_suite:,} | Ja feitas: {len(completed_ids):,} | A processar: {len(rows_to_process):,}")
    if not rows_to_process:
        print("[+] Todas as questoes ja foram concluidas! Prosseguindo para o calculo da nota...")
        score_now(args.edition, results_path, out_dir)
        return

    # 3. Preparar arquivo de log thread-safe
    log_file = results_path.open("a", encoding="utf-8")
    write_lock = threading.Lock()

    # Contadores
    counts = {"ok": 0, "unsupported": 0, "error": 0, "total_done": len(completed_ids)}
    stat_lock = threading.Lock()

    api_url = args.base_url.rstrip("/") + "/v1/systemone"

    # Criar sessão HTTP de alta velocidade usando urllib pool ou http.client
    import urllib3
    http = urllib3.PoolManager(
        maxsize=args.concurrency * 2,
        retries=urllib3.Retry(total=3, backoff_factor=0.2),
        timeout=urllib3.Timeout(connect=10.0, read=300.0)
    )

    t_start = time.perf_counter()
    last_print = [t_start]
    done_count = [0]

    def worker_task(row):
        eval_info = row["_evaluation"]
        rid = eval_info["run_id"]
        payload = {
            "model": args.model,
            "state": row["state"],
            "questions": row["questions"]
        }
        
        t0 = time.perf_counter()
        started_utc = stamp()
        status = "ok"
        error_msg = None
        response_data = None
        
        req_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        
        try:
            r = http.request(
                "POST",
                api_url,
                body=req_body,
                headers={"Content-Type": "application/json"}
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            if r.status in (400, 413, 422):
                msg = r.data.decode("utf-8", errors="replace")
                if any(m in msg for m in CAPACITY_MARKERS):
                    status = "unsupported"
                    error_msg = msg
                else:
                    status = "error"
                    error_msg = f"HTTP {r.status}: {msg}"
            elif r.status != 200:
                status = "error"
                error_msg = f"HTTP {r.status}: {r.data.decode('utf-8', errors='replace')}"
            else:
                raw_json = json.loads(r.data.decode("utf-8"))
                response_data = {k: v for k, v in raw_json.items() if k != "evaluation_trace"}
                validate(row["questions"], response_data)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            status = "error"
            error_msg = str(e)

        completed_utc = stamp()
        
        record = {
            **eval_info,
            "started_utc": started_utc,
            "completed_utc": completed_utc,
            "engine": "http",
            "status": status,
            "total_wall_ms": elapsed_ms,
            "model_request_wall_ms": elapsed_ms,
        }
        if response_data is not None:
            record["response"] = response_data
        if error_msg:
            record["error"] = error_msg

        line_str = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        
        with write_lock:
            log_file.write(line_str)
            
        with stat_lock:
            counts[status] = counts.get(status, 0) + 1
            counts["total_done"] += 1
            done_count[0] += 1
            curr_done = done_count[0]
            
            # Print periódico de status a cada 0.5s ou a cada 20 itens
            now = time.perf_counter()
            if now - last_print[0] >= 1.0 or curr_done == len(rows_to_process):
                dt = now - t_start
                rate = curr_done / dt if dt > 0 else 0
                total_done_all = counts["total_done"]
                pct = (total_done_all / total_in_suite) * 100.0 if total_in_suite > 0 else 0
                remaining = len(rows_to_process) - curr_done
                eta_sec = remaining / rate if rate > 0 else 0
                eta_m = int(eta_sec // 60)
                eta_s = int(eta_sec % 60)
                elapsed_m = int(dt // 60)
                elapsed_s = int(dt % 60)
                
                print(
                    f"\r[{total_done_all:,}/{total_in_suite:,} ({pct:.1f}%)] "
                    f"Vazao: {rate:.1f} req/s | "
                    f"OK: {counts['ok']:,} | "
                    f"Unsup: {counts['unsupported']:,} | "
                    f"Err: {counts['error']:,} | "
                    f"Tempo: {elapsed_m:02d}:{elapsed_s:02d} | "
                    f"ETA: {eta_m:02d}:{eta_s:02d}   ",
                    end="",
                    flush=True
                )
                last_print[0] = now
                log_file.flush()

    print(f"\n[*] Disparando pool de {args.concurrency} workers paralelos na A100...")
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker_task, r) for r in rows_to_process]
        for f in as_completed(futures):
            f.result()

    log_file.flush()
    log_file.close()
    
    total_time = time.perf_counter() - t_start
    final_rate = len(rows_to_process) / total_time if total_time > 0 else 0
    print(f"\n\n[✓] Benchmark concluído com sucesso!")
    print(f"    Total processado nesta sessão: {len(rows_to_process):,} requisições")
    print(f"    Tempo total decorrido: {total_time:.1f} segundos ({total_time/60:.2f} minutos)")
    print(f"    Vazão média atingida: {final_rate:.1f} requisições por segundo!")
    print(f"    Total final consolidado: {counts['total_done']:,} / {total_in_suite:,}")

    # 4. Calcular o score oficial
    score_now(args.edition, results_path, out_dir)

def score_now(edition, results_path, out_dir):
    print("\n" + "=" * 70)
    print("   CALCULANDO PONTUACAO OFICIAL (JEV DECISION INDEX 0.2.1)       ")
    print("=" * 70)
    import subprocess
    cmd = [
        sys.executable, "-m", "decision_index", "score",
        "--edition", edition,
        "--results", str(results_path),
        "--out", str(out_dir)
    ]
    print(f"Executando: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print("\n" + res.stdout)
        scores_file = Path(out_dir) / "scores.json"
        if scores_file.exists():
            data = json.loads(scores_file.read_text(encoding="utf-8"))
            print("=" * 70)
            print("   RESUMO OFICIAL DO LEADERBOARD (SCORES.JSON)                  ")
            print("=" * 70)
            print(f"   Balanced Skill (Decision Index) : {data.get('decision_index', {}).get('balanced_skill', 'N/A')}")
            print(f"   Raw Accuracy                   : {data.get('decision_index', {}).get('raw_accuracy', 'N/A')}")
            print(f"   Status Completo                : {data.get('complete', False)}")
            print("=" * 70)
    else:
        print("[ERRO] Falha ao calcular scores:")
        print(res.stderr)

if __name__ == "__main__":
    main()
