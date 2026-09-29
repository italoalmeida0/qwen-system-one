#!/usr/bin/env python3
"""
compute_scores.py - Script para preparar os arquivos da suite e calcular a pontuação oficial
do Jev Decision Index (Edição 0.2.1) sobre as 124.971 requisições concluídas.
"""

import os
import sys
import json
import gzip
import shutil
from pathlib import Path

def main():
    print("=" * 70)
    print("   QWEN-SYSTEM-ONE v1.1.0 : CALCULO FINAL DO JEV DECISION INDEX   ")
    print("=" * 70)

    base_dir = Path("/content/decision_benchmark_a100/decision-index")
    if not base_dir.exists():
        base_dir = Path(".")

    suite_dir = base_dir / "suite-0.2"
    suite_dir.mkdir(parents=True, exist_ok=True)

    results_path = Path("/content/decision_benchmark/runs/qwen-system-one-v1.1/results.jsonl")
    if not results_path.exists():
        # Fallback se estiver no dir atual
        results_path = Path("runs/qwen-system-one-v1.1/results.jsonl")

    out_dir = results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fazer backup imediato de results.jsonl para o Drive
    drive_backup = Path("/content/drive/MyDrive/qwen-system-one")
    if drive_backup.exists() and results_path.exists():
        print(f"[*] Fazendo backup de seguranca de {results_path} no Google Drive...")
        shutil.copyfile(results_path, drive_backup / "results_benchmark_final.jsonl")
        print(f"  -> Backup salvo em: {drive_backup / 'results_benchmark_final.jsonl'}")

    # 2. Localizar selected-rows e added-rows
    print("[*] Localizando arquivos da suite...")
    selected_candidates = [
        Path("/content/suite-work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz"),
        Path("/content/suite-work/artifacts/benchmark-suite/release-v1-rebuilt/selected-rows.jsonl.gz"),
        Path("selected-rows.jsonl.gz"),
    ]
    src_selected = next((p for p in selected_candidates if p.exists()), None)
    if not src_selected:
        print("[ERRO] selected-rows.jsonl.gz nao encontrado!")
        sys.exit(1)

    dst_selected = suite_dir / "selected-rows.jsonl.gz"
    if not dst_selected.exists():
        print(f"  -> Copiando selected-rows de {src_selected}...")
        shutil.copyfile(src_selected, dst_selected)

    # added-rows
    src_added = Path("/content/suite-work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz")
    dst_added = suite_dir / "added-rows.jsonl.gz"
    if src_added.exists() and not dst_added.exists():
        print(f"  -> Copiando added-rows de {src_added}...")
        shutil.copyfile(src_added, dst_added)
    elif not dst_added.exists():
        print("  -> Criando added-rows.jsonl.gz vazio para validacao de schema...")
        with gzip.open(dst_added, "wb") as f:
            pass

    # manifest
    dst_manifest = suite_dir / "manifest.json"
    hub_manifest = base_dir / "hub/0.2.1/manifest.json"
    if not hub_manifest.exists():
        hub_manifest = base_dir / "hub/manifest.json"
    if hub_manifest.exists() and not dst_manifest.exists():
        shutil.copyfile(hub_manifest, dst_manifest)
    elif not dst_manifest.exists():
        dst_manifest.write_text(json.dumps({"edition": "release-v2.1"}), encoding="utf-8")

    # excluded
    dst_excluded = suite_dir / "excluded-questions.json"
    hub_excluded = base_dir / "hub/excluded-questions.json"
    if not hub_excluded.exists():
        hub_excluded = base_dir / "hub/0.2.1/excluded-questions.json"
    if hub_excluded.exists() and not dst_excluded.exists():
        shutil.copyfile(hub_excluded, dst_excluded)
    elif not dst_excluded.exists():
        dst_excluded.write_text(json.dumps({"rows": []}), encoding="utf-8")

    print("[*] Arquivos da suite preparados com sucesso!")

    # 3. Executar o pipeline de pontuacao
    print("\n[*] Calculando pontuacao oficial do Decision Index...")
    try:
        from decision_index.suite.io import Suite
        from decision_index.pipeline import score_run

        suite = Suite(suite_dir, "0.2.1")
        scores = score_run(suite, results_path, "qwen-system-one", out_dir)

        print("\n" + "=" * 70)
        print("   RESULTADO OFICIAL DO JEV DECISION INDEX (0.2.1)               ")
        print("=" * 70)
        di = scores.get("decision_index")
        sc = scores.get("scores", {})
        print(f"   ★ Balanced Skill (Decision Index) : {di}")
        if isinstance(sc, dict):
            print(f"   ★ Balanced Raw                    : {sc.get('balanced_raw', 'N/A')}")
            print(f"   ★ Breadth Skill                   : {sc.get('breadth_skill', 'N/A')}")
        print(f"   Total de Questoes Avaliadas       : {scores.get('completed', 0):,}")
        print(f"   Status Completo                   : {scores.get('complete', False)}")
        print("\n   --- DESEMPENHO POR AREA ---")
        for area in scores.get("areas", []):
            print(f"   • {area.get('label', area.get('id')):28} : Skill = {area.get('skill', 'N/A')} | Raw = {area.get('raw', 'N/A')} (n={area.get('n', 0)})")
        print("=" * 70)

        # Salvar copia no Drive
        if drive_backup.exists():
            shutil.copyfile(out_dir / "scores.json", drive_backup / "scores.json")
            shutil.copyfile(out_dir / "index.json", drive_backup / "index.json")
            print(f"\n[+] Relatorios finais copiados para o Drive em: {drive_backup}")

    except Exception as e:
        import traceback
        print(f"[ERRO] Falha ao rodar score_run: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
