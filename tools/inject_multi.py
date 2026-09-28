#!/usr/bin/env python3
"""inject_multi.py — injeta multi_typed.jsonl na mixture_full.pkl no formato interno Decider.

Uso (Colab, célula 4, ANTES do treino):
    python inject_multi.py --mixture data/mixture_full.pkl --jsonl /content/work/multi_typed.jsonl \
        --out data/mixture_full.pkl --task minds14_multi

Converte cada linha {prompt, options, label} em Example(context=state, qs=[Q(text=head, options, gold=label)]),
usando o TEMPLATE reverso (extrai state/head do prompt) ou, preferível, regenera do parquet.
Aqui: parse simples do prompt gerado pelo make_multi_typed (formato conhecido).
"""
import argparse
import json
import pickle
import re
import sys

sys.path.insert(0, ".")


def parse_prompt(prompt):
    """Extrai (head, options, state) do prompt no formato make_multi_typed."""
    m = re.search(r"Question: (.*?)\nOptions:\n(.*?)<\|im_end\|>\n(.*?)<\|im_start\|>assistant", prompt, re.S)
    if not m:
        return None, None, None
    head, opts_block, state_block = m.group(1), m.group(2), m.group(3)
    options = []
    for line in opts_block.strip().split("\n"):
        mm = re.match(r"^[A-Z]\.\s+(.*)$", line.strip())
        if mm:
            options.append(mm.group(1))
    sm = re.search(r"<state>(.*?)</state>", state_block, re.S)
    state = sm.group(1) if sm else state_block.strip()
    return head, options, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="minds14_multi")
    args = ap.parse_args()

    from decider import data as D

    mix = pickle.load(open(args.mixture, "rb"))
    print("mixture keys:", list(mix.keys()) if isinstance(mix, dict) else type(mix))

    # A mixture é dict {split_name: [Example]} ou lista — descobre e injeta no train.
    if isinstance(mix, dict):
        train_key = next((k for k in mix if "train" in k.lower()), list(mix.keys())[0])
        train = mix[train_key]
    else:
        train, train_key = mix, None

    added = 0
    for line in open(args.jsonl, encoding="utf-8"):
        e = json.loads(line)
        head, options, state = parse_prompt(e["prompt"])
        if not head or not options:
            continue
        gold = e["label"]
        q = D.Q(head, options, gold)
        train.append(D.Example(state, [q], args.task))
        added += 1

    if train_key:
        mix[train_key] = train
    else:
        mix = train
    pickle.dump(mix, open(args.out, "wb"))
    print(f"injetados: {added} exemplos task={args.task} -> {args.out}")


if __name__ == "__main__":
    main()
