"""
Evaluate translation outputs in a run directory against a reference file.

Usage:
    python eval.py <output_dir> [--ref <reference.out>] [--start N] [--end N]

If --ref is omitted the reference file is inferred from config.json
(replacing the .in extension with .out).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def load_outputs(output_dir: str, start: int, end: int | None) -> tuple[list[str], list[int]]:
    indices = sorted(
        (int(f.split("_")[1]) for f in os.listdir(output_dir) if f.startswith("output_") and f.split("_")[1].isdigit()),
    )
    if end is not None:
        indices = [i for i in indices if start <= i < end]
    else:
        indices = [i for i in indices if i >= start]

    translations, used = [], []
    for i in indices:
        path = os.path.join(output_dir, f"output_{i}")
        with open(path, encoding="utf-8") as f:
            translations.append(f.read().strip())
        used.append(i)
    return translations, used


def load_references(ref_file: str, indices: list[int]) -> list[str]:
    with open(ref_file, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    refs = []
    for i in indices:
        if i >= len(lines):
            print(f"Warning: reference file has no line {i}, skipping", file=sys.stderr)
            continue
        refs.append(lines[i])
    return refs


def infer_ref_file(output_dir: str) -> str:
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json in {output_dir}; provide --ref explicitly")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    input_fn = config.get("input_fn", "")
    ref_fn = input_fn.replace(".in", ".out")
    if not os.path.exists(ref_fn):
        raise FileNotFoundError(f"Inferred reference file not found: {ref_fn}")
    return ref_fn


def run(args):
    try:
        import sacrebleu
    except ImportError:
        print("sacrebleu not installed. Run: pip install sacrebleu", file=sys.stderr)
        sys.exit(1)

    ref_file = args.ref or infer_ref_file(args.output_dir)
    hypotheses, indices = load_outputs(args.output_dir, args.start, args.end)
    references = load_references(ref_file, indices)

    if len(hypotheses) != len(references):
        print(f"Warning: {len(hypotheses)} hypotheses vs {len(references)} references", file=sys.stderr)
        n = min(len(hypotheses), len(references))
        hypotheses, references = hypotheses[:n], references[:n]

    if not hypotheses:
        print("No outputs found.")
        return

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])

    print(f"{'Idx':>4}  {'chrF':>6}  {'Hypothesis':<50}  Reference")
    print("-" * 100)
    for idx, hyp, ref in zip(indices, hypotheses, references):
        sent_chrf = sacrebleu.sentence_chrf(hyp, [ref]).score
        print(f"{idx:>4}  {sent_chrf:>6.2f}  {hyp[:50]:<50}  {ref}")

    print(f"\nEvaluated {len(hypotheses)} sentences")
    print(f"Reference file : {ref_file}")
    print(f"Output dir     : {args.output_dir}")
    print()
    print(f"BLEU  : {bleu.score:.2f}")
    print(f"chrF  : {chrf.score:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM translation outputs")
    parser.add_argument("output_dir", help="Path to a run output directory")
    parser.add_argument("--ref", help="Reference .out file (inferred from config.json if omitted)")
    parser.add_argument("--start", type=int, default=0, help="First sentence index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Last sentence index (exclusive)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
