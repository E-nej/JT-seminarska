import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'

parser = argparse.ArgumentParser(description='Evaluate experiment outputs for Slovenian and Manchu')
parser.add_argument('--dirs', nargs='*', help='Explicit output directories to evaluate. If omitted, evaluates all experiments/out_* dirs.')
parser.add_argument('--start', type=int, default=0, help='First sentence index (inclusive)')
parser.add_argument('--end', type=int, default=None, help='Last sentence index (exclusive)')
parser.add_argument('--ref', type=str, default=None, help='Reference file to use for all evaluations')
parser.add_argument('--log', type=str, default='evaluate_both_logs', help='Log directory under experiments/')

args = parser.parse_args()

if args.dirs:
    out_dirs = [Path(d) if Path(d).is_absolute() else PROJECT_ROOT / d for d in args.dirs]
else:
    out_dirs = sorted(EXPERIMENTS_DIR.glob('out_*'))

if not out_dirs:
    raise SystemExit('No experiment output directories found. Provide --dirs or create experiments/out_*.')

log_dir = EXPERIMENTS_DIR / args.log
log_dir.mkdir(parents=True, exist_ok=True)


def infer_language(output_dir: Path) -> str:
    config_path = output_dir / 'config.json'
    if not config_path.exists():
        return 'unknown'

    try:
        with config_path.open('r', encoding='utf-8') as fh:
            config = json.load(fh)
    except Exception:
        return 'unknown'

    paths = []
    for key in ('input_fn', 'grammar_fn', 'dict_fn', 'gloss_fn', 'src_lang'):
        value = config.get(key)
        if isinstance(value, str):
            paths.append(value.lower())

    if any('data/slovenian' in p or '\\data\\slovenian' in p or 'slovenian' == p for p in paths):
        return 'slovenian'
    if any('data/manchu' in p or '\\data\\manchu' in p or 'manchu' == p for p in paths):
        return 'manchu'
    return 'unknown'


groups = {'slovenian': [], 'manchu': [], 'unknown': []}
for out_dir in out_dirs:
    if not out_dir.exists():
        print(f'Skipping missing directory: {out_dir}')
        continue
    lang = infer_language(out_dir)
    groups.setdefault(lang, []).append(out_dir)

for lang in ('slovenian', 'manchu'):
    group = groups.get(lang, [])
    if not group:
        print(f'\nNo {lang} outputs found to evaluate.')
        continue

    print(f'\n=== Evaluating {lang} outputs ({len(group)} dirs) ===')
    for index, out_dir in enumerate(sorted(group), start=1):
        cmd = [sys.executable, str(PROJECT_ROOT / 'main.py'), 'evaluate', str(out_dir)]
        if args.ref:
            cmd += ['--ref', str(Path(args.ref).resolve())]
        if args.start:
            cmd += ['--start', str(args.start)]
        if args.end is not None:
            cmd += ['--end', str(args.end)]

        print(f'[{index}/{len(group)}] Evaluating {out_dir.name}')
        print('Command:', ' '.join(cmd))
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        print('Exit code:', proc.returncode)

        with open(log_dir / f'{lang}_{out_dir.name}.txt', 'w', encoding='utf-8') as fh:
            fh.write(' '.join(str(x) for x in cmd) + '\n')
            fh.write(f'Exit code: {proc.returncode}\n')

        if proc.returncode != 0:
            print(f'Evaluation failed for {out_dir}, stopping.')
            raise SystemExit(proc.returncode)

print(f'\nEvaluation completed. Logs saved to {log_dir}')
