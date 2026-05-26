import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'

parser = argparse.ArgumentParser(description='Evaluate all experiment outputs')
parser.add_argument('--dirs', nargs='*', help='Output directories to evaluate. If omitted, evaluates experiments/out_*')
parser.add_argument('--start', type=int, default=0, help='First sentence index (inclusive)')
parser.add_argument('--end', type=int, default=None, help='Last sentence index (exclusive)')
parser.add_argument('--ref', type=str, default=None, help='Reference file to use for all evaluations')
parser.add_argument('--log', type=str, default='evaluate_logs', help='Log directory under experiments/')

args = parser.parse_args()

if args.dirs:
    out_dirs = [Path(d) if Path(d).is_absolute() else PROJECT_ROOT / d for d in args.dirs]
else:
    out_dirs = sorted(EXPERIMENTS_DIR.glob('out_*'))

if not out_dirs:
    raise SystemExit('No experiment output directories found. Provide --dirs or create experiments/out_*.')

log_dir = EXPERIMENTS_DIR / args.log
log_dir.mkdir(parents=True, exist_ok=True)

for out_dir in out_dirs:
    if not out_dir.exists():
        print(f'Skipping missing directory: {out_dir}')
        continue

    cmd = [sys.executable, str(PROJECT_ROOT / 'main.py'), 'evaluate', str(out_dir)]
    if args.ref:
        cmd += ['--ref', str(Path(args.ref).resolve())]
    if args.start:
        cmd += ['--start', str(args.start)]
    if args.end is not None:
        cmd += ['--end', str(args.end)]

    name = out_dir.name
    print(f'\n=== Evaluating {name} ===')
    print(' '.join(str(x) for x in cmd))

    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    print(f'Exit code: {proc.returncode}')

    with open(log_dir / f'{name}.txt', 'w', encoding='utf-8') as fh:
        fh.write(' '.join(str(x) for x in cmd) + '\n')
        fh.write(f'Exit code: {proc.returncode}\n')

    if proc.returncode != 0:
        print(f'Evaluation failed for {name}, stopping.')
        raise SystemExit(proc.returncode)

print(f'\nEvaluation completed. Logs saved to {log_dir}')
