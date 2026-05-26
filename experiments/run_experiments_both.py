import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / 'experiments' / 'run_experiments.py'

parser = argparse.ArgumentParser(description='Run experiments for Slovenian first, then Manchu')
parser.add_argument('--llm', type=str, required=True, help='LLM model id to use')
parser.add_argument('--count', type=int, default=50, help='Number of sentences to process')
parser.add_argument('--extra', type=str, default='', help='Extra args to append to each call')
parser.add_argument('--log', action='store_true', help='Also print log file path for each run')

args = parser.parse_args()

calls = [
    {
        'name': 'slovenian',
        'args': [
            '--llm', args.llm,
            '--count', str(args.count),
        ],
    },
    {
        'name': 'manchu',
        'args': [
            '--llm', args.llm,
            '--count', str(args.count),
            '--work_dir', 'manchu',
            '--input_fn', 'laoqida.in',
            '--dict_name', 'manchu_dict_laoqida_new.db',
            '--demo', 'manchu.demo',
            '--grammar_fn', 'manchu_grammar_sum.md',
        ],
    },
]

total_stages = len(calls)

for index, stage in enumerate(calls, start=1):
    print(f"\n=== Stage {index}/{total_stages}: Running experiments for {stage['name']} ===")
    cmd = [sys.executable, str(SCRIPT)] + stage['args']
    if args.extra:
        cmd.extend(args.extra.split())

    print('Command:', ' '.join(cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    print(f'Stage {index}/{total_stages} finished with exit code {proc.returncode}')

    if args.log:
        print(f"Log directory: {PROJECT_ROOT / 'experiments' / 'logs'}")

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

print(f'\nAll {total_stages} runs completed successfully.')
