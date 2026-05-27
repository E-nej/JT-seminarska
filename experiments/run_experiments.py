import argparse
import subprocess
import shlex
import os
import sys

parser = argparse.ArgumentParser(description='Run four experiment configurations')
parser.add_argument('--llm', type=str, default='gemini-3.1-flash-lite', help='LLM model id to use')
parser.add_argument('--count', type=int, default=None, help='Number of sentences to process (single value)')
parser.add_argument('--counts', nargs='*', help='List of counts to run if --count is omitted (e.g. 1 3 30 50)')
parser.add_argument('--pipeline', type=str, default='dict_translate', help='Pipeline to use for generation')
parser.add_argument('--lang', choices=['slovenian', 'manchu'], default='slovenian', help='Language to run experiments for')
parser.add_argument('--work_dir', type=str, default=None, help='Working directory under data/ (overrides --lang defaults)')
parser.add_argument('--src', type=str, default=None, help='Source language (overrides --lang defaults)')
parser.add_argument('--tgt', type=str, default='english', help='Target language')
parser.add_argument('--input_fn', type=str, default=None, help='Input filename under data/<work_dir>')
parser.add_argument('--dict_name', type=str, default=None, help='Dictionary filename under data/<work_dir>')
parser.add_argument('--demo', type=str, default=None, help='Demo examples file under data/<work_dir>')
parser.add_argument('--grammar_fn', type=str, default=None, help='Grammar filename under data/<work_dir>')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory prefix for experiment folders')
parser.add_argument('--extra', type=str, default='', help='Extra args to append to each call')

args = parser.parse_args()

DEFAULT_COUNTS = [1, 3, 30, 50]
if args.lang == 'slovenian':
    if args.work_dir is None:
        args.work_dir = 'slovenian'
    if args.src is None:
        args.src = 'slovenian'
    if args.input_fn is None:
        args.input_fn = 'flores.in'
    if args.dict_name is None:
        args.dict_name = 'slovenian_dict.db'
    if args.demo is None:
        args.demo = 'slovenian.demo'
    if args.grammar_fn is None:
        args.grammar_fn = 'slovenian_grammar_sum.md'
else:
    if args.work_dir is None:
        args.work_dir = 'manchu'
    if args.src is None:
        args.src = 'manchu'
    if args.input_fn is None:
        args.input_fn = 'laoqida.in'
    if args.dict_name is None:
        args.dict_name = 'manchu_dict_laoqida_new.db'
    if args.demo is None:
        args.demo = 'manchu.demo'
    if args.grammar_fn is None:
        args.grammar_fn = 'manchu_grammar_sum.md'

if args.count is not None:
    counts = [args.count]
elif args.counts:
    try:
        counts = [int(x) for x in args.counts]
    except Exception:
        counts = DEFAULT_COUNTS
else:
    counts = DEFAULT_COUNTS


configs = [
    ('baseline', ''),
    ('rag', '--use_rag --rag_k 3'),
    ('compression', '--use_compression --compression_target 1200'),
    ('rag_compression', '--use_rag --rag_k 3 --use_compression --compression_target 1200'),
]

os.makedirs('experiments/logs', exist_ok=True)

output_prefix = args.output_dir or f'experiments/out_{args.lang}'

for count in counts:
    base_cmd = (
        f'"{sys.executable}" main.py generate '
        f"--src {args.src} --tgt {args.tgt} --pipeline {args.pipeline} --work_dir {args.work_dir} "
        f"--input_fn {args.input_fn} --dict_name {args.dict_name} --demo {args.demo} --grammar_fn {args.grammar_fn} "
        f"--llm {args.llm} --count {count} "
    )

    for name, flags in configs:
        out_dir = f"{output_prefix}_{name}_{count}"
        cmd = base_cmd + flags + ' ' + args.extra + f' --output_dir {out_dir}'
        print('\n=== Running:', args.lang, name, f'(count={count})')
        print(cmd)
        process = subprocess.run(shlex.split(cmd), cwd=os.path.dirname(os.path.dirname(__file__)))
        print('Exit code:', process.returncode)
        # ohrani logs
        log_fn = f"experiments/logs/{args.lang}_{name}_{count}.txt"
        with open(log_fn, 'w', encoding='utf-8') as fh:
            fh.write(f'Command: {cmd}\nExit code: {process.returncode}\n')

print('\nAll experiments finished. Results are in the experiments/out_* folders and experiments/logs/.')
