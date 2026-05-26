import argparse
import subprocess
import shlex
import os
import sys

parser = argparse.ArgumentParser(description='Run four Gemini experiment configurations')
parser.add_argument('--llm', type=str, required=True, help='LLM model id to use')
parser.add_argument('--count', type=int, default=50, help='Number of sentences to process')
parser.add_argument('--work_dir', type=str, default='slovenian')
parser.add_argument('--input_fn', type=str, default='flores.in')
parser.add_argument('--dict_name', type=str, default='slovenian_dict.db')
parser.add_argument('--demo', type=str, default='slovenian.demo')
parser.add_argument('--grammar_fn', type=str, default='slovenian_grammar_sum.md')
parser.add_argument('--extra', type=str, default='', help='Extra args to append to each call')

args = parser.parse_args()

base_cmd = (
    f'"{sys.executable}" main.py generate '
    f"--src slovenian --tgt english --pipeline dict_translate --work_dir {args.work_dir} "
    f"--input_fn {args.input_fn} --dict_name {args.dict_name} --demo {args.demo} --grammar_fn {args.grammar_fn} "
    f"--llm {args.llm} --count {args.count} "
)

configs = [
    ('baseline', ''),
    ('rag', '--use_rag --rag_k 3'),
    ('compression', '--use_compression --compression_target 1200'),
    ('rag_compression', '--use_rag --rag_k 3 --use_compression --compression_target 1200'),
]

os.makedirs('experiments/logs', exist_ok=True)

for name, flags in configs:
    out_dir = f"experiments/out_{name}_{args.count}"
    cmd = base_cmd + flags + ' ' + args.extra + f' --output_dir {out_dir}'
    print('\n=== Running:', name)
    print(cmd)
    process = subprocess.run(shlex.split(cmd), cwd=os.path.dirname(os.path.dirname(__file__)))
    print('Exit code:', process.returncode)
    # keep logs
    with open(f"experiments/logs/{name}.txt", 'w', encoding='utf-8') as fh:
        fh.write(f'Command: {cmd}\nExit code: {process.returncode}\n')

print('\nAll experiments finished. Results are in the experiments/out_* folders and experiments/logs/.')
