import argparse
from lingollm.pipelines import PIPELINES
import os
from datetime import datetime
import json
import shutil
import glob
import time
from tqdm import tqdm
from lingollm.llms import get_llm_wrapper

parser = argparse.ArgumentParser(description='Generate translation')
parser.add_argument('--src', type=str, help='source language', required=True)
parser.add_argument('--tgt', type=str, help='target language', required=True)
parser.add_argument('--gloss_fn', type=str, help='gloss', required=False)
parser.add_argument('--pipeline', choices=PIPELINES.keys(), help='generation pipeline', required=True)
parser.add_argument('--work_dir', type=str, help='working directory, like `gitksan`', required=True)
parser.add_argument('--input_fn', type=str, help='input filename, like `dev.in`', required=True)
parser.add_argument('--dict_name', type=str, help='dictionary filename, like `gitksan_dict.db`', required=True)
parser.add_argument('--output_dir', type=str, default=None, help='output directory, like `direct`')
parser.add_argument('--grammar_fn', type=str, default=None, help='grammar file name')
parser.add_argument('--iter', default=None, type=int, help='iteration number')
parser.add_argument('--demo', type=str, default="", help='demo examples', required=False)
parser.add_argument("--llm", type=str, default="", help="LLM model id", required=True)
parser.add_argument("--start", type=int, default=0, help="Start from line")
parser.add_argument("--count", type=int, default=None, help="Number of sentences to process (limit runtime)")
parser.add_argument("--copy_prompt", type=str, default="", help="the output directory to copy prompts from", required=False)
parser.add_argument("--use_rag", action="store_true")
parser.add_argument("--rag_k", type=int, default=3)
parser.add_argument("--use_compression", action="store_true")
parser.add_argument("--compression_target", type=int, default=1200)


def check_dirs(args):
    work_dir = args.work_dir
    input_fn = args.input_fn
    dict_fn = args.dict_name
    output_dir = args.output_dir
    pipeline_name = args.pipeline
    src_lang = args.src
    tgt_lang = args.tgt
    
    if not os.path.exists(f'data/{work_dir}'):
        print(f"Working directory data/{work_dir} does not exist!")
        exit(1)
    
    if not os.path.exists(f'data/{work_dir}/{input_fn}'):
        print(f"Input file data/{work_dir}/{input_fn} does not exist!")
        exit(1)
    
    # if not os.path.exists(f'data/{work_dir}/{dict_fn}'):
    #     print(f"Dictionary data/{work_dir}/{dict_fn} does not exist!")
    #     exit(1)
    
    if output_dir is None:
        output_dir = pipeline_name
        if output_dir.endswith('_translate'):
            output_dir = output_dir[:-10]
        # time stamping the output
        now = datetime.now()
        formatted_date = now.strftime("%h%d_%H%M_%S")
        output_dir += f'_{formatted_date}'
        
    return output_dir


def make_logs(src_lang, tgt_lang, pipeline_name, input_fn, dict_fn, output_dir, gloss_fn, grammar_fn, iter, demo_fn, llm, copy_prompt):
    config = {
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'pipeline_name': pipeline_name,
        'input_fn': input_fn,
        'dict_fn': dict_fn,
        'gloss_fn': gloss_fn,
        "grammar_fn": grammar_fn,
        'iter': iter,
        'llm': llm,
        'copy_prompt': copy_prompt,
    }
    
    json.dump(config, open(f'{output_dir}/config.json', 'w'))
    
    os.makedirs(f'{output_dir}/code_bak', exist_ok=True)
    
    # # save code backup
    # for fn in glob.glob('lingollm/*.py'):
    #     shutil.copy(fn, f'{output_dir}/code_bak/{fn.split("/")[-1]}')
        
    if os.path.exists(demo_fn):
        shutil.copy(demo_fn, f'{output_dir}/code_bak/{demo_fn.split("/")[-1]}')


def _collect_stats(idx, elapsed, llm):
    call_history = getattr(llm, 'call_history', [])
    compression_stats = getattr(llm, 'compression_stats', None)

    total_input = sum(c.get('input_tokens') or 0 for c in call_history)
    total_output = sum(c.get('output_tokens') or 0 for c in call_history)

    stats = {
        "sentence_idx": idx,
        "total_latency_s": round(elapsed, 3),
        "total_input_tokens": total_input if total_input else None,
        "total_output_tokens": total_output if total_output else None,
        "llm_calls": call_history,
    }
    if compression_stats:
        stats["compression"] = compression_stats
    return stats


def _write_summary(output_dir, all_stats, start_idx):
    if not all_stats:
        return

    n = len(all_stats)
    total_latency = sum(s["total_latency_s"] for s in all_stats)

    input_tokens = [s["total_input_tokens"] for s in all_stats if s.get("total_input_tokens")]
    output_tokens = [s["total_output_tokens"] for s in all_stats if s.get("total_output_tokens")]

    summary = {
        "sentences_processed": n,
        "start_idx": start_idx,
        "total_latency_s": round(total_latency, 3),
        "avg_latency_s": round(total_latency / n, 3),
    }

    if input_tokens:
        summary["total_input_tokens"] = sum(input_tokens)
        summary["avg_input_tokens"] = round(sum(input_tokens) / len(input_tokens), 1)
    if output_tokens:
        summary["total_output_tokens"] = sum(output_tokens)
        summary["avg_output_tokens"] = round(sum(output_tokens) / len(output_tokens), 1)

    comp_list = [s["compression"] for s in all_stats if s.get("compression")]
    if comp_list:
        summary["compression"] = {
            "avg_original_tokens": round(sum(c["original_tokens"] for c in comp_list) / len(comp_list), 1),
            "avg_compressed_tokens": round(sum(c["compressed_tokens"] for c in comp_list) / len(comp_list), 1),
            "avg_ratio": round(sum(c["ratio"] for c in comp_list if c.get("ratio")) / len(comp_list), 3),
        }

    json.dump(summary, open(f'{output_dir}/stats_summary.json', 'w'), indent=2)


def run(args):
    work_dir = args.work_dir
    input_fn = args.input_fn
    pipeline_name = args.pipeline
    src_lang = args.src
    tgt_lang = args.tgt
    start = args.start
    count = args.count
    use_rag = args.use_rag
    rag_k = args.rag_k
    use_compression = args.use_compression
    compression_target = args.compression_target
    output_dir = check_dirs(args)

    if args.gloss_fn is None:
        args.gloss_fn = input_fn

    work_dir = f"data/{work_dir}"
    input_fn = f"{work_dir}/{input_fn}"
    dict_fn = f"{work_dir}/{args.dict_name}"
    output_dir = f"{work_dir}/outputs/{output_dir}"
    gloss_fn = f"{work_dir}/{args.gloss_fn}"
    demo_fn = f"{work_dir}/{args.demo}"
    grammar_fn = ""
    copy_prompt = args.copy_prompt or None
    if args.grammar_fn:
        grammar_fn = f"{work_dir}/{args.grammar_fn}"

    print(f"OUTPUT_DIR: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    llm_id = args.llm
    make_logs(src_lang, tgt_lang, pipeline_name, input_fn, dict_fn, output_dir, gloss_fn, grammar_fn, args.iter, demo_fn, llm_id, copy_prompt)

    pipeline = PIPELINES[pipeline_name]
    llm = get_llm_wrapper(llm_id)

    grammar = "[]" if grammar_fn == "" else open(grammar_fn, 'r', encoding='utf-8').read()
    demo = "" if demo_fn == "" else open(demo_fn, 'r', encoding='utf-8').read()
    if grammar_fn.endswith('.json'):
        grammar = json.loads(grammar)

    all_stats = []
    with open(input_fn, 'r', encoding='utf-8') as f:
        with open(gloss_fn, 'r', encoding='utf-8') as g:
            for i, (sent, gloss) in tqdm(enumerate(zip(f, g))):
                if i < start:
                    continue
                if count is not None and i >= start + count:
                    break
                sent = sent.strip()
                if sent == '':
                    continue
                history = []
                if copy_prompt:
                    with open(f'{work_dir}/outputs/{copy_prompt}/history_{i}.json', 'r') as hf:
                        history = json.load(hf)[:2]

                llm.reset_stats()
                t0 = time.time()
                try:
                    res, messages = pipeline(llm, history, src_lang, tgt_lang, sent, dict_fn, gloss, demo, grammar, args.iter, use_rag, rag_k, use_compression, compression_target)
                except RuntimeError as exc:
                    print(f"\nGeneration stopped at line {i}: {exc}")
                    raise SystemExit(1)
                elapsed = time.time() - t0

                with open(f'{output_dir}/output_{i}', 'w') as out:
                    out.write(res)
                with open(f'{output_dir}/history_{i}.json', 'w') as out:
                    out.write(json.dumps(messages, indent=2, ensure_ascii=False))
                    out.write('\n')

                stats = _collect_stats(i, elapsed, llm)
                all_stats.append(stats)
                with open(f'{output_dir}/stats_{i}.json', 'w') as out:
                    json.dump(stats, out, indent=2)
                    out.write('\n')

    _write_summary(output_dir, all_stats, start)

    if os.path.exists(dict_fn):
        shutil.copy(dict_fn, f'{output_dir}/code_bak/{dict_fn.split("/")[-1]}')

    return output_dir


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)