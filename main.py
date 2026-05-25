"""
LingoLLM unified CLI.

Commands:
  generate   Run a translation pipeline          (wraps gen.py)
  evaluate   Score outputs against a reference   (wraps eval.py)
  db init    Create PostgreSQL extension/table/index if absent
  db ingest  Embed a grammar file and store it in the DB
"""

import argparse
import os
import sys

_DB_URL = os.getenv("RAG_DB_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")

_INIT_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    id          BIGSERIAL PRIMARY KEY,
    language    TEXT NOT NULL,
    source      TEXT NOT NULL,
    chunk_index INT  NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
ON rag_chunks
USING hnsw (embedding vector_cosine_ops);
"""


# ---------------------------------------------------------------------------
# command handlers
# ---------------------------------------------------------------------------

def cmd_generate(args):
    from gen import run
    run(args)


def cmd_evaluate(args):
    from eval import run
    run(args)


def cmd_db_init(_):
    try:
        import psycopg
    except ImportError:
        print("psycopg not installed. Run: pip install psycopg[binary]", file=sys.stderr)
        sys.exit(1)

    try:
        with psycopg.connect(_DB_URL) as conn:
            with conn.cursor() as cur:
                for statement in _INIT_SQL.strip().split(";"):
                    statement = statement.strip()
                    if statement:
                        cur.execute(statement)
            conn.commit()
        print("Database initialised successfully.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_db_ingest(args):
    print(f"Ingesting '{args.grammar}' for language '{args.language}' ...")
    from lingollm.rag import ingest_grammar
    ingest_grammar(args.grammar, args.language)
    print("Done.")


# ---------------------------------------------------------------------------
# parser construction
# ---------------------------------------------------------------------------

def _add_generate_parser(sub):
    from lingollm.pipelines import PIPELINES
    p = sub.add_parser("generate", help="Run a translation pipeline")
    p.add_argument("--src",         required=True,  help="Source language")
    p.add_argument("--tgt",         required=True,  help="Target language")
    p.add_argument("--pipeline",    required=True,  choices=PIPELINES.keys(), help="Pipeline name")
    p.add_argument("--work_dir",    required=True,  help="Working directory name under data/")
    p.add_argument("--input_fn",    required=True,  help="Input file (e.g. laoqida.in)")
    p.add_argument("--dict_name",   required=True,  help="Dictionary file (e.g. manchu_dict.db)")
    p.add_argument("--llm",         required=True,  help="LLM model id")
    p.add_argument("--gloss_fn",    default=None,   help="Gloss file (defaults to input_fn)")
    p.add_argument("--grammar_fn",  default=None,   help="Grammar file")
    p.add_argument("--demo",        default="",     help="Demo examples file")
    p.add_argument("--output_dir",  default=None,   help="Output directory name (auto-generated if omitted)")
    p.add_argument("--copy_prompt", default="",     help="Output dir to copy prompts from")
    p.add_argument("--iter",        default=None, type=int)
    p.add_argument("--start",       default=0,    type=int, help="Resume from sentence N")
    p.add_argument("--use_rag",     action="store_true")
    p.add_argument("--rag_k",       default=3,    type=int)
    p.add_argument("--use_compression", action="store_true")
    p.add_argument("--compression_target", default=1200, type=int)
    p.set_defaults(func=cmd_generate)


def _add_evaluate_parser(sub):
    p = sub.add_parser("evaluate", help="Score translation outputs against a reference file")
    p.add_argument("output_dir",  help="Path to a run output directory")
    p.add_argument("--ref",       help="Reference .out file (inferred from config.json if omitted)")
    p.add_argument("--start",     default=0,    type=int, help="First sentence index (inclusive)")
    p.add_argument("--end",       default=None, type=int, help="Last sentence index (exclusive)")
    p.set_defaults(func=cmd_evaluate)


def _add_db_parser(sub):
    db = sub.add_parser("db", help="Manage the RAG PostgreSQL database")
    db_sub = db.add_subparsers(dest="db_command", required=True)

    init = db_sub.add_parser("init", help="Create extension, table, and index if they don't exist")
    init.set_defaults(func=cmd_db_init)

    ingest = db_sub.add_parser("ingest", help="Embed a grammar file and store chunks in the DB")
    ingest.add_argument("--grammar",  required=True, help="Path to the grammar file")
    ingest.add_argument("--language", required=True, help="Language identifier (e.g. manchu)")
    ingest.set_defaults(func=cmd_db_ingest)


def build_parser():
    root = argparse.ArgumentParser(
        prog="main.py",
        description="LingoLLM — translate, evaluate, manage RAG database",
    )
    sub = root.add_subparsers(dest="command", required=True)
    _add_generate_parser(sub)
    _add_evaluate_parser(sub)
    _add_db_parser(sub)
    return root


# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
