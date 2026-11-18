"""Command line: python -m internmate.cli <command>"""
import argparse
import json
import logging
import time

from internmate.db import SessionLocal, init_db
from internmate.embeddings import get_embedder
from internmate.ingestion import SOURCES, run_ingestion


def cmd_ingest(args):
    while True:
        with SessionLocal() as s:
            run = run_ingestion(s, get_embedder(), args.source)
            print(f"inserted={run.inserted} updated={run.updated} unchanged={run.unchanged} "
                  f"expired={run.expired} errors={run.errors}")
        if not args.every_hours:
            return
        time.sleep(args.every_hours * 3600)


def cmd_search(args):
    from internmate.recommender import recommend
    from internmate.search import SearchFilters

    with SessionLocal() as s:
        filters = SearchFilters(remote_only=args.remote, location=args.location)
        for r in recommend(s, get_embedder(), args.query, args.user, filters, args.k):
            d = r.to_dict()
            print(f"[{d['job_id']}] {d['score']:.3f}  {d['title']} @ {d['company']} ({d['location']})"
                  f"  missing={d['missing_skills']}")


def cmd_chat(args):
    from internmate.agent import CareerAgent
    from internmate.llm import get_llm

    agent = CareerAgent(get_llm(), get_embedder())
    print("InternMate - type 'exit' to quit")
    while (msg := input("\nyou> ").strip()).lower() not in {"exit", "quit"}:
        if not msg:
            continue
        reply = agent.chat(args.user, msg)
        print(f"\ninternmate> {reply.message}")
        for c in reply.recommendations:
            print(f"  - [{c.job_id}] {c.title} @ {c.company} ({c.match_score}/100): {c.why_matched}")


def cmd_eval(args):
    from internmate.evaluation import run_eval, to_markdown

    judge = None
    if args.judge:
        from internmate.llm import get_llm
        judge = get_llm()
    with SessionLocal() as s:
        result = run_eval(s, get_embedder(), judge_llm=judge)
    print(to_markdown(result))
    with open("evals/results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\nSaved evals/results.json")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="internmate")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ingest", help="refresh the job index")
    sp.add_argument("--source", nargs="+", default=["sample"], choices=sorted(SOURCES))
    sp.add_argument("--every-hours", type=float, default=0, help="keep running, refreshing every N hours")
    sp.set_defaults(fn=cmd_ingest)

    sp = sub.add_parser("search", help="ranked search without the LLM")
    sp.add_argument("query")
    sp.add_argument("--user")
    sp.add_argument("--remote", action="store_true")
    sp.add_argument("--location")
    sp.add_argument("-k", type=int, default=10)
    sp.set_defaults(fn=cmd_search)

    sp = sub.add_parser("chat", help="chat with the agent (needs GOOGLE_API_KEY)")
    sp.add_argument("--user", default="demo")
    sp.set_defaults(fn=cmd_chat)

    sp = sub.add_parser("eval", help="evaluate retrieval quality on evals/queries.json")
    sp.add_argument("--judge", action="store_true", help="also run LLM-as-judge (needs GOOGLE_API_KEY)")
    sp.set_defaults(fn=cmd_eval)

    args = p.parse_args()
    init_db()
    args.fn(args)


if __name__ == "__main__":
    main()
