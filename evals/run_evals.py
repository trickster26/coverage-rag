"""Evaluate the assistant on evals/golden.jsonl.

Always runs (no API keys needed):
  - retrieval: recall@k and MRR, judged by whether a retrieved chunk from the
    expected document contains the expected text
  - refusal: out-of-scope questions must be refused by the evidence gate
  - redaction: identifiers in the question must be masked

With LLM_PROVIDER set, also scores the generated answers:
  - must_contain facts present, citations valid, no refusal on answerable questions

Usage:  python -m evals.run_evals [--min-recall 0.9] [--json out.json]
Exits non-zero when a threshold is missed, so CI can gate on it.
"""

import argparse
import json
import sys
from pathlib import Path

from coverage_rag.rag import default_assistant

GOLDEN = Path(__file__).with_name("golden.jsonl")


def matches(source, case) -> bool:
    return source.id.startswith(case["expect_source"]) and case["expect_text"].lower() in source.text.lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-recall", type=float, default=0.9)
    ap.add_argument("--min-refusal", type=float, default=1.0)
    ap.add_argument("--min-answer", type=float, default=0.8)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    assistant = default_assistant()
    cases = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    rows, rr, hits, refusals, oos, answered, answer_ok, redact_ok, redact_n = [], [], 0, 0, 0, 0, 0, 0, 0

    for case in cases:
        result = assistant.ask(case["question"], case.get("jurisdiction"))
        row = {"id": case["id"], "refused": result.refused}

        if case.get("expect_refusal"):
            oos += 1
            refusals += result.refused
            row["pass"] = result.refused
        else:
            rank = next((s.n for s in result.sources if matches(s, case)), None)
            hits += rank is not None
            rr.append(1 / rank if rank else 0.0)
            row.update(rank=rank, pass_retrieval=rank is not None)
            if assistant.llm is not None:
                answered += 1
                text = result.answer.lower()
                ok = (not result.refused and not result.invalid_citations
                      and all(f.lower() in text for f in case.get("must_contain", [])) and "[" in text)
                answer_ok += ok
                row.update(pass_answer=ok, answer=result.answer)

        if "expect_redactions" in case:
            redact_n += 1
            got = set(result.redactions)
            row["redactions"] = sorted(got)
            redact_ok += set(case["expect_redactions"]) <= got
        rows.append(row)

    n_answerable = len(rr)
    summary = {
        "recall_at_k": hits / n_answerable,
        "mrr": sum(rr) / n_answerable,
        "refusal_rate_out_of_scope": refusals / oos if oos else None,
        "redaction_pass": redact_ok / redact_n if redact_n else None,
        "answer_pass": answer_ok / answered if answered else None,
        "k": assistant.s.top_k,
        "cases": len(cases),
    }
    for row in rows:
        flag = "ok " if row.get("pass", row.get("pass_retrieval")) and row.get("pass_answer", True) else "FAIL"
        print(f"{flag} {row['id']:<28} rank={row.get('rank')} refused={row['refused']}"
              + (f" answer={'ok' if row.get('pass_answer') else 'FAIL'}" if "pass_answer" in row else ""))
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    failed = summary["recall_at_k"] < args.min_recall
    failed |= summary["refusal_rate_out_of_scope"] is not None and summary["refusal_rate_out_of_scope"] < args.min_refusal
    failed |= summary["redaction_pass"] is not None and summary["redaction_pass"] < 1.0
    failed |= summary["answer_pass"] is not None and summary["answer_pass"] < args.min_answer
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
