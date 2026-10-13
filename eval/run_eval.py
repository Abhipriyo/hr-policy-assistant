"""Measure how well the assistant finds and answers policy questions.

Two levels:
1. Retrieval (always, no API key needed): is the page that holds the answer among the
   top results? Reported as hit@1, hit@3, hit@5 and MRR, plus how often the evidence
   gate correctly refuses unanswerable questions.
2. Answers (--answers): runs the full pipeline and checks each answer contains the
   expected fact, cites the right page, and refuses when it should. Uses the LLM if
   LLM_API_KEY is set, otherwise the extractive fallback.

Examples:
    python eval/run_eval.py                      # hybrid search, default chunk size
    python eval/run_eval.py --no-dense           # keyword search only
    python eval/run_eval.py --chunk-words 60     # compare chunk sizes
    python eval/run_eval.py --answers --save eval/results/groq_hybrid.json
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from policy_assistant.config import Settings
from policy_assistant.ingest import load_directory
from policy_assistant.pipeline import PolicyAssistant
from policy_assistant.retrieval import has_enough_evidence

HERE = Path(__file__).resolve().parent


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank_of_answer_page(hits, q) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if h.chunk.doc_id == q["doc"] and h.chunk.page == q["page"]:
            return rank
    return None


def evaluate(assistant: PolicyAssistant, questions: list[dict], check_answers: bool) -> dict:
    s = assistant.settings
    answerable = [q for q in questions if q["doc"]]
    unanswerable = [q for q in questions if not q["doc"]]
    rows, ranks = [], []

    for q in questions:
        hits = assistant.retriever.search(q["question"], 5)
        passed_gate, stats = has_enough_evidence(q["question"], hits, s.min_keyword_coverage,
                                                 s.min_dense_similarity,
                                                 assistant.retriever.idf)
        row = {"id": q["id"], "question": q["question"], "passed_gate": passed_gate, **stats}
        if q["doc"]:
            row["rank"] = rank_of_answer_page(hits, q)
            ranks.append(row["rank"])
        if check_answers:
            ans = assistant.ask(q["question"])
            row.update(status=ans.status, mode=ans.mode, answer=ans.answer,
                       latency_ms=ans.latency_ms,
                       tokens=ans.prompt_tokens + ans.completion_tokens)
            if q["doc"]:
                row["fact_ok"] = any(e.lower() in ans.answer.lower() for e in q["expect"])
                row["citation_ok"] = any(c.doc_title and c.page == q["page"] and
                                         c.doc_title == _title(assistant, q["doc"])
                                         for c in ans.citations)
            else:
                row["refused_ok"] = ans.status == "not_found"
        rows.append(row)

    n = len(ranks)
    summary = {
        "retrieval_mode": assistant.retriever.mode,
        "chunk_words": s.chunk_words,
        "answerable_questions": n,
        "unanswerable_questions": len(unanswerable),
        "hit@1": sum(r == 1 for r in ranks if r) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "hit@5": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "gate_false_refusals": sum(not r["passed_gate"] for r in rows if r["id"] in
                                   {q["id"] for q in answerable}) / n,
        "gate_correct_refusals": (sum(not r["passed_gate"] for r in rows if r["id"] in
                                      {q["id"] for q in unanswerable}) / len(unanswerable)
                                  if unanswerable else None),
    }
    if check_answers:
        ans_rows = [r for r in rows if "fact_ok" in r]
        ref_rows = [r for r in rows if "refused_ok" in r]
        summary.update(
            answer_mode=rows[0].get("mode") or "extractive",
            fact_accuracy=sum(r["fact_ok"] for r in ans_rows) / len(ans_rows),
            citation_accuracy=sum(r["citation_ok"] for r in ans_rows) / len(ans_rows),
            refusal_accuracy=(sum(r["refused_ok"] for r in ref_rows) / len(ref_rows)
                              if ref_rows else None),
            median_latency_ms=sorted(r["latency_ms"] for r in rows)[len(rows) // 2],
            total_tokens=sum(r["tokens"] for r in rows),
        )
    return {"summary": summary, "rows": rows}


def sweep_gate(assistant: PolicyAssistant, questions: list[dict]) -> None:
    """Show how each evidence-gate threshold trades missed answers against wrong answers.

    A false refusal blocks a question the documents DO answer. A correct refusal stops an
    off-topic question before it reaches the LLM. Pick the threshold with the best balance.
    """
    s = assistant.settings
    searched = [(q, assistant.retriever.search(q["question"], 5)) for q in questions]
    n_ans = sum(1 for q, _ in searched if q["doc"])
    n_un = len(searched) - n_ans

    def report(label, kw, dense):
        false_ref = correct_ref = 0
        for q, hits in searched:
            ok, _ = has_enough_evidence(q["question"], hits, kw, dense, assistant.retriever.idf)
            if q["doc"] and not ok:
                false_ref += 1
            if not q["doc"] and not ok:
                correct_ref += 1
        print(f"| {label} | {false_ref / n_ans:.2f} | {correct_ref / n_un:.2f} |")

    print("\n| Threshold | False refusals (answerable) | Correct refusals (off-topic) |")
    print("| --- | --- | --- |")
    for kw in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4):
        report(f"keyword {kw}", kw, 9.0)          # 9.0 = semantic route switched off
    if assistant.retriever.dense:
        for d in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
            report(f"semantic {d}", 9.0, d)       # 9.0 = keyword route switched off


def _title(assistant: PolicyAssistant, doc_id: str) -> str:
    return next((d["title"] for d in assistant.documents if d["doc_id"] == doc_id), "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    parser.add_argument("--no-dense", action="store_true", help="keyword search only")
    parser.add_argument("--chunk-words", type=int, default=None)
    parser.add_argument("--answers", action="store_true", help="also evaluate full answers")
    parser.add_argument("--sweep-gate", action="store_true",
                        help="print evidence-gate results for a range of thresholds")
    parser.add_argument("--save", type=Path, help="write detailed results to this JSON file")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.no_dense:
        settings = replace(settings, use_dense=False)
    if args.chunk_words:
        settings = replace(settings, chunk_words=args.chunk_words)
    chunks = load_directory(settings.policies_dir, settings.chunk_words, settings.chunk_overlap_words)
    assistant = PolicyAssistant(settings, chunks=chunks)

    result = evaluate(assistant, load_questions(args.questions), args.answers)
    print("| Metric | Value |\n| --- | --- |")
    for key, value in result["summary"].items():
        print(f"| {key} | {value:.2f} |" if isinstance(value, float) else f"| {key} | {value} |")

    misses = [r for r in result["rows"] if "rank" in r and r["rank"] != 1]
    if misses:
        print("\nAnswer page not ranked first:")
        for r in misses:
            print(f"  {r['id']} rank={r['rank']}  {r['question']}")
    if args.sweep_gate:
        sweep_gate(assistant, load_questions(args.questions))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved details to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
