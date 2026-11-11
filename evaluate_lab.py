"""Run a repeatable baseline-versus-policy-RAG experiment without executing refunds.

Run from your local venv:
  python evaluate_lab.py --repeat 2
  python evaluate_lab.py --repeat 1 --limit 4

Each tool proposal pauses at HITL and is recorded, NOT approved or executed.
All test messages are fictional. LangSmith tracing is controlled by your .env.
"""
import argparse
import csv
import json
import os
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
FIELDS = [
    "case_id", "case_title", "variant", "repetition", "run_id", "run_name",
    "status", "category", "expected_policy", "retrieved_policies",
    "retrieval_hit", "refund_expected", "refund_proposed", "decision_match",
    "proposed_amount", "elapsed_seconds", "error",
]


def extract_actions(result):
    interruptions = result.get("__interrupt__", ()) if isinstance(result, dict) else ()
    proposed = []
    for interruption in interruptions:
        payload = getattr(interruption, "value", {})
        if isinstance(payload, dict):
            actions = payload.get("action_requests", [])
            if isinstance(actions, list):
                proposed.extend(actions)
    return proposed


def extract_policies(context):
    return [f"POL-{n:03}" for n in range(1, 11) if f"[POL-{n:03}]" in context]


def evaluate_one(module, variant, case, repetition):
    # A unique thread avoids checkpoint contamination between cases and attempts.
    run_id = str(uuid.uuid4())
    run_name = f"lab6_{variant}_{case['id']}_r{repetition}"
    config = {
        "configurable": {"thread_id": run_name},
        "run_id": uuid.UUID(run_id),
        "run_name": run_name,
        "tags": ["lab6", "extension-evaluation", variant, case["id"]],
        "metadata": {
            "variant": variant, "case_id": case["id"], "repetition": repetition,
            "model": "qwen2.5:3b",
        },
    }
    state = {
        "raw_message": case["message"], "redacted_summary": "",
        "category": "", "refund_outcome": "",
    }
    if variant == "rag":
        state["policy_context"] = ""

    start = time.perf_counter()
    error = ""
    try:
        output = module.support_graph.invoke(state, config=config)
        # For interrupted graphs, snapshot contains the nodes already completed.
        snapshot = module.support_graph.get_state(config).values
        actions = extract_actions(output)
        if not actions:
            # Future LangGraph versions can expose pending interrupt in tasks.
            saved = module.support_graph.get_state(config)
            for task in getattr(saved, "tasks", ()):
                for intr in getattr(task, "interrupts", ()):
                    payload = getattr(intr, "value", {})
                    if isinstance(payload, dict):
                        actions.extend(payload.get("action_requests", []))
        refund_actions = [a for a in actions if a.get("name") == "issue_refund"]
        proposed = bool(refund_actions)
        amounts = "|".join(str(a.get("args", {}).get("amount", "")) for a in refund_actions)
        policies = extract_policies(snapshot.get("policy_context", ""))
        status = "proposed_refund_paused" if proposed else "finished_no_refund"
        category = snapshot.get("category", "")
    except Exception as exc:
        status, category, policies, proposed, amounts = "error", "", [], False, ""
        error = f"{type(exc).__name__}: {str(exc)[:250]}"
    elapsed = round(time.perf_counter() - start, 2)
    expected = bool(case["refund_proposal_expected"])
    return {
        "case_id": case["id"], "case_title": case["title"], "variant": variant,
        "repetition": repetition, "run_id": run_id, "run_name": run_name,
        "status": status, "category": category,
        "expected_policy": case["expected_policy"],
        "retrieved_policies": "|".join(policies),
        "retrieval_hit": str(case["expected_policy"] in policies) if variant == "rag" and not error else "",
        "refund_expected": str(expected), "refund_proposed": str(proposed) if not error else "",
        "decision_match": str(proposed == expected) if not error else "",
        "proposed_amount": amounts, "elapsed_seconds": elapsed, "error": error,
    }


def summarize(rows, path):
    stats = {}
    for variant in ("baseline", "rag"):
        successful = [row for row in rows if row["variant"] == variant and not row["error"]]
        if successful:
            correct = sum(row["decision_match"] == "True" for row in successful)
            stats[variant] = {
                "completed": len(successful),
                "accuracy_on_refund_proposal_labels": round(correct / len(successful), 3),
                "mean_seconds": round(statistics.mean(float(row["elapsed_seconds"]) for row in successful), 2),
            }
            if variant == "rag":
                stats[variant]["retrieval_recall_at_3"] = round(
                    sum(row["retrieval_hit"] == "True" for row in successful) / len(successful), 3
                )
    output = path.with_suffix(".summary.json")
    output.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "qwen2.5:3b",
        "notes": [
            "Refund-proposal labels assume facts in fictional complaints; no transaction lookup is implemented.",
            "Only presence/absence of refund tool proposals is automatically graded, not response quality or refund amount correctness.",
            "Unsuccessful cases excluded from metrics; inspect CSV for failures.",
            "First run may be slower due to cold-start; inspect individual times.",
            "This experiment intentionally pauses HITL without executing or resuming payment actions.",
            "Copy public LangSmith share links manually after reviewing traces for synthetic data.",
        ],
        "metrics": stats,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary written: {output}")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=1, help="Repeats each case per variant (default 1)")
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N test cases (0 = all)")
    parser.add_argument("--variant", choices=["baseline", "rag", "both"], default="both")
    parser.add_argument("--output", default="evaluation_results.csv", help="Output CSV filename")
    args = parser.parse_args()
    if args.repeat < 1 or args.limit < 0:
        parser.error("--repeat must be >=1 and --limit must be >=0")

    print("LangSmith project:", os.environ.get("LANGSMITH_PROJECT", "default"))
    print("Tracing enabled:", os.environ.get("LANGSMITH_TRACING", "").lower() == "true")
    print("API key available:", bool(os.environ.get("LANGSMITH_API_KEY")))
    print("Safety: refund proposals are captured at HITL; no approvals or payments executed.")
    print("MODEL: qwen2.5:3b for both variants")
    import middlewareHITLMAS as baseline
    import middlewareHITLMAS_RAG as rag
    cases = json.loads((ROOT / "evaluation_cases.json").read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[:args.limit]

    variants = (("baseline", baseline), ("rag", rag)) if args.variant == "both" else (
        (args.variant, baseline if args.variant == "baseline" else rag),
    )
    rows = []
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for repetition in range(1, args.repeat + 1):
            for case in cases:
                for variant, module in variants:
                    print(f"\n[{len(rows)+1}/{len(cases)*len(variants)*args.repeat}] {variant} {case['id']} repeat {repetition}", flush=True)
                    row = evaluate_one(module, variant, case, repetition)
                    rows.append(row)
                    writer.writerow(row)
                    file.flush()
                    print(f"  status={row['status']} correct={row['decision_match']} "
                          f"retrieval={row['retrieved_policies']} time={row['elapsed_seconds']}s", flush=True)
    print(f"\nRaw results written: {destination}")
    summarize(rows, destination)


if __name__ == "__main__":
    main()
