# IS 590R Lab 6 — Middleware, HITL, and policy RAG extension

## What this branch changes

**Baseline preserved:** `middlewareHITLMAS.py` is unchanged from the professor's original lab. It uses Qwen2.5 3B, sanitizes intake, classifies requests, and proposes refunds through a human-in-the-loop (HITL) tool.

**Extension:** `middlewareHITLMAS_RAG.py` inserts `retrieve_policy_node` between triage and the billing refund agent. The node searches **10 general, fictional company refund policies** in `refund_policy.json` and passes up to three matching policy passages to the refund agent. The model's prompt no longer contains hardcoded dollar-amount examples. Original PII and HITL architecture is retained, including its privacy limitations.

```text
Baseline: START → sanitize_intake → triage → (BILLING?) → refund agent / HITL → END
Extension: START → sanitize_intake → triage → (BILLING?) → retrieve_policy → refund agent / HITL → END
```

Retrieval is **keyword-based RAG**, not semantic vector search. This isolates the benefit of providing general external policy context without an extra LLM or embedding model, and runs locally on an 8 GB Mac. No real payment processor is connected. **Do not use real customer data.**

## Files

| File | Purpose |
| --- | --- |
| `middlewareHITLMAS.py` | Original professor-provided baseline |
| `middlewareHITLMAS_RAG.py` | RAG workflow; preserves human approval in interactive mode |
| `refund_policy.json` | Ten general illustrative policies, not tailored to the original sample amounts |
| `evaluation_cases.json` | Sixteen new synthetic cases with expected relevant policy and provisional refund-proposal label |
| `evaluate_lab.py` | Noninteractive paired baseline/RAG evaluation (never approves refunds) |
| `test_policy_retrieval.py` | Offline unit tests for data and retrieval |
| `CODEX_EVALUATION_PROMPT.md` | Ready-to-paste local Codex prompt to execute and analyze the benchmark |

## Local setup

Use the same Python environment and Ollama model you used for the earlier lab:

```bash
git switch feature/rag-refund-policy
git pull origin feature/rag-refund-policy
source .venv/bin/activate
pip install langchain langgraph langchain-ollama python-dotenv typing-extensions
ollama pull qwen2.5:3b
python -m unittest -v test_policy_retrieval.py
python -m py_compile middlewareHITLMAS.py middlewareHITLMAS_RAG.py evaluate_lab.py
```

In a local `.env` (the file is gitignored):

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=IS590R-Lab6-Extension
```

The runner reports whether tracing is configured but cannot guarantee traces reached LangSmith: verify in the project dashboard. Never print or commit API keys.

## Baseline-versus-RAG test protocol

First run a fast smoke test:

```bash
python evaluate_lab.py --limit 3 --repeat 1 --output smoke_results.csv
```

Then run all 16 cases through **both** versions, twice each:

```bash
python evaluate_lab.py --repeat 2 --output evaluation_results.csv
```

This produces **64 variant runs** in total (16 cases × 2 versions × 2 repetitions), assuming all complete. Runs use the same Qwen2.5 3B model, identical messages, and uniquely tagged LangSmith traces. The CLI prints progress and flushes each row to CSV immediately, so interrupted experiments retain completed rows. It creates `evaluation_results.summary.json` with aggregate statistics.

**Safety and methodology:** The benchmark stops at HITL interrupt, **never calls `Command(resume)`**, and therefore never executes proposed refunds. It records tool proposals and final non-refund outcomes; it cannot assess real financial eligibility. It measures:
- Correctness of **refund proposed / not proposed** against predeclared synthetic labels, not correctness of explanations
- Policy retrieval recall@3 against a predeclared reference policy ID
- End-to-end per-case wall-clock latency
- Error counts and representative trace IDs

Review LangSmith manually for whether a refusal gave appropriate explanations, amounts were correct, and multiple policies were followed. A proposal of a refund is **not** authorization to pay; billing records still need verification. Cold-start runs may be slower, so keep individual timings and avoid overstating mean differences.

### Evaluation caveats

The new test cases are outside the original five classroom samples, but were written for this fictional policy framework. They are not a genuine independent industry benchmark. Retrieval may fail on paraphrases. Policy IDs and refund labels are teaching assumptions rather than verified ground truth. Some cases call for clarification/escalation rather than automatic refunds; a no-refund tool proposal is insufficient to establish a good outcome. The baseline/extension comparison tests **access to company policy context** and retrieval, not a broad claim that RAG outperforms all possible prompting. Traces can expose fictional message inputs and tool arguments, so only share public traces after reviewing their content.

## LangSmith evidence for the memo

All evaluation runs use tags: `lab6`, `extension-evaluation`, variant `baseline` or `rag`, and case ID like `H01`. Run names look like `lab6_rag_H01_r1`. Each CSV row also records a trace run UUID for lookup. Look under the `LANGSMITH_PROJECT` you configured.

1. Find at least two paired baseline + RAG comparisons and inspect their child nodes.
2. Verify the extension trace includes `retrieve_policy` while the baseline does not.
3. Record **actual** public LangSmith share URLs (never invent them).
4. Capture the LangSmith project runs page and relevant trace detail/screenshots.
5. Compare policy correctness and any observed latency overhead without claiming improvement in the absence of evidence.
6. Fill in the provided lab extension memo, including AI use disclosure and limitations.

**Codex:** Open `CODEX_EVALUATION_PROMPT.md` and paste the instruction into Codex running locally where Ollama is accessible. Cloud Codex may not reach your Mac. Neither the runner nor this repository creates public share links or screenshots automatically.

## Still-existing limitations

The extension's refund agent still receives the original complaint with possible PII, while retrieval uses only the sanitized summary. No real payment verification or persistent checkpoints exist. The human approval flow is present for interactive runs. The benchmark does not resume human approvals because it evaluates tool proposals only. These are important production tradeoffs, not problems silently claimed as solved.
