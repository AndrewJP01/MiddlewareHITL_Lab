# Codex prompt — IS 590R Lab 6 policy-RAG evaluation

Copy the text below into **Codex running locally in VS Code or Codex CLI** with this repository opened. Cloud environments generally cannot reach my Mac's Ollama server.

---

We are evaluating a LangGraph HITL refund agent for a university lab extension memo. The branch `feature/rag-refund-policy` contains two variants:

- `middlewareHITLMAS.py`: original, unmodified baseline.
- `middlewareHITLMAS_RAG.py`: policy retrieval extension using ten general illustrative policies in `refund_policy.json`.

`evaluation_cases.json` contains sixteen **synthetic, newly written cases**, independent of the original five examples. We want a fair comparison without ever executing refund tools. `evaluate_lab.py` invokes each graph only until it either finishes or is interrupted by the HITL approval middleware; it never approves or resumes refund calls.

Please do the following:

1. Inspect the Python scripts, knowledge base, dataset, and `README_RAG.md` to understand the experiment. Do not silently change the evaluation labels or tailor policy wording to outcomes. Treat the labels as **hypotheses/heuristics**, not verified financial truth.
2. Verify local environment: the project's `.venv`, Ollama availability, `qwen2.5:3b`, and the presence (NOT values) of `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` in the locally loaded environment. **Never display, copy, commit, or upload secrets or real customer data.** Do not replace the baseline model or use an alternate provider.
3. Run `python -m unittest -v test_policy_retrieval.py` and `python -m py_compile middlewareHITLMAS.py middlewareHITLMAS_RAG.py evaluate_lab.py`. If something fails, debug without changing the original baseline behavior or answer labels.
4. Run an initial smoke test: `python evaluate_lab.py --limit 3 --repeat 1 --output smoke_results.csv`. Inspect errors and verify both baseline and RAG LangSmith traces show up in my intended project. If Ollama is unavailable in this environment, **do not fake results**; tell me to run these commands on my Mac instead.
5. If smoke tests work, run `python evaluate_lab.py --repeat 2 --output evaluation_results.csv` (32 comparisons / 64 total variant runs for 16 cases, two repetitions). If this is too slow on an 8 GB Mac, ask whether to reduce to `--repeat 1`. The runner records raw results as it goes. **Do not call or approve the refund tool.**
6. Analyze the resulting CSV and summary JSON. Report counts of completed/error runs, number of correct refund-proposal classifications under the predeclared labels, policy retrieval recall@3, average and median wall-clock latency, and notable disagreements. Note that nonrefund outcomes do **not** prove good customer responses: inspect representative LangSmith traces and manually evaluate policy adherence and refund amounts. Do not assert any performance improvement unless observed.
7. Identify run names/IDs and help me create **public LangSmith trace share links** for 2–4 especially informative cases (paired baseline and RAG). Only share traces containing fictional test data. If trace-sharing needs UI access or my sign-in, explain exactly what I need to click and leave URLs blank instead of inventing them.
8. Prepare `extension_evidence_summary.md` locally (gitignored) with a concise results table, the exact commands used, any errors, representative case-level evidence, the methodological limitations, placeholders for **real** public LangSmith URLs and project screenshots, and a draft benefits/tradeoffs argument for the provided Lab Extension Memo. Do **not** claim screenshots or public links exist if they haven't been created.
9. Keep experiment data and `.env` out of git. If you make any code fixes, use a separate commit on the feature branch and explain changes before suggesting a merge. Do not alter `main`.

Be skeptical of validity: The two variants must use the **same model and exact same cases**. Prior initial experiments used different models; those observations are not fair RAG-versus-baseline comparisons. The policy corpus was designed without the prior sample-specific $49.99 wording, but the new held-out cases are still synthetic and their outcomes should not be treated as proof of production accuracy.

---

**Manual commands if Codex can't access Ollama:**

```bash
git switch feature/rag-refund-policy
git pull origin feature/rag-refund-policy
source .venv/bin/activate
python -m unittest -v test_policy_retrieval.py
python evaluate_lab.py --limit 3 --repeat 1 --output smoke_results.csv
python evaluate_lab.py --repeat 2 --output evaluation_results.csv
```

Each benchmark case stops on HITL and logs the proposed action; no interaction is required.
