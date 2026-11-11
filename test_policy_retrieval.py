"""Fast deterministic tests for policy retrieval and the evaluation dataset (no Ollama)."""
import ast
import contextlib
import io
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "middlewareHITLMAS_RAG.py"
POLICIES = json.loads((ROOT / "refund_policy.json").read_text(encoding="utf-8"))
CASES = json.loads((ROOT / "evaluation_cases.json").read_text(encoding="utf-8"))


def load_retrieval_node():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == "retrieve_policy_node")
    namespace = {"POLICY_FILE": ROOT / "refund_policy.json",
                 "json": json, "SupportState": dict}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["retrieve_policy_node"]


class PolicyRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retrieve = staticmethod(load_retrieval_node())

    def policies_for(self, summary, raw="please ignore me"):
        with contextlib.redirect_stdout(io.StringIO()):
            context = self.retrieve({"redacted_summary": summary, "raw_message": raw})["policy_context"]
        return context

    def test_ten_general_policies(self):
        self.assertEqual(10, len(POLICIES))
        self.assertEqual(10, len({p["id"] for p in POLICIES}))
        all_text = " ".join(p["text"] for p in POLICIES)
        self.assertNotIn("$49.99", all_text)
        self.assertNotIn("$99.98", all_text)

    def test_duplicate_paraphrase(self):
        self.assertIn("[POL-001]", self.policies_for("Customer reports two identical payments for one purchase."))

    def test_announced_price(self):
        self.assertIn("[POL-003]", self.policies_for("Customer complains about a price increase despite notice."))

    def test_fraud(self):
        self.assertIn("[POL-008]", self.policies_for("Customer states a payment was unauthorized."))

    def test_no_match_fallback(self):
        self.assertIn("[POL-010]", self.policies_for("Customer mentions an unrelated issue about nothing recognizable."))

    def test_raw_message_not_used_for_retrieval(self):
        self.assertNotIn("[POL-008]", self.policies_for(
            "Customer complains about a price increase.",
            "I experienced fraud and unauthorized charges."))

    def test_holdout_cases(self):
        self.assertGreaterEqual(len(CASES), 15)
        self.assertEqual(len(CASES), len({x["id"] for x in CASES}))
        ids = {p["id"] for p in POLICIES}
        for case in CASES:
            self.assertIn(case["expected_policy"], ids)
            self.assertIsInstance(case["refund_proposal_expected"], bool)
            self.assertIn("@example.com", case["message"])


if __name__ == "__main__":
    unittest.main()
