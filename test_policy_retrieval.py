"""Test retrieval without starting Ollama or calling any language model."""
import ast
import contextlib
import io
import json
import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name("middlewareHITLMAS_RAG.py")


def load_retrieval_node():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "retrieve_policy_node")
    namespace = {"POLICY_FILE": SOURCE.with_name("refund_policy.json"), "json": json, "SupportState": dict}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["retrieve_policy_node"]


class PolicyRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retrieve = staticmethod(load_retrieval_node())

    def policies_for(self, summary):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.retrieve({"redacted_summary": summary, "raw_message": "irrelevant raw PII"})["policy_context"]

    def test_duplicate_charge(self):
        self.assertIn("[POL-001]", self.policies_for("Customer was charged twice on the same day."))

    def test_announced_price_increase(self):
        self.assertIn("[POL-002]", self.policies_for("Customer disputes a subscription price increase."))

    def test_prior_refund(self):
        self.assertIn("[POL-004]", self.policies_for("Customer was already refunded."))

    def test_unclear_fallback(self):
        self.assertIn("[POL-003]", self.policies_for("Customer says something went wrong."))

    def test_raw_message_does_not_affect_retrieval(self):
        # The retrieval step should not scan raw contact/payment information.
        self.assertNotIn("[POL-004]", self.retrieve_without_stdout({
            "redacted_summary": "Customer disputes a price increase on their plan.",
            "raw_message": "I was already refunded and want another refund.",
        }))

    def retrieve_without_stdout(self, state):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.retrieve(state)["policy_context"]


if __name__ == "__main__":
    unittest.main()
