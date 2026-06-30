"""
react_agent/agent.py — ReAct pattern: Reason, Act, Observe, repeat.

Domain: Accounts Payable — investigating a disputed invoice.

The agent alternates between:
  REASON  — decide what to do next given current knowledge
  ACT     — call exactly one tool
  OBSERVE — incorporate the tool's result into knowledge
until it reaches a conclusion or hits max_steps.

This is intentionally NOT an LLM-backed implementation — it uses a
deterministic reasoning function so the pattern is runnable and testable
with zero API keys. Swap `_reason()` for an LLM call in production; the
ReAct loop structure itself does not change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Step:
    step_num: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[str] = None


@dataclass
class ReActResult:
    conclusion: str
    resolved: bool
    steps: list[Step] = field(default_factory=list)
    escalation_reason: Optional[str] = None


class InvoiceDisputeAgent:
    """
    ReAct agent investigating a disputed AP invoice.

    Tools available: lookup_vendor, lookup_po, lookup_receipt, check_duplicate.
    The agent decides which to call, in what order, based on what it learns
    at each step — this is the defining trait of ReAct versus a fixed pipeline.
    """

    MAX_STEPS = 6

    def __init__(self, tools: Optional[dict[str, Callable]] = None) -> None:
        self.tools = tools or self._default_tools()

    def run(self, invoice: dict) -> ReActResult:
        steps: list[Step] = []
        knowledge: dict = {"invoice": invoice}

        for i in range(1, self.MAX_STEPS + 1):
            thought, action, action_input = self._reason(knowledge, steps)
            step = Step(step_num=i, thought=thought, action=action, action_input=action_input)

            if action is None:
                # Agent has reached a conclusion — no further tool call needed.
                steps.append(step)
                return self._finalize(knowledge, steps)

            observation = self._act(action, action_input)
            step.observation = observation
            knowledge[self._knowledge_key(action)] = observation
            steps.append(step)

            logger.debug("Step %d: %s -> %s -> %s", i, thought, action, observation)

        # Exhausted max_steps without a conclusion — escalate rather than guess.
        return ReActResult(
            conclusion="Could not resolve within step budget — escalating to AP manager.",
            resolved=False,
            steps=steps,
            escalation_reason="MAX_STEPS_EXCEEDED",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # REASON — decide the next action given current knowledge
    # ─────────────────────────────────────────────────────────────────────────

    def _reason(self, knowledge: dict, steps: list[Step]) -> tuple[str, Optional[str], Optional[dict]]:
        """
        Deterministic reasoning policy standing in for an LLM call.
        Each branch represents a "thought" an LLM would produce given the
        same knowledge state — written explicitly here so the pattern is
        inspectable and testable without nondeterminism.
        """
        invoice = knowledge["invoice"]

        if "vendor_record" not in knowledge:
            return (
                "I don't know if this vendor is recognised. Let me look them up first.",
                "lookup_vendor",
                {"vendor_id": invoice["vendor_id"]},
            )

        vendor = knowledge["vendor_record"]
        if vendor.get("status") == "BLOCKED":
            return (
                f"Vendor {invoice['vendor_id']} is blocked. No further investigation needed.",
                None,
                None,
            )

        if "po_record" not in knowledge:
            return (
                "Vendor is valid. Now I need to check if there's a matching PO.",
                "lookup_po",
                {"po_number": invoice.get("po_number")},
            )

        po = knowledge["po_record"]
        if po is None:
            if "duplicate_check" not in knowledge:
                return (
                    "No PO found for this invoice. I should check for a duplicate before escalating.",
                    "check_duplicate",
                    {"invoice_id": invoice["invoice_id"], "vendor_id": invoice["vendor_id"]},
                )
            if knowledge["duplicate_check"].get("is_duplicate"):
                return (
                    "This appears to be a duplicate of an already-paid invoice. Concluding.",
                    None,
                    None,
                )
            # No PO and not a duplicate — nothing more to investigate, conclude and escalate.
            return ("No PO match and not a duplicate. Escalating for manual review.", None, None)

        if po is not None and "receipt_record" not in knowledge:
            return (
                "PO found. Checking goods receipt to complete the three-way match.",
                "lookup_receipt",
                {"po_number": po["po_number"]},
            )

        # All evidence gathered — reach a conclusion.
        return ("All evidence gathered. Forming conclusion.", None, None)

    @staticmethod
    def _knowledge_key(action: str) -> str:
        """Map a tool/action name to the knowledge dict key _reason() checks for."""
        mapping = {
            "lookup_vendor": "vendor_record",
            "lookup_po": "po_record",
            "lookup_receipt": "receipt_record",
            "check_duplicate": "duplicate_check",
        }
        return mapping.get(action, action)

    # ─────────────────────────────────────────────────────────────────────────
    # ACT — call exactly one tool
    # ─────────────────────────────────────────────────────────────────────────

    def _act(self, action: str, action_input: dict) -> str:
        tool = self.tools.get(action)
        if tool is None:
            return f"ERROR: unknown tool '{action}'"
        try:
            result = tool(**action_input)
            return result
        except Exception as exc:
            logger.warning("Tool %s failed: %s", action, exc)
            return f"ERROR: {exc}"

    # ─────────────────────────────────────────────────────────────────────────
    # FINALIZE — turn accumulated knowledge into a conclusion
    # ─────────────────────────────────────────────────────────────────────────

    def _finalize(self, knowledge: dict, steps: list[Step]) -> ReActResult:
        vendor = knowledge.get("vendor_record")
        if vendor and vendor.get("status") == "BLOCKED":
            return ReActResult(
                conclusion=f"Invoice rejected — vendor {knowledge['invoice']['vendor_id']} is blocked.",
                resolved=True,
                steps=steps,
                escalation_reason="VENDOR_BLOCKED",
            )

        dup = knowledge.get("duplicate_check")
        if dup and dup.get("is_duplicate"):
            return ReActResult(
                conclusion=f"Invoice rejected — duplicate of {dup.get('original_invoice_id')}.",
                resolved=True,
                steps=steps,
                escalation_reason="DUPLICATE_INVOICE",
            )

        po = knowledge.get("po_record")
        receipt = knowledge.get("receipt_record")

        if po is None:
            return ReActResult(
                conclusion="No PO found and not a duplicate — escalate to AP manager for non-PO approval.",
                resolved=False,
                steps=steps,
                escalation_reason="NO_PO_MATCH",
            )

        if receipt and receipt.get("quantity_received", 0) < po.get("quantity_ordered", 0):
            return ReActResult(
                conclusion=(
                    f"Partial receipt detected — received {receipt['quantity_received']} of "
                    f"{po['quantity_ordered']} ordered. Hold for partial payment review."
                ),
                resolved=False,
                steps=steps,
                escalation_reason="PARTIAL_RECEIPT",
            )

        return ReActResult(
            conclusion="Three-way match complete — vendor valid, PO matched, receipt confirmed. Approve for payment.",
            resolved=True,
            steps=steps,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Default mock tools — replace with real ERP/vendor master calls in production
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _default_tools() -> dict[str, Callable]:
        # In-memory mock data store standing in for real ERP/vendor master systems.
        vendors = {
            "V-1001": {"vendor_id": "V-1001", "name": "Acme Chemicals", "status": "ACTIVE"},
            "V-1002": {"vendor_id": "V-1002", "name": "Blocked Supplies Ltd", "status": "BLOCKED"},
            "V-1003": {"vendor_id": "V-1003", "name": "Global Reagents Inc", "status": "ACTIVE"},
        }
        pos = {
            "PO-5001": {"po_number": "PO-5001", "vendor_id": "V-1001", "quantity_ordered": 100},
            "PO-5003": {"po_number": "PO-5003", "vendor_id": "V-1003", "quantity_ordered": 50},
        }
        receipts = {
            "PO-5001": {"po_number": "PO-5001", "quantity_received": 100},
            "PO-5003": {"po_number": "PO-5003", "quantity_received": 30},  # partial
        }
        paid_invoices = {
            ("V-1001", "INV-9001"): True,
        }

        def lookup_vendor(vendor_id: str) -> dict:
            return vendors.get(vendor_id, {"vendor_id": vendor_id, "status": "UNKNOWN"})

        def lookup_po(po_number: Optional[str]) -> Optional[dict]:
            if not po_number:
                return None
            return pos.get(po_number)

        def lookup_receipt(po_number: str) -> Optional[dict]:
            return receipts.get(po_number)

        def check_duplicate(invoice_id: str, vendor_id: str) -> dict:
            is_dup = (vendor_id, invoice_id) in paid_invoices
            return {
                "is_duplicate": is_dup,
                "original_invoice_id": invoice_id if is_dup else None,
            }

        return {
            "lookup_vendor": lookup_vendor,
            "lookup_po": lookup_po,
            "lookup_receipt": lookup_receipt,
            "check_duplicate": check_duplicate,
        }
