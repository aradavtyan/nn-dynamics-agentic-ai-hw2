from __future__ import annotations

import argparse
import hashlib
from contextlib import contextmanager
import json
import operator
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


from task6 import agentic_system as hw1
from task6.agentic_system import (
    MAX_DECISIONS,
    STDERR_MARK,
    STDOUT_MARK,
    TARGET_TEST_RMSE,
    AgenticSilverboxSystem,
    ModelCaller,
    PythonCodingTool,
    Turn,
    count_role,
    format_dialogue,
    latest_orchestrator_decision,
    parse_json_object,
    save_dialogue,
)

ALLOWED_NEXT = hw1.ALLOWED_NEXT | {"human"}
HUMAN_REQUESTS = {"clarification", "approval", "final_approval"}
DEFAULT_THREAD_ID = "hw2-task1"

ACTION_TYPE = "python_write_and_run"
FINAL_ACTION_TYPE = "report_final_measurement"
FINAL_REPORT_FILE = "final_measurement.json"
PREPARED_MARK = "ACTION_PREPARED"
REJECTED_MARK = "ACTION_REJECTED"


NOT_EXECUTED_MARKS = (PREPARED_MARK, REJECTED_MARK)


class HitlState(TypedDict, total=False):

    dialogue: Annotated[list[Turn], operator.add]


    pending_action: dict | None

    pending_final: dict | None

    final_report: dict | None


class OverwriteRefused(RuntimeError):
    pass


def ensure_fresh_workdir(workdir: Path) -> Path:


    workdir = Path(workdir)
    if workdir.exists() and (not workdir.is_dir() or any(workdir.iterdir())):
        raise OverwriteRefused(
            f"refusing to use {workdir}: it already exists and is not an empty directory, "
            f"and this run would overwrite its dialogue.json/dialogue.txt. Overwriting "
            f"existing results is not supported; choose a new --workdir."
        )
    return workdir


def write_new_file(path: Path, text: str) -> Path:

    try:
        with open(path, "x", encoding="utf-8") as fh:
            fh.write(text)
    except FileExistsError:
        raise OverwriteRefused(
            f"refusing to overwrite existing result {path}; overwriting is not supported"
        ) from None
    return Path(path)


def write_final_report(workdir: Path, final_report: dict) -> Path:
    return write_new_file(Path(workdir) / FINAL_REPORT_FILE, json.dumps(final_report, indent=2))


REPO_ROOT = Path(__file__).resolve().parents[1]

HISTORY_SOURCE = "results/silverbox_reference.json"


def load_historical_results(root: Path = REPO_ROOT) -> dict:
    history = {"records": [], "sources_read": [], "problems": []}
    path = Path(root) / HISTORY_SOURCE
    if not path.is_file():
        history["problems"].append(f"{HISTORY_SOURCE}: missing")
        return history
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        for key in ("task", "run_id", "provenance", "configuration", "units", "verification", "source_file"):
            if not isinstance(record[key], str):
                raise ValueError(f"{key} must be text")
        for key in ("train_rmse", "test_rmse"):
            value = record[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
                raise ValueError(f"{key} must be a finite nonnegative number")
        record["source_file"] = HISTORY_SOURCE
    except (OSError, ValueError, KeyError, TypeError) as exc:
        history["problems"].append(f"{HISTORY_SOURCE}: unusable ({type(exc).__name__}: {exc})")
        return history
    history["sources_read"].append(HISTORY_SOURCE)
    history["records"].append(record)
    return history


def format_history(history: dict) -> str:

    lines = [
        "RECORDED RESULTS FROM EARLIER WORK",
        "(Read-only historical evidence, not produced during the current run. These are not "
        "executor attempts of this run, do not count as results of this run, and cannot be "
        "used to finish this run.)",
    ]
    for r in history["records"]:
        train = f"{r['train_rmse']} {r['units']}" if r["train_rmse"] is not None else "not recorded"
        lines.append(
            f"- [{r['task']} / {r['run_id']}] {r['provenance']}; {r['configuration']}; "
            f"TRAIN_RMSE = {train}; TEST_RMSE = {r['test_rmse']} {r['units']}; "
            f"verification: {r['verification']}; source: {r['source_file']}"
        )
    if not history["records"]:
        lines.append("- (no recorded results could be read)")
    for problem in history["problems"]:
        lines.append(f"- unavailable source: {problem}")
    return "\n".join(lines)


def executed_dialogue(dialogue: list[Turn]) -> list[Turn]:

    return [
        t for t in dialogue
        if not (t["role"] == "executor" and t["text"].startswith(NOT_EXECUTED_MARKS))
    ]


def code_sha256(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def public_action(action: dict) -> dict:

    keys = ("action_id", "action_type", "target_role", "operation", "filename",
            "code_sha256", "code", "status")
    return {k: action[k] for k in keys}


def describe_action(action: dict | None) -> str:
    if not action:
        return "(none)"
    text = (
        f"action_id={action['action_id']} status={action['status']} "
        f"operation={action['operation']}"
    )
    if action.get("refusal_reason"):
        text += f" refusal_reason={action['refusal_reason']}"
    return text


def executed_results(dialogue: list[Turn]) -> list[dict]:

    view = executed_dialogue(dialogue)
    turns = [t for t in view if t["role"] == "executor"]
    attempts = hw1.executor_attempts(view)
    for attempt, turn in zip(attempts, turns):
        action_id = re.search(r"^action_id=(\S+)", turn["text"], re.MULTILINE)
        digest = re.search(r"^code_sha256=([0-9a-f]{64})$", turn["text"], re.MULTILINE)
        attempt["execution_action_id"] = action_id.group(1) if action_id else None
        attempt["code_sha256"] = digest.group(1) if digest else None
    return attempts


def build_final_action(dialogue: list[Turn], request_no: int) -> dict | None:


    best = hw1.best_executor_result(executed_dialogue(dialogue))
    if best is None:
        return None
    record = next(a for a in executed_results(dialogue) if a["attempt"] == best["attempt"])
    measurement = {
        "quantity": "TEST_RMSE",
        "description": "complete free-run RMSE on the 1024-sample Silverbox test segment, "
                       "printed by the executed Task 6 agent script",
        "value": record["test_rmse"],
        "units": "V",
        "train_rmse": record["train_rmse"],
        "run_file": record["file"],
        "attempt": record["attempt"],
        "exit_code": record["exit_code"],
        "execution_action_id": record["execution_action_id"],
        "code_sha256": record["code_sha256"],
    }
    digest = hashlib.sha256(json.dumps(measurement, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "action_id": f"final-{request_no:02d}-{digest[:12]}",
        "action_type": FINAL_ACTION_TYPE,
        "target_role": "finish",
        "operation": (
            f"end the dialogue and report TEST_RMSE = {measurement['value']} V "
            f"from {measurement['run_file']} as the final measurement"
        ),
        "measurement": measurement,
        "measurement_sha256": digest,
        "status": "awaiting_approval",
    }


def describe_final(action: dict) -> str:
    m = action["measurement"]
    return (
        f"Approval requested for action {action['action_id']}: {action['operation']}.\n"
        f"Measurement: {m['quantity']} = {m['value']} {m['units']} ({m['description']}); "
        f"TRAIN_RMSE = {m['train_rmse']} {m['units']}.\n"
        f"Produced by: run {m['run_file']} (attempt {m['attempt']:02d}, exit_code {m['exit_code']}, "
        f"execution action {m['execution_action_id']}, code_sha256 {m['code_sha256']}).\n"
        f"Intended final action: end the dialogue and report exactly this measurement."
    )


def approval(action_id: str, approved: bool, text: str) -> dict:

    return {"approved": approved, "action_id": action_id, "text": text}


def interpret_approval(reply: Any, action_id: str) -> tuple[bool, str, str]:

    if not isinstance(reply, dict):
        return False, str(reply), "reply was not a structured approval decision"
    text = str(reply.get("text") or "")
    if reply.get("approved") is not True:
        return False, text, "human refused"
    if reply.get("action_id") != action_id:
        return False, text, (
            f"approval names action {reply.get('action_id')!r}, not the pending {action_id!r}"
        )
    return True, text, ""


EXECUTOR_TASK_ALIGNMENT = """You are the EXECUTOR in the HW2 human-in-the-loop system.

TASK ALIGNMENT. This section takes precedence over the HW1 instructions below.
- Implement exactly what the USER TASK and the ORCHESTRATOR INSTRUCTION ask for, as one
  NEW Python script. A human reviews this exact code before anything runs.
- The HW1 instructions below describe the Silverbox neural state-space MODELING task
  from Homework 1. Apply its modeling requirements (PyTorch acceleration network, training, RK4
  free run, FREE-RUN CONTRACT, saved model artifacts) ONLY when the user asks for that
  modeling, fitting or training work, or a scientist plan for it is being implemented.
- For any other computation, do NOT define or train a model and do not add the modeling
  artifacts; write only the code the request needs.
- In every case keep the HW1 data conventions: load SNLS80mV.mat from the path in the
  environment variable SILVERBOX_MAT_PATH, channel V1 is the input u and V2 the output y,
  use the excerpt [42650:46746], subtract the excerpt mean of u and of y, the first 3072
  samples are the training record and the last 1024 the test record. No network access,
  CPU only, finish well within the time limit, never modify the data file.
- Compute every reported number in the script from the data; never hard-code a result.
  Print each result on its own line as NAME=<number>, for example TEST_RMSE=<number>
  (print TRAIN_RMSE=<number> only if it was actually computed).

HW1 EXECUTOR INSTRUCTIONS (reference; see TASK ALIGNMENT for when they apply):
"""


class _TaskAlignedExecutorLLM:


    def __init__(self, llm):
        self.llm = llm

    def call(self, instructions: str, input_text: str) -> str:
        return self.llm.call(EXECUTOR_TASK_ALIGNMENT + instructions, input_text)


SECRET_ENV_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|^OPENAI_", re.IGNORECASE)


@contextmanager
def secrets_removed_from_environment():


    names = [name for name in os.environ if SECRET_ENV_NAME.search(name)]
    removed = {name: os.environ.pop(name) for name in names}
    try:
        yield sorted(removed)
    finally:
        os.environ.update(removed)


class _PreparedAction(Exception):
    def __init__(self, filename: str, code: str):
        super().__init__(filename)
        self.filename = filename
        self.code = code


class _CaptureTool:


    def write_and_run(self, filename: str, code: str) -> dict:
        raise _PreparedAction(filename, code)


class HumanInTheLoopSystem(AgenticSilverboxSystem):


    def __init__(self, llm, workdir: Path, data_path: Path, coding_tool=None, history: dict | None = None):


        self.llm = llm
        self.coding_tool = coding_tool or PythonCodingTool(workdir=workdir, data_path=data_path)
        self.runs: list[dict] = []
        self.history_context = format_history(history) if history is not None else None


    def orchestrator(self, state: HitlState) -> dict:
        dialogue = state["dialogue"]
        attempts_view = executed_dialogue(dialogue)
        decision_no = count_role(dialogue, "orchestrator") + 1
        best = hw1.best_executor_result(attempts_view)
        best_rmse = None if best is None else best["test_rmse"]
        remaining_after = MAX_DECISIONS - decision_no
        finish_allowed = best_rmse is not None and (
            best_rmse < TARGET_TEST_RMSE or remaining_after < 2
        )

        instructions = """You are the ORCHESTRATOR in a LangGraph system with four agent roles
and a human operator.

You see the complete dialogue and must choose exactly one next destination.

Return ONLY a JSON object with exactly these five fields:
{
  "next_role": "scientist" | "general" | "human",
  "request": "clarification" | "approval" | "final_approval" (only when next_role is human, else ""),
  "instruction": "specific instruction for the role (for an approval: the executor's instruction)",
  "question": "the exact question for the human (empty string unless next_role is human)",
  "reason": "one-sentence reason"
}

FIRST, UNDERSTAND THE REQUEST. From the user's words and the dialogue, decide what
the user is asking for:
  (a) new scientific or computational work, such as fitting, training or improving a
      Silverbox model in this run;
  (b) information about existing results, such as results recorded from earlier work;
  (c) a request that is ambiguous;
  (d) another supported operation, such as explaining a term, the data or a metric.
- The Silverbox fitting workflow, the quality target and the QUALITY GATE below apply
  ONLY to (a). Do not start new fitting work the user did not ask for.
- AMBIGUITY: if the request has more than one reasonable interpretation and the
  interpretations would lead to a different reported value or a different next
  destination (for example, several recorded results could match, or it is unclear
  whether new work or an existing result is wanted), choose next_role "human" with
  request "clarification" and ask ONE specific question that names the alternatives.
  Do not ask when the user's words, the dialogue or the recorded results already
  determine the answer, and do not ask merely to confirm.
- RECORDED RESULTS FROM EARLIER WORK, if shown, are historical evidence read from disk.
  They are not executor attempts of this run, they never allow finishing this run, and
  they must not be presented as new measurements. That this run has no executor
  attempts yet does not mean no results exist. Quote a recorded value only together
  with its source and run name, exactly as recorded.
- A clear question about recorded results or terminology can go to general, with an
  instruction that quotes the relevant recorded values and provenance exactly.

Rules:
- Your decision MUST be made from the dialogue; do not use a fixed role sequence.
- You do theory and no coding yourself.
- scientist owns the differential equation, network/state/training plan, free-run test,
  and a numbered implementation plan.
- general answers side questions about the record, terminology, or meaning of metrics.
- executor implements the latest scientific plan, writes a NEW Python script and runs it
  with the coding tool, and can repair a prior failed run.
- HUMAN APPROVAL: the executor's coding tool may run ONLY with explicit human approval.
  To have code written and run, choose next_role "human" with request "approval", put
  the executor's instruction in "instruction", and ask the human in "question" to
  approve running it. The executor then prepares the exact script, the human approves
  or refuses that exact action, and only an approved action runs, exactly once.
  Never choose next_role "executor" directly: it is rejected. Each approval covers one
  action only; every further run needs a new approval. If the human refused, take the
  refusal into account before deciding what to do next.
- human with request "clarification": use it for the AMBIGUITY case above or when
  information is missing that no role can determine from the dialogue or the recorded
  results. A HUMAN turn in the dialogue is the operator's answer.
- A traceback or failed program belongs with the executor (via human approval).
- A question about the scientific plan belongs with scientist.
- FINAL APPROVAL: the dialogue ends with a reported measurement only after the human
  approves it. To finish, choose next_role "human" with request "final_approval" and
  ask the human in "question" to approve ending the dialogue and reporting the result.
  The program attaches the exact measurement (the kept real executor result of THIS
  run) and its provenance; do not state numbers that no executor turn printed. Never choose
  next_role "finish" directly: it is rejected. If the human refuses, continue the
  work or ask what they want instead.
- You may request final_approval ONLY if an executor turn records exit_code=0 AND a
  numeric TEST_RMSE printed by a script that actually ran.
- QUALITY GATE (new fitting work in this run only): a zero predictor scores about
  0.06 V test RMSE. If the best real
  TEST_RMSE is not below the quality target and enough decisions remain, do NOT
  request final_approval: route to the scientist to diagnose the weak free-run result
  and propose a materially better plan, or request approval for the executor to
  implement an already proposed improvement in a NEW script. Choose whichever the
  dialogue shows is needed. Only when the decision budget is nearly exhausted may you
  request final_approval for the best real result obtained so far.
- The run keeps the successful attempt with the lowest complete free-run TEST_RMSE.
- Never invent a measured RMSE.
"""

        history_block = f"{self.history_context}\n\n" if self.history_context else ""
        input_text = (
            f"This is orchestrator decision {decision_no} of at most {MAX_DECISIONS}.\n"
            f"Quality target (applies only to new Silverbox model fitting or optimization "
            f"in this run): TEST_RMSE < {TARGET_TEST_RMSE} V.\n"
            f"Best verified executor TEST_RMSE in this run so far: {best_rmse!r}\n"
            f"finish currently allowed by the program: {finish_allowed}\n"
            f"LAST CODING ACTION: {describe_action(state.get('pending_action'))}\n"
            f"LAST FINAL REPORT ACTION: {describe_action(state.get('pending_final'))}\n"
            f"EXECUTOR ATTEMPTS IN THIS RUN:\n{hw1.attempts_table(attempts_view)}\n\n"
            f"{history_block}"
            f"DIALOGUE:\n{format_dialogue(dialogue)}"
        )


        feedback = ""
        decision = None
        for _ in range(3):
            raw = self.llm.call(instructions, input_text + feedback)
            try:
                candidate = parse_json_object(raw)
                role = str(candidate.get("next_role", "")).strip().lower()
                request = str(candidate.get("request") or "").strip().lower() or "clarification"
                if role not in ALLOWED_NEXT:
                    raise ValueError(f"next_role {role!r} is not one of {sorted(ALLOWED_NEXT)}")
                if role == "executor":
                    raise ValueError(
                        "next_role executor is forbidden without human approval: the coding "
                        "tool runs only after the human approves the exact action; choose "
                        "next_role human with request approval and the executor's instruction"
                    )
                if role == "human" and request not in HUMAN_REQUESTS:
                    raise ValueError(f"request {request!r} is not one of {sorted(HUMAN_REQUESTS)}")
                if role == "human" and not str(candidate.get("question") or "").strip():
                    raise ValueError("next_role is human but question is empty")
                if role == "human" and request == "approval" and not str(
                    candidate.get("instruction") or ""
                ).strip():
                    raise ValueError("an approval request needs the executor's instruction")
                finishing = role == "finish" or (role == "human" and request == "final_approval")
                if finishing and best_rmse is None:
                    raise ValueError(
                        "finish is forbidden: no executor turn has exit_code=0 "
                        "with a printed TEST_RMSE yet"
                    )
                if finishing and not finish_allowed:
                    raise ValueError(
                        f"finish is forbidden by the quality gate: best TEST_RMSE "
                        f"{best_rmse} V is not below {TARGET_TEST_RMSE} V and "
                        f"{remaining_after} decisions remain; route to scientist or "
                        f"request approval for the executor to improve the model"
                    )
                if role == "finish":
                    raise ValueError(
                        "finish is forbidden without human approval of the final measurement: "
                        "choose next_role human with request final_approval"
                    )
                decision = candidate
                break
            except Exception as exc:
                feedback = (
                    f"\n\nYOUR PREVIOUS REPLY WAS REJECTED ({exc}). Reply again with ONLY the "
                    f"JSON object with next_role, request, instruction, question, reason.\n"
                    f"Previous reply:\n{raw[:2000]}"
                )
        if decision is None:
            raise RuntimeError("Orchestrator LLM did not return a valid decision in 3 attempts.")

        next_role = str(decision["next_role"]).strip().lower()
        is_human = next_role == "human"
        normalized = {
            "next_role": next_role,
            "request": (str(decision.get("request") or "").strip().lower() or "clarification")
            if is_human else "",
            "instruction": str(decision.get("instruction") or "").strip(),
            "question": str(decision.get("question") or "").strip() if is_human else "",
            "reason": str(decision.get("reason") or "").strip(),
        }
        update: dict = {
            "dialogue": [
                {"role": "orchestrator", "text": json.dumps(normalized, ensure_ascii=False)}
            ]
        }
        if normalized["request"] == "final_approval":

            request_no = sum(
                t["role"] == "orchestrator" and json.loads(t["text"]).get("request") == "final_approval"
                for t in dialogue
            ) + 1
            update["pending_final"] = build_final_action(dialogue, request_no)
        return update


    def human(self, state: HitlState) -> dict:


        decision = latest_orchestrator_decision(state["dialogue"])
        final = state.get("pending_final")
        if decision.get("request") == "final_approval":
            if final is None or final["status"] != "awaiting_approval":

                return {"dialogue": [{"role": "human", "text": "(no final measurement awaiting approval)"}]}
            reply = interrupt({
                "kind": "final_approval",
                "question": f"{decision['question']}\n\n{describe_final(final)}",
                "reason": decision.get("reason", ""),
                "action": final,
            })
            approved, text, refusal_reason = interpret_approval(reply, final["action_id"])
            digest = hashlib.sha256(
                json.dumps(final["measurement"], sort_keys=True).encode("utf-8")
            ).hexdigest()
            if approved and digest != final["measurement_sha256"]:
                approved, refusal_reason = False, "measurement no longer matches its approved SHA-256"
            if not approved:
                return {
                    "dialogue": [{"role": "human", "text": text}],
                    "pending_final": {**final, "status": "refused", "refusal_reason": refusal_reason},
                }
            report = {
                **final["measurement"],
                "approved_action_id": final["action_id"],
                "measurement_sha256": final["measurement_sha256"],
                "approval_text": text,
            }
            return {
                "dialogue": [{"role": "human", "text": text}],
                "pending_final": {**final, "status": "finalized"},
                "final_report": report,
            }

        action = state.get("pending_action")
        if (
            decision.get("request") == "approval"
            and action is not None
            and action["status"] == "awaiting_approval"
        ):
            reply = interrupt({
                "kind": "approval",
                "question": decision["question"],
                "reason": decision.get("reason", ""),
                "action": public_action(action),
            })
            approved, text, refusal_reason = interpret_approval(reply, action["action_id"])
            updated = {**action, "status": "approved" if approved else "refused"}
            if not approved:
                updated["refusal_reason"] = refusal_reason
            return {"dialogue": [{"role": "human", "text": text}], "pending_action": updated}

        reply = interrupt({
            "kind": "clarification",
            "question": decision["question"],
            "reason": decision.get("reason", ""),
        })
        return {"dialogue": [{"role": "human", "text": str(reply)}]}


    def executor(self, state: HitlState) -> dict:
        dialogue = state["dialogue"]
        action = state.get("pending_action")
        last = dialogue[-1]

        if action is not None and action["status"] == "approved" and last["role"] == "human":
            return self._execute_approved(state, action)

        if last["role"] == "orchestrator":
            decision = latest_orchestrator_decision(dialogue)
            if decision.get("next_role") == "human" and decision.get("request") == "approval":
                return self._prepare(state, decision)

        return self._reject(
            action,
            "the coding tool runs only for an action the human approved in the "
            "immediately preceding turn; no such approval exists",
        )

    def _prepare(self, state: HitlState, decision: dict) -> dict:


        real_tool, real_llm = self.coding_tool, self.llm
        self.coding_tool = _CaptureTool()
        self.llm = _TaskAlignedExecutorLLM(real_llm)
        try:
            AgenticSilverboxSystem.executor(self, {"dialogue": executed_dialogue(state["dialogue"])})
        except _PreparedAction as prepared:
            filename, code = prepared.filename, prepared.code
        else:
            raise RuntimeError("HW1 executor returned without requesting the coding tool")
        finally:
            self.coding_tool, self.llm = real_tool, real_llm

        prepared_no = sum(
            t["role"] == "executor" and t["text"].startswith(PREPARED_MARK)
            for t in state["dialogue"]
        ) + 1
        digest = code_sha256(code)
        action = {
            "action_id": f"action-{prepared_no:02d}-{digest[:12]}",
            "action_type": ACTION_TYPE,
            "target_role": "executor",
            "operation": f"write NEW file {filename} and run it (hard timeout {hw1.TIMEOUT_S} s)",
            "filename": filename,
            "code": code,
            "code_sha256": digest,
            "instruction": decision.get("instruction", ""),
            "status": "awaiting_approval",
        }
        report = (
            f"{PREPARED_MARK} (NOT EXECUTED: awaiting human approval)\n"
            f"action_id={action['action_id']}\n"
            f"action_type={ACTION_TYPE}\n"
            f"target_role=executor\n"
            f"operation={action['operation']}\n"
            f"code_sha256={digest}\n"
            f"code_lines={len(code.splitlines())}"
        )
        return {"dialogue": [{"role": "executor", "text": report}], "pending_action": action}

    def _execute_approved(self, state: HitlState, action: dict) -> dict:
        if code_sha256(action["code"]) != action["code_sha256"]:
            return self._reject(action, "the stored code no longer matches the approved SHA-256")


        with secrets_removed_from_environment() as withheld:
            result = self.coding_tool.write_and_run(filename=action["filename"], code=action["code"])
        self.runs.append({k: v for k, v in result.items() if k not in ("stdout", "stderr")})

        stdout = result["stdout"][-16000:]
        stderr = result["stderr"][-12000:]
        tool_report = (
            f"action_id={action['action_id']} (approved by human; executed once)\n"
            f"code_sha256={action['code_sha256']}\n"
            f"file={result['relpath']}\n"
            f"timed_out={result['timed_out']}\n"
            f"elapsed_s={result['elapsed_s']}\n"
            f"exit_code={result['exit_code']}\n"
            f"{STDOUT_MARK}\n{stdout}\n"
            f"{STDERR_MARK}\n{stderr}\n"
            f"--- traceback ---\n{result['traceback'] or '(none)'}"
        )
        executed = {
            **action,
            "status": "executed",
            "outcome": {
                "file": result["relpath"],
                "exit_code": result["exit_code"],
                "timed_out": result["timed_out"],
                "train_rmse": result.get("train_rmse"),
                "test_rmse": result.get("test_rmse"),
                "env_vars_withheld_from_child": withheld,
            },
        }
        return {"dialogue": [{"role": "executor", "text": tool_report}], "pending_action": executed}

    @staticmethod
    def _reject(action: dict | None, why: str) -> dict:
        report = (
            f"{REJECTED_MARK} (NOT EXECUTED)\n"
            f"reason={why}\n"
            f"pending_action={describe_action(action)}"
        )
        update: dict = {"dialogue": [{"role": "executor", "text": report}]}
        if action is not None and action["status"] in ("awaiting_approval", "approved"):
            update["pending_action"] = {**action, "status": "rejected", "refusal_reason": why}
        return update


    @staticmethod
    def route_after_orchestrator(state: HitlState) -> str:
        decision = latest_orchestrator_decision(state["dialogue"])
        if decision["next_role"] == "human" and decision.get("request") == "approval":
            return "prepare"
        return str(decision["next_role"])

    @staticmethod
    def route_after_executor(state: HitlState) -> str:
        action = state.get("pending_action")
        if action is not None and action["status"] == "awaiting_approval":
            return "human"
        return AgenticSilverboxSystem.route_after_specialist(state)

    @staticmethod
    def route_after_human(state: HitlState) -> str:
        decision = latest_orchestrator_decision(state["dialogue"])
        final = state.get("pending_final")
        if decision.get("request") == "final_approval":
            if final is not None and final["status"] == "finalized":
                return END
            return AgenticSilverboxSystem.route_after_specialist(state)
        action = state.get("pending_action")
        if action is not None and action["status"] == "approved":
            return "executor"
        return AgenticSilverboxSystem.route_after_specialist(state)

    def build(self, checkpointer=None):
        builder = StateGraph(HitlState)

        builder.add_node("orchestrator", self.orchestrator)
        builder.add_node("scientist", self.scientist)
        builder.add_node("general", self.general)
        builder.add_node("executor", self.executor)
        builder.add_node("human", self.human)

        builder.add_edge(START, "orchestrator")
        builder.add_conditional_edges(
            "orchestrator",
            self.route_after_orchestrator,
            {
                "scientist": "scientist",
                "general": "general",
                "executor": "executor",
                "prepare": "executor",
                "human": "human",
                "finish": END,
            },
        )
        for node in ("scientist", "general"):
            builder.add_conditional_edges(
                node,
                self.route_after_specialist,
                {"orchestrator": "orchestrator", END: END},
            )
        builder.add_conditional_edges(
            "executor",
            self.route_after_executor,
            {"human": "human", "orchestrator": "orchestrator", END: END},
        )
        builder.add_conditional_edges(
            "human",
            self.route_after_human,
            {"executor": "executor", "orchestrator": "orchestrator", END: END},
        )

        return builder.compile(checkpointer=checkpointer or InMemorySaver())


def thread_config(thread_id: str = DEFAULT_THREAD_ID) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}


def pending_interrupt(graph, config: dict) -> dict | None:

    snapshot = graph.get_state(config)
    if not snapshot.interrupts:
        return None
    return snapshot.interrupts[0].value


def pending_question(graph, config: dict) -> str | None:
    payload = pending_interrupt(graph, config)
    return None if payload is None else payload["question"]


def advance(graph, config: dict, graph_input, on_turn: Callable[[dict], None] | None = None) -> dict:


    seen = len(graph.get_state(config).values.get("dialogue", []))
    for state in graph.stream(graph_input, config=config, stream_mode="values"):
        dialogue = state["dialogue"]
        for turn in dialogue[seen:]:
            if on_turn:
                on_turn(turn)
        seen = max(seen, len(dialogue))
    return graph.get_state(config).values


def start(graph, task_text: str, config: dict, on_turn=None) -> dict:
    return advance(graph, config, {"dialogue": [{"role": "user", "text": task_text}]}, on_turn)


def resume(graph, human_reply, config: dict, on_turn=None) -> dict:

    return advance(graph, config, Command(resume=human_reply), on_turn)


def run_with_human(graph, task_text: str, config: dict, ask: Callable[[dict], Any], on_turn=None) -> dict:

    state = start(graph, task_text, config, on_turn)
    while (payload := pending_interrupt(graph, config)) is not None:
        state = resume(graph, ask(payload), config, on_turn)
    return state


def ask_in_terminal(payload: dict):
    print(f"\n===== QUESTION FOR HUMAN ({payload['kind']}) =====\n{payload['question']}", flush=True)
    if payload["kind"] == "clarification":
        return input("Your reply> ")

    action = payload["action"]
    if payload["kind"] == "final_approval":
        text = input("Your reply> ")
        confirm = input(f"Type 'approve' to report exactly {action['action_id']}; anything else refuses> ")
        return approval(action["action_id"], confirm.strip().lower() == "approve", text)

    print(
        f"\nACTION TO APPROVE\n"
        f"action_id:   {action['action_id']}\n"
        f"operation:   {action['operation']}\n"
        f"code_sha256: {action['code_sha256']}\n"
        f"----- code -----\n{action['code']}----- end of code -----",
        flush=True,
    )
    text = input("Your reply> ")
    confirm = input(f"Type 'approve' to authorise exactly {action['action_id']}; anything else refuses> ")
    return approval(action["action_id"], confirm.strip().lower() == "approve", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hw2.task1_hitl",
        description=(
            "HW2 Task 1: Task 6 agentic system with a model-selected human destination "
            "and human approval before every coding-tool run."
        ),
    )
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--task",
        default=str(REPO_ROOT / "hw2/silverbox_request.txt"),
        help="Text file containing the task given to the orchestrator.",
    )
    task_group.add_argument("--prompt", default=None, help="Task text given directly.")
    parser.add_argument("--data", default=None, help="Path to SNLS80mV.mat (default: located automatically).")
    parser.add_argument(
        "--workdir", default=None,
        help="Directory for executor scripts and logs (default: runs/hw2_<timestamp>).",
    )
    parser.add_argument(
        "--model", default=os.environ.get("AGENT_MODEL", "openrouter/auto" if os.environ.get("OPENROUTER_API_KEY") else "gpt-6-luna"),
        help="API model used for role decisions and role responses.",
    )
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID, help="LangGraph thread_id.")
    parser.add_argument(
        "--with-history", action="store_true",
        help="Show the orchestrator the recorded Homework 1 Silverbox results (read-only).",
    )
    args = parser.parse_args(argv)

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        print("ERROR: Set OPENAI_API_KEY or OPENROUTER_API_KEY.", file=sys.stderr)
        return 2

    if args.prompt is not None:
        task_text = args.prompt
    else:
        task_path = Path(args.task).resolve()
        if not task_path.exists():
            print(f"ERROR: task file not found: {task_path}", file=sys.stderr)
            return 2
        task_text = task_path.read_text(encoding="utf-8")

    data_path = hw1.locate_data(args.data)
    if data_path is None or not data_path.exists():
        print("ERROR: SNLS80mV.mat not found; pass --data.", file=sys.stderr)
        return 2
    workdir = Path(args.workdir or f"runs/hw2_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    try:
        ensure_fresh_workdir(workdir)
    except OverwriteRefused as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    workdir.mkdir(parents=True, exist_ok=True)

    history = load_historical_results() if args.with_history else None
    system = HumanInTheLoopSystem(ModelCaller(args.model), workdir=workdir, data_path=data_path, history=history)
    graph = system.build()
    config = thread_config(args.thread_id)

    print(f"Model: {args.model}\nData: {data_path}\nWorkdir: {workdir}\nThread: {args.thread_id}")
    print("Starting LangGraph run...\n", flush=True)

    def on_turn(turn: dict) -> None:
        print(f"\n===== {turn['role'].upper()} =====\n{turn['text']}", flush=True)
        save_dialogue(graph.get_state(config).values.get("dialogue", []), workdir)

    state = run_with_human(graph, task_text, config, ask_in_terminal, on_turn)
    dialogue = state["dialogue"]
    save_dialogue(dialogue, workdir)

    decisions = [json.loads(t["text"])["next_role"] for t in dialogue if t["role"] == "orchestrator"]
    print("\n===== FINAL STATUS =====")
    print(f"Turn sequence: {' -> '.join(t['role'] for t in dialogue)}")
    print(f"Orchestrator decisions ({len(decisions)}): {decisions}")
    print(f"Last coding action: {describe_action(state.get('pending_action'))}")
    final_report = state.get("final_report")
    if final_report is None:
        print("No human-approved final measurement; nothing is reported as final.")
        print(f"Logs saved to: {workdir}")
        return 2
    path = write_final_report(workdir, final_report)
    print(f"FINAL_TEST_RMSE={final_report['value']} {final_report['units']} "
          f"(approved action {final_report['approved_action_id']}, run {final_report['run_file']})")
    print(f"Final measurement saved to: {path}")
    print(f"Logs saved to: {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
