"""Filesystem-grep baseline: the null memory system.

Constructs nothing. The reader gets two tools — grep over the transcript and
read-session — with a hard cap on tool rounds, then must answer. Letta showed
this shape scores ~74% on LoCoMo; it is the honest null hypothesis against
"memory construction earns its cost". Its extra reader-side LLM calls are
counted and reported — grep may be a good score, not a cheap one.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from cognitrace.harness.schema import QAItem, Session

MAX_TOOL_ROUNDS = 4
_GREP_MAX_LINES = 30
_SESSION_MAX_CHARS = 8000

_SYSTEM = (
    "You answer questions about a user from their conversation history, which "
    "is stored as dated session transcripts you can search. Use the tools to "
    "find evidence, then answer concisely with the specific fact requested. "
    "You have a limited number of tool calls; use them well. If the history "
    "does not contain the answer, say exactly: I don't know."
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Case-insensitive substring/regex search over every "
                           "turn of every session. Returns matching lines as "
                           "[session @ date] speaker: text",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_session",
            "description": "Read one session transcript in full by session_id.",
            "parameters": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
        },
    },
]


@dataclass
class AgentAnswer:
    text: str
    ok: bool
    error: str | None
    usage_in: int
    usage_out: int
    llm_calls: int
    tool_calls: int
    session_ids: list[str] = field(default_factory=list)  # touched, in order
    turn_ids: list[str] = field(default_factory=list)


class GrepAgent:
    def __init__(self, sessions: list[Session], max_tool_rounds: int = MAX_TOOL_ROUNDS):
        self.sessions = {s.session_id: s for s in sessions}
        self.max_tool_rounds = max_tool_rounds
        self._index = f"Available sessions:\n" + "\n".join(
            f"- {s.session_id}" + (f" ({s.date})" if s.date else "")
            for s in sessions
        )

    def _grep(self, pattern: str, touched_sessions: list[str], touched_turns: list[str]) -> str:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        lines = []
        for s in self.sessions.values():
            for t in s.turns:
                if rx.search(t.content):
                    lines.append(
                        f"[{s.session_id}{' @ ' + s.date if s.date else ''}] {t.role}: {t.content}"
                    )
                    if s.session_id not in touched_sessions:
                        touched_sessions.append(s.session_id)
                    if t.turn_id:
                        touched_turns.append(t.turn_id)
                    if len(lines) >= _GREP_MAX_LINES:
                        return "\n".join(lines) + "\n(truncated)"
        return "\n".join(lines) if lines else "(no matches)"

    def _read_session(self, session_id: str, touched_sessions: list[str], touched_turns: list[str]) -> str:
        s = self.sessions.get(session_id)
        if s is None:
            return f"(no such session: {session_id})"
        if session_id not in touched_sessions:
            touched_sessions.append(session_id)
        touched_turns.extend(t.turn_id for t in s.turns if t.turn_id)
        body = "\n".join(f"{t.role}: {t.content}" for t in s.turns)
        header = f"=== Session {s.session_id}" + (f" ({s.date})" if s.date else "") + " ==="
        text = f"{header}\n{body}"
        return text[:_SESSION_MAX_CHARS] + ("\n(truncated)" if len(text) > _SESSION_MAX_CHARS else "")

    def answer(self, qa: QAItem, model: str, seed: int | None = None) -> AgentAnswer:
        from cognitrace.harness.reader import _openai_client  # shared client/key handling

        user = self._index + "\n\n"
        if qa.question_date:
            user += f"Current date: {qa.question_date}\n"
        user += f"Question: {qa.question}"
        messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]

        usage_in = usage_out = llm_calls = tool_calls = 0
        touched_sessions: list[str] = []
        touched_turns: list[str] = []
        try:
            client = _openai_client()
            for round_no in range(self.max_tool_rounds + 1):
                final_round = round_no == self.max_tool_rounds
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=_TOOLS,
                    tool_choice="none" if final_round else "auto",
                    temperature=0,
                    seed=seed,
                )
                llm_calls += 1
                u = getattr(resp, "usage", None)
                usage_in += getattr(u, "prompt_tokens", 0) or 0
                usage_out += getattr(u, "completion_tokens", 0) or 0
                msg = resp.choices[0].message
                if not msg.tool_calls:
                    return AgentAnswer(
                        text=(msg.content or "").strip(), ok=True, error=None,
                        usage_in=usage_in, usage_out=usage_out,
                        llm_calls=llm_calls, tool_calls=tool_calls,
                        session_ids=touched_sessions, turn_ids=touched_turns,
                    )
                messages.append(msg.model_dump(exclude_none=True))
                for call in msg.tool_calls:
                    tool_calls += 1
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if call.function.name == "grep":
                        result = self._grep(str(args.get("pattern", "")), touched_sessions, touched_turns)
                    elif call.function.name == "read_session":
                        result = self._read_session(str(args.get("session_id", "")), touched_sessions, touched_turns)
                    else:
                        result = f"(unknown tool: {call.function.name})"
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            # Unreachable: the final round forces tool_choice="none".
            raise RuntimeError("tool loop did not terminate")
        except Exception as exc:  # noqa: BLE001 - typed as status, not answer
            time.sleep(0)  # keep failure path cheap; retries live in the caller's rerun
            return AgentAnswer(
                text="", ok=False, error=f"{type(exc).__name__}: {exc}",
                usage_in=usage_in, usage_out=usage_out,
                llm_calls=llm_calls, tool_calls=tool_calls,
                session_ids=touched_sessions, turn_ids=touched_turns,
            )


def build(sessions: list[Session], max_tool_rounds: int = MAX_TOOL_ROUNDS) -> GrepAgent:
    return GrepAgent(sessions, max_tool_rounds=max_tool_rounds)
