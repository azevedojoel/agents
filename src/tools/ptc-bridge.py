#!/usr/bin/env python3
"""
Local PTC bridge: runs LLM Python code with async tool stubs.
IPC with parent (Node) via lines on real stdout prefixed with __PTC_IPC__ + JSON.
User print()/stderr are captured and returned in the completed payload.
"""
from __future__ import annotations

import asyncio
import json
import sys
import io
import textwrap
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

IPC_PREFIX = "__PTC_IPC__"

# Must match packages/agents ProgrammaticToolCalling.ts PYTHON_KEYWORDS + normalize rules
PYTHON_KEYWORDS = frozenset(
    {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    }
)


def normalize_python_identifier(name: str) -> str:
    normalized = name.replace("-", "_").replace(" ", "_")
    out = []
    for ch in normalized:
        if ch.isalnum() or ch == "_":
            out.append(ch)
    normalized = "".join(out)
    if normalized and normalized[0].isdigit():
        normalized = "_" + normalized
    if normalized in PYTHON_KEYWORDS:
        normalized = normalized + "_tool"
    return normalized


def emit_ipc(obj: dict[str, Any]) -> None:
    sys.__stdout__.write(IPC_PREFIX + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def read_tool_results_line() -> dict[str, Any]:
    line = sys.stdin.readline()
    if not line:
        raise EOFError("Parent closed stdin before sending tool results")
    line = line.strip()
    if not line.startswith(IPC_PREFIX):
        raise ValueError(f"Expected IPC-prefixed line from parent, got: {line[:120]!r}")
    return json.loads(line[len(IPC_PREFIX) :])


_call_counter = 0


def make_tool_stub(original_name: str, lock: asyncio.Lock):
    async def stub(**kwargs: Any) -> Any:
        global _call_counter
        async with lock:
            _call_counter += 1
            call_id = f"call_{_call_counter:03d}"
            emit_ipc(
                {
                    "status": "tool_call_required",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "name": original_name,
                            "input": dict(kwargs),
                        }
                    ],
                }
            )
            data = read_tool_results_line()
            results = data.get("tool_results")
            if not isinstance(results, list):
                raise RuntimeError("Invalid tool_results from parent")
            for r in results:
                if r.get("call_id") == call_id:
                    if r.get("is_error"):
                        raise RuntimeError(r.get("error_message") or "Tool execution failed")
                    return r.get("result")
            raise RuntimeError(f"No tool result for {call_id}")

    return stub


async def run_user_code(code: str, tools: list[dict[str, Any]]) -> tuple[str, str]:
    lock = asyncio.Lock()
    ns: dict[str, Any] = {"__builtins__": __builtins__, "asyncio": asyncio}
    used_py: set[str] = set()

    for t in tools:
        if not isinstance(t, dict) or "name" not in t:
            continue
        orig = t["name"]
        if not isinstance(orig, str):
            continue
        binding = t.get("python_binding")
        if isinstance(binding, str) and binding.strip():
            py_name = binding.strip()
        else:
            py_name = normalize_python_identifier(orig)
        base = py_name
        n = 2
        while py_name in used_py:
            py_name = f"{base}_{n}"
            n += 1
        used_py.add(py_name)
        ns[py_name] = make_tool_stub(orig, lock)

    body = code.strip()
    # Name must match lookup below (__ptc_main__ with trailing dunders).
    wrapped = "async def __ptc_main__():\n" + textwrap.indent(body, "    ") + "\n"
    exec(compile(wrapped, "<ptc_user>", "exec"), ns, ns)
    main = ns["__ptc_main__"]
    if not asyncio.iscoroutinefunction(main):
        raise TypeError("__ptc_main__ must be async")

    user_out = io.StringIO()
    user_err = io.StringIO()
    with redirect_stdout(user_out), redirect_stderr(user_err):
        await main()
    return user_out.getvalue(), user_err.getvalue()


def main() -> None:
    try:
        first = sys.stdin.readline()
        if not first:
            emit_ipc({"status": "error", "error": "No bootstrap payload on stdin"})
            return
        first = first.strip()
        payload = json.loads(first)
        code = payload.get("code")
        tools = payload.get("tools")
        if not isinstance(code, str) or not code.strip():
            emit_ipc({"status": "error", "error": "Missing or empty code"})
            return
        if not isinstance(tools, list):
            tools = []

        stdout_text, stderr_text = asyncio.run(run_user_code(code, tools))
        emit_ipc(
            {
                "status": "completed",
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
        )
    except Exception as e:
        tb = traceback.format_exc()
        msg = f"{e}\n{tb}" if str(e) else tb
        emit_ipc({"status": "error", "error": msg})


if __name__ == "__main__":
    main()
