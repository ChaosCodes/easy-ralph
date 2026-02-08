"""Test 5: AskUserQuestion with can_use_tool callback (streaming mode).

Tests can_use_tool callback with streaming prompt to fix permission issue.
"""
import asyncio
import json
import logging
import sys
from collections.abc import AsyncIterator
from typing import Any

import anyio

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    UserMessage,
    query,
)

# Enable debug logging for the SDK
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr, format="%(name)s:%(levelname)s: %(message)s")


def log(msg: str):
    print(msg, flush=True)


async def allow_ask_user_question(tool_name: str, tool_input: dict, context) -> PermissionResultAllow | PermissionResultDeny:
    """Allow AskUserQuestion, deny everything else."""
    log(f"  [PERMISSION] {tool_name} → {'ALLOW' if tool_name == 'AskUserQuestion' else 'DENY'}")
    if tool_name == "AskUserQuestion":
        return PermissionResultAllow()
    return PermissionResultDeny(message=f"Tool '{tool_name}' not allowed")


async def prompt_stream() -> AsyncIterator[dict[str, Any]]:
    """Yield user message, then keep stream alive."""
    log("  [STREAM] Yielding user message...")
    yield {
        "type": "user",
        "session_id": "",
        "message": {
            "role": "user",
            "content": "用 AskUserQuestion 问用户喜欢什么颜色（红/蓝/绿）。收到答案后说'谢谢，你选了 X'然后停止。",
        },
        "parent_tool_use_id": None,
    }
    log("  [STREAM] Message yielded, keeping stream alive...")
    try:
        await anyio.sleep(3600)
    except BaseException as e:
        log(f"  [STREAM] Sleep interrupted: {type(e).__name__}")


async def main():
    log("=== Test 5: AskUserQuestion + can_use_tool callback ===\n")

    messages = []
    tool_log = []
    error_count = 0

    async for msg in query(
        prompt=prompt_stream(),
        options=ClaudeAgentOptions(
            system_prompt="你是助手。用 AskUserQuestion 工具问用户问题。收到答案后说'谢谢，你选了 X'然后停止。",
            allowed_tools=["AskUserQuestion"],
            can_use_tool=allow_ask_user_question,
            max_turns=5,
            cwd=".",
        ),
    ):
        msg_type = type(msg).__name__
        log(f"  [MSG] {msg_type}")
        messages.append(msg)

        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if hasattr(block, "text") and block.text:
                    text = block.text.strip()
                    if text:
                        log(f"    [AI] {text[:200]}")
                if hasattr(block, "name"):
                    tool_log.append(block.name)
                    input_str = json.dumps(getattr(block, "input", {}), ensure_ascii=False)[:300]
                    log(f"    [TOOL] {block.name}: {input_str}")

        elif isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, list):
                for item in content:
                    if hasattr(item, "content"):
                        result_str = str(item.content)[:200]
                        is_err = getattr(item, "is_error", False)
                        if is_err:
                            error_count += 1
                            log(f"    [ERROR] {result_str}")
                        else:
                            log(f"    [RESULT] {result_str}")

        elif isinstance(msg, ResultMessage):
            log(f"\n  [DONE] turns={msg.num_turns}, cost=${msg.total_cost_usd:.4f}")

    log(f"\nTool calls: {' -> '.join(tool_log)}, Errors: {error_count}")
    log("✅ SUCCESS" if error_count == 0 and "AskUserQuestion" in tool_log else "❌ FAILED")


if __name__ == "__main__":
    asyncio.run(main())
