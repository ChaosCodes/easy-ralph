"""Test 4: AskUserQuestion with permission_mode="bypassPermissions".

Root cause analysis:
- AskUserQuestion fails because ClaudeAgentOptions has no permission handling
- SDK sends can_use_tool control request → no callback → exception → is_error=True
- Agent then hallucinates an answer

This test bypasses all permission checks entirely.

Key questions:
1. Does bypassPermissions fix the permission error?
2. Does the interactive form render in the terminal?
   (CLI has stdin=PIPE, stdout=PIPE — will it use /dev/tty for the form?)
"""
import asyncio
import json

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    UserMessage,
    query,
)


async def main():
    print("=== Test 4: AskUserQuestion + bypassPermissions ===")
    print("Testing: permission_mode='bypassPermissions'")
    print("Observe: does an interactive form appear below?\n")

    messages = []
    tool_log = []
    error_count = 0

    async for msg in query(
        prompt="用 AskUserQuestion 问用户喜欢什么颜色（红/蓝/绿）。收到答案后说'谢谢，你选了 X'然后停止。",
        options=ClaudeAgentOptions(
            system_prompt="你是助手。用 AskUserQuestion 工具问用户问题。收到答案后说'谢谢，你选了 X'然后停止。",
            allowed_tools=["AskUserQuestion"],
            permission_mode="bypassPermissions",
            max_turns=5,
            cwd=".",
        ),
    ):
        messages.append(msg)

        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if hasattr(block, "text") and block.text:
                    text = block.text.strip()
                    if text:
                        print(f"  [AI] {text[:200]}")
                if hasattr(block, "name"):
                    tool_log.append(block.name)
                    input_str = ""
                    if hasattr(block, "input"):
                        input_str = json.dumps(block.input, ensure_ascii=False)[:300]
                    print(f"  [TOOL] {block.name}: {input_str}")

        elif isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, str):
                print(f"  [USER] {content[:200]}")
            elif isinstance(content, list):
                for item in content:
                    if hasattr(item, "content"):
                        result_str = str(item.content)[:200]
                        is_err = getattr(item, "is_error", False)
                        if is_err:
                            error_count += 1
                            print(f"  [ERROR] {result_str}")
                        else:
                            print(f"  [RESULT] {result_str}")
                    elif hasattr(item, "text"):
                        print(f"  [USER_TEXT] {item.text[:200]}")

        elif isinstance(msg, ResultMessage):
            print(f"\n  [DONE] turns={msg.num_turns}, cost=${msg.total_cost_usd:.4f}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Total messages: {len(messages)}")
    print(f"Tool calls: {' -> '.join(tool_log)}")
    print(f"Errors: {error_count}")

    if error_count > 0:
        print("\n❌ FAILED: AskUserQuestion returned errors (permission still blocked)")
    elif "AskUserQuestion" in tool_log:
        print("\n✅ AskUserQuestion was called without permission error")
        print("   Check above: did the interactive form appear?")
    else:
        print("\n⚠️  AskUserQuestion was never called")

    print("\n=== Test 4 Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
