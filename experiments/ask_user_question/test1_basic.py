"""Test 1: Does AskUserQuestion work through query()?

Minimal test — just ask one question and observe:
- Does the interactive form appear in terminal?
- What message types flow through the async for loop?
- Does UserMessage appear with the user's answer?
"""
import asyncio

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    UserMessage,
    query,
)


async def main():
    print("=== Test 1: Basic AskUserQuestion ===")
    print("Observe: does an interactive form appear below?\n")

    async for msg in query(
        prompt="使用 AskUserQuestion 问用户一个问题：你喜欢什么颜色？提供红、蓝、绿三个选项。",
        options=ClaudeAgentOptions(
            system_prompt="你是助手。用 AskUserQuestion 工具问用户问题。收到答案后说'谢谢，你选了 X'然后停止。",
            allowed_tools=["AskUserQuestion"],
            max_turns=5,
            cwd=".",
        ),
    ):
        print(f"\n--- Message type: {type(msg).__name__} ---")

        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if hasattr(block, "text") and block.text:
                    print(f"  TEXT: {block.text[:200]}")
                if hasattr(block, "name"):
                    print(f"  TOOL: {block.name}")
                    if hasattr(block, "input"):
                        import json
                        print(f"  INPUT: {json.dumps(block.input, ensure_ascii=False, indent=2)[:500]}")

        elif isinstance(msg, UserMessage):
            print(f"  CONTENT: {msg.content}")
            if msg.tool_use_result:
                print(f"  TOOL_RESULT: {msg.tool_use_result}")

        elif isinstance(msg, ResultMessage):
            print(f"  RESULT: turns={msg.num_turns}, cost=${msg.total_cost_usd:.4f}")

    print("\n=== Test 1 Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
