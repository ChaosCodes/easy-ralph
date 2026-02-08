"""Test 3: AskUserQuestion with mixed tools (mimics clarify_requirements).

Tests whether AskUserQuestion works when other tools are also available,
and whether the order of tool calls matters.

Observation goals:
- Does AskUserQuestion work after other tool calls (Glob, Read)?
- Does mixing tools break the interactive form?
- What's the message flow with mixed tools?
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
    print("=== Test 3: Mixed tools + AskUserQuestion ===")
    print("Agent will first explore files, then ask a question.\n")

    messages = []
    tool_log = []

    async for msg in query(
        prompt="先用 Glob 看看当前目录有什么文件，然后用 AskUserQuestion 问用户想创建什么类型的项目（Web/CLI/Library）。收到答案后说'好的，你选择了 X'然后停止。",
        options=ClaudeAgentOptions(
            system_prompt="你是助手。先探索文件，再用 AskUserQuestion 问用户问题。",
            allowed_tools=["Glob", "Grep", "Read", "AskUserQuestion"],
            max_turns=10,
            cwd=".",
        ),
    ):
        messages.append(msg)

        # Minimal logging — only tool names, to avoid interfering with form
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if hasattr(block, "name"):
                    tool_log.append(block.name)
                    # Only print non-AskUserQuestion tools to avoid interference
                    if block.name != "AskUserQuestion":
                        print(f"  [{block.name}]")

    # Print full results after query completes
    print(f"\n--- Results ---")
    print(f"Total messages: {len(messages)}")
    print(f"Tool call order: {' -> '.join(tool_log)}")

    print(f"\n--- Message details ---")
    for i, msg in enumerate(messages):
        print(f"\n[{i}] {type(msg).__name__}")
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if hasattr(block, "text") and block.text:
                    text = block.text.strip()
                    if text:
                        print(f"  text: {text[:200]}")
                if hasattr(block, "name"):
                    input_str = json.dumps(block.input, ensure_ascii=False)[:200] if hasattr(block, "input") else ""
                    print(f"  tool: {block.name} | input: {input_str}")
        elif isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, str):
                print(f"  content: {content[:200]}")
            elif isinstance(content, list):
                for item in content:
                    if hasattr(item, "text"):
                        print(f"  text_block: {item.text[:200]}")
                    elif hasattr(item, "content"):
                        print(f"  tool_result: {str(item.content)[:200]}")
                    else:
                        print(f"  block: {type(item).__name__} = {str(item)[:200]}")
            if msg.tool_use_result:
                print(f"  tool_use_result: {msg.tool_use_result}")
        elif isinstance(msg, ResultMessage):
            print(f"  result: turns={msg.num_turns}, cost=${msg.total_cost_usd:.4f}")

    print("\n=== Test 3 Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
