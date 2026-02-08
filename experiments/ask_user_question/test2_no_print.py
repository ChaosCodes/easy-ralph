"""Test 2: AskUserQuestion with no interleaved output.

Same as test1 but collects messages silently — test if our console.print
interfering with stdin/stdout causes the interaction failure.

Observation goals:
- Does the AskUserQuestion form still appear when we don't print?
- Is there a difference vs test1?
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
    print("=== Test 2: AskUserQuestion (silent during query) ===")
    print("No output during query loop — form should have full terminal control.\n")

    messages = []
    async for msg in query(
        prompt="使用 AskUserQuestion 问用户：你喜欢猫还是狗？提供两个选项。收到答案后输出'完成'。",
        options=ClaudeAgentOptions(
            system_prompt="你是助手。用 AskUserQuestion 工具问用户问题。只问一个问题，收到答案后输出'完成'。",
            allowed_tools=["AskUserQuestion"],
            max_turns=5,
            cwd=".",
        ),
    ):
        messages.append(msg)  # collect silently, no print during loop

    # Print results AFTER query completes
    print(f"\nCollected {len(messages)} messages:")
    for i, msg in enumerate(messages):
        print(f"\n[{i}] {type(msg).__name__}")
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if hasattr(block, "text") and block.text:
                    print(f"  text: {block.text[:200]}")
                if hasattr(block, "name"):
                    print(f"  tool: {block.name}")
                    if hasattr(block, "input"):
                        import json
                        print(f"  input: {json.dumps(block.input, ensure_ascii=False, indent=2)[:500]}")
        elif isinstance(msg, UserMessage):
            print(f"  content: {msg.content}")
            if msg.tool_use_result:
                print(f"  tool_result: {msg.tool_use_result}")
        elif isinstance(msg, ResultMessage):
            print(f"  result: turns={msg.num_turns}, cost=${msg.total_cost_usd:.4f}")

    print("\n=== Test 2 Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
