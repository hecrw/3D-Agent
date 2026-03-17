from boy import llm, tools
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """You are a 3D model generation assistant. You help users create 3D models from text descriptions.

You have these tools:
1. find_and_download_image — searches the internet for an image and downloads it locally
2. generate_3d_mesh — converts a downloaded image into a 3D .glb model

ALWAYS follow these steps in order:

STEP 1: Call find_and_download_image with a good search query and a filename.
- Add words like "product photo", "white background", "single object" to your query for better results.
- Example: find_and_download_image(query="red sports car product photo white background", filename="red_car")

STEP 2: If step 1 returns SUCCESS, call generate_3d_mesh with the file path from step 1.
- Example: generate_3d_mesh(image_path="images/red_car.png", output_name="red_car_3d")

STEP 3: Tell the user the result.

IMPORTANT RULES:
- If a tool returns FAILED, try calling it again with DIFFERENT parameters (different query, different filename).
- NEVER stop without telling the user what happened.
- NEVER skip a step.
- You MUST call the tools. Do not just describe what you would do.
"""

agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


def main():
    print("=== 3D Agent ===")
    print("Describe what you want to generate as a 3D model, or type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nAgent is working...\n")

        try:
            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                stream_mode="updates",
            ):
                for node_name, node_output in chunk.items():
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            if not hasattr(msg, "content") or not msg.content:
                                continue
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print(f"  [Calling: {tc['name']}({tc.get('args', {})})]")
                            elif msg.type == "ai":
                                print(f"\nAgent: {msg.content}")
                            elif msg.type == "tool":
                                preview = msg.content[:300]
                                print(f"  [Result: {preview}]")
        except Exception as e:
            print(f"\nError during agent execution: {e}")

        print()


if __name__ == "__main__":
    main()
