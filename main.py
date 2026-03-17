from boy import llm, tools
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """You are a 3D model generation assistant. Your job is to help users create 3D models from text descriptions.

When a user describes what they want as a 3D model, follow these steps:

1. **Search for images**: Use the search_images tool with a descriptive query to find reference images that best match what the user wants. Think about what search terms will yield clean, well-lit, single-object images with simple backgrounds — these work best for 3D generation.

2. **Pick the best image**: Review the search results and choose the image that:
   - Shows a single, clear object (not a collage or scene)
   - Has good lighting and minimal background clutter
   - Best represents what the user described
   - Has a reasonable resolution (at least 512x512 preferred)

3. **Download the image**: Use download_image to save the chosen image locally.

4. **Generate the 3D mesh**: Use generate_3d_mesh with the downloaded image path to create the 3D model. Choose a descriptive output name.

5. **Report the result**: Tell the user where the .glb file was saved and summarize what was generated.

Important tips:
- For search queries, add terms like "3D reference", "product photo", "white background", "isolated object" to get cleaner images.
- If the first search doesn't yield good results, try refining your query.
- Always tell the user what image you chose and why before generating the 3D model.
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

        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="updates",
        ):
            # Print agent messages as they come
            for node_name, node_output in chunk.items():
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if hasattr(msg, "content") and msg.content:
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print(f"[Using tool: {tc['name']}]")
                            elif msg.type == "ai":
                                print(f"\nAgent: {msg.content}")
                            elif msg.type == "tool":
                                print(f"[Tool result: {msg.content[:200]}...]" if len(msg.content) > 200 else f"[Tool result: {msg.content}]")

        print()


if __name__ == "__main__":
    main()
