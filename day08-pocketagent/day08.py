import subprocess
import os
import sys
import time
import json
import platform

# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():
    """Verify ollama is running and model is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            print("ERROR: ollama is not running.")
            print("Fix: Open another terminal and run: ollama serve")
            sys.exit(1)

        if "qwen2.5:3b" not in result.stdout.lower():
            print("ERROR: qwen2.5:3b model not found.")
            print("Fix: Run: ollama pull qwen2.5:3b")
            sys.exit(1)

        print("✓ ollama is running")
        print("✓ qwen2.5:3b model available")
        return True

    except FileNotFoundError:
        print("ERROR: ollama not installed.")
        print("Fix: Download from https://ollama.com")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: ollama not responding.")
        print("Fix: Restart ollama: ollama serve")
        sys.exit(1)


check_ollama()


# ============================================================
# OLLAMA CHAT FUNCTION
# ============================================================

MODEL = "qwen2.5:3b"


def chat_with_ollama(messages):
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")

    prompt_parts.append("Assistant:")
    full_prompt = "\n".join(prompt_parts)

    start_time = time.time()

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, full_prompt],
            capture_output=True, text=True, timeout=120
        )

        elapsed = time.time() - start_time
        response = result.stdout.strip()

        token_estimate = len(response) / 4
        tps = token_estimate / elapsed if elapsed > 0 else 0

        return response, tps

    except subprocess.TimeoutExpired:
        return "Error: Model timed out. Try a shorter question.", 0
    except Exception as e:
        return f"Error: {e}", 0


# ============================================================
# TOOLS
# ============================================================

def tool_list_directory(path="."):
    """List files and folders in a directory."""
    try:
        items = os.listdir(path)
        dirs = [f"📁 {item}" for item in items if os.path.isdir(os.path.join(path, item))]
        files = [f"📄 {item}" for item in items if os.path.isfile(os.path.join(path, item))]
        result = f"Contents of '{path}':\n"
        result += "\n".join(sorted(dirs) + sorted(files))
        result += f"\n\n({len(dirs)} folders, {len(files)} files)"
        return result
    except Exception as e:
        return f"Error listing '{path}': {e}"


def tool_read_file(filepath):
    """Read the contents of a text file."""
    try:
        with open(filepath, "r") as f:
            content = f.read(2000)
        truncated = " (truncated)" if len(content) >= 2000 else ""
        return f"Contents of '{filepath}'{truncated}:\n\n{content}"
    except Exception as e:
        return f"Error reading '{filepath}': {e}"


def tool_system_info():
    """Get basic system information."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
    }
    result = "System Information:\n"
    for key, value in info.items():
        result += f"  {key}: {value}\n"
    return result


def tool_current_time():
    """Get the current date and time."""
    from datetime import datetime
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def tool_disk_usage():
    """Show disk space usage for the current drive."""
    import shutil
    total, used, free = shutil.disk_usage("/")
    gb = 1024 ** 3
    return (
        f"Disk Usage:\n"
        f"  Total : {total / gb:.1f} GB\n"
        f"  Used  : {used  / gb:.1f} GB\n"
        f"  Free  : {free  / gb:.1f} GB\n"
        f"  Used% : {used/total*100:.1f}%"
    )


# ============================================================
# TOOL REGISTRY
# ============================================================

AVAILABLE_TOOLS = {
    "list_directory": {
        "function": tool_list_directory,
        "description": "List files and folders in a directory",
        "usage": "list_directory [path]",
    },
    "read_file": {
        "function": tool_read_file,
        "description": "Read the contents of a text file",
        "usage": "read_file <filepath>",
    },
    "system_info": {
        "function": tool_system_info,
        "description": "Get system information (OS, Python version, etc)",
        "usage": "system_info",
    },
    "current_time": {
        "function": tool_current_time,
        "description": "Get the current date and time",
        "usage": "current_time",
    },
    "disk_usage": {
        "function": tool_disk_usage,
        "description": "Show disk space usage for the current drive",
        "usage": "disk_usage",
    },
}


# ============================================================
# TOOL ROUTING
# ============================================================

def try_parse_tool_call(response):
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("TOOL:"):
            parts = line[5:].strip().split(maxsplit=1)
            tool_name = parts[0].lower().strip() if parts else None
            argument = parts[1].strip() if len(parts) > 1 else None

            if tool_name in AVAILABLE_TOOLS:
                return tool_name, argument

    return None, None


def execute_tool(tool_name, argument):
    tool = AVAILABLE_TOOLS[tool_name]
    func = tool["function"]

    try:
        if argument:
            return func(argument)
        else:
            return func()
    except Exception as e:
        return f"Tool error: {e}"


# ============================================================
# SYSTEM PROMPT
# ============================================================

tools_description = "\n".join(
    f"  - {name}: {info['description']} (usage: {info['usage']})"
    for name, info in AVAILABLE_TOOLS.items()
)

SYSTEM_PROMPT = f"""You are PocketAgent, a local AI assistant running entirely on this device. You have NO internet access.

You can use these tools:
{tools_description}

RULES — follow these exactly:
1. If the user asks about files, directories, disk, system, or time → use the appropriate tool.
2. To call a tool, respond with EXACTLY one line: TOOL: <tool_name> [optional argument]
3. Do NOT add any other text in the same response as a TOOL: call.
4. Do NOT guess file contents or system info — always use a tool to get real data.
5. For questions you can answer from knowledge (math, coding, general facts), answer directly with no tool.
6. Keep all answers short and factual.

Examples of correct tool calls:
TOOL: list_directory .
TOOL: read_file README.md
TOOL: system_info
TOOL: disk_usage
TOOL: current_time"""


# ============================================================
# MAIN CHAT LOOP
# ============================================================

def print_header():
    print()
    print("=" * 55)
    print("  🤖 PocketAgent — Local AI Assistant")
    print(f"  Model: {MODEL} | Running on: {platform.system()}")
    print("=" * 55)
    print()
    print("  Available tools:")
    for name, info in AVAILABLE_TOOLS.items():
        print(f"    • {name} — {info['description']}")
    print()
    print("  Type a question or command. Type 'quit' to exit.")
    print("  Try: 'What files are in this directory?'")
    print("       'What system am I running on?'")
    print("       'How much disk space do I have?'")
    print()


def main():
    print_header()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        print("\n⏳ Thinking...", end="", flush=True)
        response, tps = chat_with_ollama(messages)
        print(f"\r                    \r", end="")

        tool_name, argument = try_parse_tool_call(response)

        if tool_name:
            print(f"🔧 Using tool: {tool_name}", end="")
            if argument:
                print(f" ({argument})")
            else:
                print()

            tool_output = execute_tool(tool_name, argument)
            print(f"\n{tool_output}\n")

            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Tool result:\n{tool_output}\n\nBriefly explain what this shows."
            })

            print("⏳ Analyzing...", end="", flush=True)
            explanation, tps2 = chat_with_ollama(messages)
            print(f"\r                    \r", end="")

            print(f"Agent > {explanation}")
            tps = tps2
            messages.append({"role": "assistant", "content": explanation})

        else:
            print(f"Agent > {response}")
            messages.append({"role": "assistant", "content": response})

        print(f"\n  ⚡ {tps:.1f} tokens/sec")
        print()

        if len(messages) > 12:
            messages = [messages[0]] + messages[-10:]


if __name__ == "__main__":
    main()
