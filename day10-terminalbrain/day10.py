import subprocess
import threading
import sys
import queue
import re
import ollama
from rich.console import Console

console = Console()
error_cache = {}
error_queue = queue.Queue()


def normalize_error(err):
    err = re.sub(r'File ".*?", line \d+', '', err)
    err = re.sub(r'\d+', '', err)
    return err.strip()


def get_fix_suggestion(error_text):
    try:
        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": f"Explain this error in 1 sentence and give a fix:\n{error_text}"
                }
            ]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"LLM error: {e}"


def handle_error(error_text):
    norm = normalize_error(error_text)

    if norm in error_cache:
        suggestion = error_cache[norm]
    else:
        suggestion = get_fix_suggestion(error_text)
        error_cache[norm] = suggestion

    console.print("\n[bold yellow]💡 Fix suggestion:[/bold yellow]")
    console.print(suggestion)
    console.print("-" * 50)


def error_worker():
    while True:
        error_text = error_queue.get()
        if error_text is None:
            break
        handle_error(error_text)


def read_stream(stream, is_error=False):
    buffer = []

    for line in iter(stream.readline, ''):
        if not line:
            break

        if is_error:
            console.print(f"[red]{line.rstrip()}[/red]")

            # Start collecting when traceback begins
            if "Traceback" in line:
                buffer = [line]

            elif buffer:
                buffer.append(line)

                # End of error → send full traceback
                if any(k in line for k in ["Error", "Exception"]):
                    error_queue.put("".join(buffer))
                    buffer = []

            else:
                # fallback (single-line errors)
                if any(k in line for k in ["Error", "Exception"]):
                    error_queue.put(line)

        else:
            console.print(line.rstrip())

    stream.close()


def run_command(cmd):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        shell=True
    )

    t_out = threading.Thread(target=read_stream, args=(process.stdout, False))
    t_err = threading.Thread(target=read_stream, args=(process.stderr, True))
    t_worker = threading.Thread(target=error_worker)

    t_worker.start()
    t_out.start()
    t_err.start()

    t_out.join()
    t_err.join()

    error_queue.put(None)
    t_worker.join()
    process.wait()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Usage: python script.py <command>[/red]")
        sys.exit(1)

    command = " ".join(sys.argv[1:])
    run_command(command)