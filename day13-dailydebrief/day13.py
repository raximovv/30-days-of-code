"""
BUILDCORED ORCAS — Day 13: DailyDebrief

Collect your day's activity and get an AI summary.
Hardware concept: Flight Data Recorder

Collect all streams -> compress -> report.

Run:
    python day13.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    from git import Repo, InvalidGitRepositoryError
except ImportError:
    Repo = None
    InvalidGitRepositoryError = Exception

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    print("pip install rich gitpython")
    sys.exit(1)

console = Console()
MODEL = "qwen2.5:3b"


def check_ollama():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "qwen2.5" not in result.stdout.lower():
            console.print("[red]Run: ollama pull qwen2.5:3b[/red]")
            sys.exit(1)
    except Exception:
        console.print("[red]ollama not found. Install Ollama first.[/red]")
        sys.exit(1)


def ensure_git_repo():
    """
    If no git repo exists, create a tiny demo repo so the script still works.
    """
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return
    except Exception:
        pass

    demo_dir = Path.cwd() / "dailydebrief_demo_repo"
    demo_dir.mkdir(exist_ok=True)

    readme = demo_dir / "README.md"
    readme.write_text("# Demo Repo\n\nCreated automatically for DailyDebrief.\n", encoding="utf-8")

    try:
        subprocess.run(["git", "init"], cwd=demo_dir, check=True, capture_output=True, text=True)
        subprocess.run(["git", "add", "."], cwd=demo_dir, check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "commit", "-m", "init demo repo for DailyDebrief"],
            cwd=demo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        os.chdir(demo_dir)
    except Exception:
        # Even if commit fails, continue. The rest of the script can still use other sources.
        os.chdir(demo_dir)


def get_git_commits(hours=24):
    """
    Get recent git commits using GitPython when available.
    """
    since_dt = datetime.now() - timedelta(hours=hours)

    if Repo is not None:
        try:
            repo = Repo(Path.cwd(), search_parent_directories=True)
            commits = []
            for commit in repo.iter_commits("HEAD", max_count=20):
                committed = datetime.fromtimestamp(commit.committed_date)
                if committed >= since_dt:
                    commits.append(f"{commit.hexsha[:7]} {commit.summary}")
            return commits[:20]
        except InvalidGitRepositoryError:
            return []
        except Exception:
            pass

    try:
        since = since_dt.isoformat()
        result = subprocess.run(
            ["git", "log", f"--since={since}", "--pretty=format:%h %s"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return [line for line in result.stdout.splitlines() if line.strip()][:20]
    except Exception:
        return []


def get_recent_files(hours=24, limit=30):
    """
    Files modified in the last N hours under the current working directory first,
    then fallback to home directory.
    """
    roots = [Path.cwd(), Path.home()]
    cutoff = time.time() - (hours * 3600)

    ignore_parts = {
        ".git",
        ".cache",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "Library",
        ".local",
    }

    seen = set()
    recent = []

    for root in roots:
        try:
            for path in root.rglob("*"):
                try:
                    if not path.is_file():
                        continue

                    path_str = str(path)
                    if any(part in path_str for part in ignore_parts):
                        continue

                    if path.stat().st_mtime <= cutoff:
                        continue

                    try:
                        display = str(path.relative_to(root))
                    except ValueError:
                        display = str(path)

                    if display not in seen:
                        seen.add(display)
                        recent.append(display)

                    if len(recent) >= limit:
                        return recent
                except Exception:
                    continue
        except Exception:
            continue

    return recent


def get_shell_history(lines=30):
    """
    Last N shell commands from zsh or bash history.
    """
    candidates = [Path.home() / ".zsh_history", Path.home() / ".bash_history"]

    for hist_path in candidates:
        if not hist_path.exists():
            continue

        try:
            content = hist_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            cleaned = []
            for line in content[-200:]:
                line = line.strip()
                if not line:
                    continue

                # zsh extended history format: ": 1710000000:0;command"
                if ";" in line and line.startswith(": "):
                    line = line.split(";", 1)[-1].strip()

                cleaned.append(line)

            return cleaned[-lines:]
        except Exception:
            continue

    return []


def get_vscode_recent(limit=20):
    """
    4th data source:
    Parse VS Code storage JSON for recently opened workspaces/files.
    """
    possible_files = [
        Path.home() / ".config/Code/User/globalStorage/storage.json",  # Linux
        Path.home() / "Library/Application Support/Code/User/globalStorage/storage.json",  # macOS
        Path.home() / "AppData/Roaming/Code/User/globalStorage/storage.json",  # Windows
    ]

    for storage_path in possible_files:
        if not storage_path.exists():
            continue

        try:
            data = json.loads(storage_path.read_text(encoding="utf-8", errors="ignore"))
            hits = []

            def walk(node):
                if isinstance(node, dict):
                    for key, value in node.items():
                        if isinstance(value, (dict, list)):
                            walk(value)
                        elif isinstance(value, str):
                            lower = value.lower()
                            if any(token in lower for token in ["file://", "vscode", "workspace", ".py", ".md", ".js", ".ts"]):
                                hits.append(value)
                elif isinstance(node, list):
                    for item in node:
                        walk(item)

            walk(data)

            cleaned = []
            seen = set()
            for item in hits:
                pretty = item.replace("file://", "").replace("%20", " ")
                pretty = pretty.replace("vscode-remote://", "")
                if pretty not in seen:
                    seen.add(pretty)
                    cleaned.append(pretty)

            return cleaned[:limit]
        except Exception:
            continue

    return []


DEBRIEF_PROMPT = """
You are analyzing a developer's real daily activity log.

Your task:
Infer what happened during the day from the signals below.

Output rules:
- Output EXACTLY 5 lines.
- Each line must start with one of these labels in ALL CAPS:
  BUILT:
  BROKE:
  LEARNED:
  PATTERN:
  NEXT:
- One line per label only.
- No intro.
- No bullets.
- No markdown fences.
- No extra commentary.
- Be specific and grounded in the data.
- If evidence is weak, say "Likely ..." rather than inventing facts.
- Keep each line under 22 words.

Signals:
{data}
""".strip()


def get_debrief(data_text):
    prompt = DEBRIEF_PROMPT.format(data=data_text[:5000])

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=90,
        )
        output = result.stdout.strip()
        if not output:
            return "[LLM returned empty output]"
        return output
    except Exception as exc:
        return f"[LLM error] {exc}"


def build_combined_data(commits, files, history, vscode_items):
    chunks = []

    if commits:
        chunks.append("GIT COMMITS:\n" + "\n".join(commits[:10]))
    if files:
        chunks.append("FILES MODIFIED:\n" + "\n".join(files[:15]))
    if history:
        chunks.append("SHELL HISTORY:\n" + "\n".join(history[-15:]))
    if vscode_items:
        chunks.append("VS CODE RECENTS:\n" + "\n".join(vscode_items[:15]))

    return "\n\n".join(chunks)


def main():
    console.print("\n[bold cyan]DailyDebrief[/bold cyan]\n")

    check_ollama()
    ensure_git_repo()

    console.print("[dim]Collecting data from the last 24 hours...[/dim]\n")

    commits = get_git_commits()
    files = get_recent_files()
    history = get_shell_history()
    vscode_items = get_vscode_recent()

    stats = Table(show_header=True, header_style="bold cyan")
    stats.add_column("Source")
    stats.add_column("Count", justify="right")
    stats.add_row("Git commits", str(len(commits)))
    stats.add_row("Recent files", str(len(files)))
    stats.add_row("Shell commands", str(len(history)))
    stats.add_row("VS Code recents", str(len(vscode_items)))
    console.print(stats)
    console.print()

    combined = build_combined_data(commits, files, history, vscode_items)

    if not combined.strip():
        console.print("[yellow]No useful data found. Make a few edits or commands, then run again.[/yellow]")
        sys.exit(0)

    console.print("[dim]Asking the brain...[/dim]")
    start = time.time()
    debrief = get_debrief(combined)
    elapsed = time.time() - start

    console.print(
        Panel(
            debrief,
            title=f"Today's Debrief ({elapsed:.1f}s)",
            border_style="cyan",
        )
    )

    console.print("\n[green]Done.[/green]")


if __name__ == "__main__":
    main()