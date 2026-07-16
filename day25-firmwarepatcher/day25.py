import binascii
import struct
import os
import ollama
from rich.console import Console
from rich.table import Table

console = Console()

def setup_environment():
    """Creates the assets folder and a dummy firmware file if they don't exist."""
    if not os.path.exists("assets"):
        os.makedirs("assets")
    
    file_path = "assets/firmware_blob.bin"
    if not os.path.exists(file_path):
        console.print("[bold yellow]![/bold yellow] assets/firmware_blob.bin not found. Generating simulated firmware...")
        # Create a mock binary: Magic Number + Padding + Hidden Creds
        with open(file_path, "wb") as f:
            content = (
                b"\x7fELF" +                   # ELF Header
                b"\x00" * 20 + 
                b"BOOTLOADER_v2.1" +           # System String
                b"\x00" * 40 +
                b"admin_api_key=ORCA_778899" + # Hidden Credential
                b"\x00" * 10
            )
            f.write(content)

def load_firmware(path):
    with open(path, "rb") as f:
        return f.read()

def hex_dump(data, width=16):
    rows = []
    for i in range(0, len(data), width):
        chunk = data[i:i+width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        rows.append((i, hex_part, ascii_part))
    return rows

def display_hex(rows):
    table = Table(title="Firmware Hex Inspection", show_header=True, header_style="bold magenta")
    table.add_column("Offset", style="cyan", no_wrap=True)
    table.add_column("Hex View")
    table.add_column("ASCII", style="green")

    for offset, hex_part, ascii_part in rows:
        table.add_row(f"{offset:08x}", hex_part, ascii_part)
    console.print(table)

def find_strings(data, min_len=4):
    result = []
    current = b""
    for b in data:
        if 32 <= b < 127:
            current += bytes([b])
        else:
            if len(current) >= min_len:
                result.append(current.decode())
            current = b""
    return result

def find_magic_numbers(data):
    patterns = {
        b'\x7fELF': "ELF (Linux Executable)",
        b'MZ': "PE (Windows Executable)",
        b'\x89PNG': "PNG Image",
        b'PK\x03\x04': "ZIP/Archive"
    }
    found = []
    for sig, name in patterns.items():
        idx = data.find(sig)
        if idx != -1:
            found.append((name, idx))
    return found

def find_credentials(strings):
    """Specific detector for baked-in sensitive data."""
    keywords = ["key", "pass", "secret", "token", "admin", "api"]
    return [s for s in strings if any(k in s.lower() for k in keywords)]

def llm_annotate(text_summary):
    """Feeds extracted patterns to Ollama for expert commentary."""
    try:
        prompt = f"As a firmware security expert, analyze these strings found in a binary and identify potential risks:\n{text_summary}"
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        return f"LLM Connection failed: {e}. (Make sure Ollama is running!)"

def main():
    console.print("[bold blue]Starting FirmwarePatcher Analysis...[/bold blue]\n")
    
    # 1. Setup & Load
    setup_environment()
    data = load_firmware("assets/firmware_blob.bin")

    # 2. Display Hex (First 256 bytes)
    rows = hex_dump(data[:256])
    display_hex(rows)

    # 3. Analyze
    strings = find_strings(data)
    magics = find_magic_numbers(data)
    creds = find_credentials(strings)

    # 4. Report Findings
    console.print("\n[bold yellow]─── Pattern Analysis ───[/bold yellow]")
    for name, idx in magics:
        console.print(f"[bold green]✔[/bold green] Found {name} at offset {hex(idx)}")

    console.print("\n[bold red]─── Sensitive Credentials Found ───[/bold red]")
    if creds:
        for c in creds:
            console.print(f"[blink bold red]![/blink bold red] {c}")
    else:
        console.print("No credentials found.")

    # 5. AI Commentary
    console.print("\n[bold cyan]─── LLM Security Commentary ───[/bold cyan]")
    with console.status("[bold green]Asking AI to analyze findings..."):
        summary = ", ".join(strings[:15])
        analysis = llm_annotate(summary)
        console.print(analysis)

if __name__ == "__main__":
    main()