import subprocess
import sys
import time

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    print("pip install rich")
    sys.exit(1)

console = Console()
MODEL = "qwen2.5:3b"


class CPU:
    def __init__(self):
        self.registers = [0] * 8
        self.pc = 0
        self.flags = {"ZERO": False, "NEG": False}
        self.halted = False
        self.program = []

    def reset(self):
        self.__init__()


def alu(op, a, b):
    if op == "ADD":
        return a + b
    if op == "SUB":
        return a - b
    if op == "MUL":
        return a * b
    if op == "AND":
        return a & b
    if op == "OR":
        return a | b
    raise ValueError(f"Unknown ALU op: {op}")


def reg_index(token):
    return int(token[1:])


def is_register(token):
    return token.startswith("R") and token[1:].isdigit()


def get_value(cpu, token):
    if is_register(token):
        return cpu.registers[reg_index(token)]
    return int(token)


def sanitize_program(raw_program):
    cleaned = []
    for line in raw_program:
        if isinstance(line, str):
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if parts:
                cleaned.append(parts)
        elif isinstance(line, list):
            cleaned.append(line)
    return cleaned


def execute(cpu, instruction):
    op = instruction[0].upper()
    args = instruction[1:]

    if op == "MOV":
        dst = reg_index(args[0])
        value = get_value(cpu, args[1])
        cpu.registers[dst] = value
        cpu.pc += 1
        return f"R{dst} <- {value}"

    if op == "ADD":
        dst = reg_index(args[0])
        old = cpu.registers[dst]
        rhs = get_value(cpu, args[1])
        cpu.registers[dst] = alu("ADD", old, rhs)
        cpu.pc += 1
        return f"R{dst} = {old} + {rhs} = {cpu.registers[dst]}"

    if op == "SUB":
        dst = reg_index(args[0])
        old = cpu.registers[dst]
        rhs = get_value(cpu, args[1])
        cpu.registers[dst] = alu("SUB", old, rhs)
        cpu.pc += 1
        return f"R{dst} = {old} - {rhs} = {cpu.registers[dst]}"

    if op == "MUL":
        dst = reg_index(args[0])
        old = cpu.registers[dst]
        rhs = get_value(cpu, args[1])
        cpu.registers[dst] = alu("MUL", old, rhs)
        cpu.pc += 1
        return f"R{dst} = {old} * {rhs} = {cpu.registers[dst]}"

    if op == "CMP":
        left = get_value(cpu, args[0])
        right = get_value(cpu, args[1])
        cpu.flags["ZERO"] = left == right
        cpu.flags["NEG"] = left < right
        cpu.pc += 1
        return f"CMP {left} vs {right} -> ZERO={cpu.flags['ZERO']} NEG={cpu.flags['NEG']}"

    if op == "JMP":
        target = int(args[0])
        cpu.pc = target
        return f"Jumped to instruction {target}"

    if op == "JZ":
        target = int(args[0])
        if cpu.flags["ZERO"]:
            cpu.pc = target
            return f"ZERO flag set, jumped to instruction {target}"
        cpu.pc += 1
        return "ZERO flag not set, no jump"

    if op == "JNZ":
        target = int(args[0])
        if not cpu.flags["ZERO"]:
            cpu.pc = target
            return f"ZERO flag clear, jumped to instruction {target}"
        cpu.pc += 1
        return "ZERO flag set, no jump"

    if op == "HALT":
        cpu.halted = True
        return "CPU halted"

    raise ValueError(f"Unknown instruction: {' '.join(instruction)}")


NARRATOR_PROMPT = """
You are narrating a tiny CPU simulation for a beginner.
Explain exactly one executed instruction in one short sentence.
Be concrete, educational, and concise.
Mention registers, ALU, flags, or control flow only if relevant.
Do not use bullet points.
Do not add filler.

Instruction: {instr}
Effect: {effect}
Registers: {regs}
Flags: ZERO={zero}, NEG={neg}
Program Counter: {pc}

One sentence:
""".strip()


def narrate(instr, effect, cpu):
    regs = ", ".join(f"R{i}={v}" for i, v in enumerate(cpu.registers))
    prompt = NARRATOR_PROMPT.format(
        instr=" ".join(instr),
        effect=effect,
        regs=regs,
        zero=cpu.flags["ZERO"],
        neg=cpu.flags["NEG"],
        pc=cpu.pc,
    )
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def show_state(cpu, step, instr, effect, narration):
    table = Table(title=f"Step {step}: {' '.join(instr)}", show_header=True)
    for i in range(8):
        table.add_column(f"R{i}", justify="center")
    table.add_row(*[str(value) for value in cpu.registers])

    console.print(table)
    console.print(f"[cyan]Effect:[/cyan] {effect}")
    console.print(
        f"[dim]Flags: ZERO={cpu.flags['ZERO']} | NEG={cpu.flags['NEG']} | PC={cpu.pc}[/dim]"
    )

    if narration:
        console.print(Panel(narration, title="Narrator", border_style="green"))

    console.print()


PROGRAM = [
    ["MOV", "R0", "5"],
    ["MOV", "R1", "1"],
    ["MOV", "R2", "0"],
    ["MOV", "R3", "1"],
    ["CMP", "R0", "R2"],
    ["JZ", "9"],
    ["MUL", "R1", "R0"],
    ["SUB", "R0", "R3"],
    ["JMP", "4"],
    ["HALT"],
]


def run_program(program, delay=0.3, max_steps=100):
    cpu = CPU()
    cpu.program = sanitize_program(program)

    console.print("\n[bold cyan]RegisterBot - Tiny CPU Simulator[/bold cyan]")
    console.print("[dim]Running factorial demo: 5![/dim]\n")

    step = 0

    while not cpu.halted and 0 <= cpu.pc < len(cpu.program) and step < max_steps:
        instruction = cpu.program[cpu.pc]
        step += 1

        try:
            effect = execute(cpu, instruction)
        except Exception as error:
            console.print(f"[bold red]Execution error:[/bold red] {error}")
            break

        narration = narrate(instruction, effect, cpu)
        show_state(cpu, step, instruction, effect, narration)
        time.sleep(delay)

    console.print(f"[bold green]Program finished in {step} steps[/bold green]")
    console.print(f"[bold]Final result: R1 = {cpu.registers[1]}[/bold]")

    if cpu.registers[1] == 120:
        console.print("[bold green]Success: factorial of 5 is 120[/bold green]")
    else:
        console.print("[bold yellow]Check logic: expected 120[/bold yellow]")


if __name__ == "__main__":
    run_program(PROGRAM)