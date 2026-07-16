from rich.console import Console
from rich.table import Table

console = Console()

class Cache:
    def __init__(self, size):
        self.size = size
        self.store = {}
        self.time = 0

    def access(self, addr):
        self.time += 1
        if addr in self.store:
            self.store[addr] = self.time
            return True
        if len(self.store) >= self.size:
            lru = min(self.store, key=self.store.get)
            del self.store[lru]
        self.store[addr] = self.time
        return False

class CacheSim:
    def __init__(self, l1_size=4, l2_size=8):
        self.l1 = Cache(l1_size)
        self.l2 = Cache(l2_size)
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
        self.total = 0

    def access(self, addr):
        self.total += 1

        if self.l1.access(addr):
            self.l1_hits += 1
            return "L1"

        if self.l2.access(addr):
            self.l2_hits += 1
            self.l1.access(addr)
            return "L2"

        self.misses += 1
        self.l2.access(addr)
        self.l1.access(addr)
        return "MISS"

    def reset_stats(self):
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
        self.total = 0

    def stats(self):
        l1 = self.l1_hits / self.total if self.total else 0
        l2 = self.l2_hits / self.total if self.total else 0
        return l1, l2

def visualize(pattern, sim, title):
    table = Table(title=title)
    table.add_column("Step")
    table.add_column("Address")
    table.add_column("Result")

    for i, addr in enumerate(pattern):
        res = sim.access(addr)

        if res == "L1":
            color = "[green]L1 HIT[/green]"
        elif res == "L2":
            color = "[yellow]L2 HIT[/yellow]"
        else:
            color = "[red]MISS[/red]"

        table.add_row(str(i), str(addr), color)

    console.print(table)
    l1, l2 = sim.stats()
    console.print(f"L1 Hit Rate: {l1:.2f}")
    console.print(f"L2 Hit Rate: {l2:.2f}")

pattern_0 = list(range(20))
pattern_100 = [1] * 20

sim1 = CacheSim()
visualize(pattern_0, sim1, "0% L1 hit pattern")

sim2 = CacheSim()
sim2.access(1)
sim2.reset_stats()
visualize(pattern_100, sim2, "100% L1 hit pattern")