import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class I2CPacket:
    def __init__(self, address, rw=0):
        self.address = address & 0x7F
        self.rw = rw & 0x01
        self.events = []

    def add_event(self, scl, sda, label):
        self.events.append({
            "SCL": scl,
            "SDA": sda,
            "label": label
        })

    def start_condition(self):
        self.add_event(1, 1, "IDLE")
        self.add_event(1, 0, "START")

    def stop_condition(self):
        self.add_event(0, 0, "PRE_STOP")
        self.add_event(1, 0, "STOP_SETUP")
        self.add_event(1, 1, "STOP")

    def clock_bit(self, bit, label):
        self.add_event(0, bit, label)
        self.add_event(1, bit, label)
        self.add_event(0, bit, label)

    def clock_stretch(self, cycles=3):
        current_sda = self.events[-1]["SDA"] if self.events else 1
        for _ in range(cycles):
            self.add_event(0, current_sda, "CLOCK_STRETCH")

    def send_byte(self, byte, ack=True, stretch=False):
        bits = [(byte >> i) & 1 for i in range(7, -1, -1)]

        for bit in bits:
            self.clock_bit(bit, f"BIT {bit}")

        if stretch:
            self.clock_stretch(cycles=5)

        ack_bit = 0 if ack else 1
        self.clock_bit(ack_bit, "ACK" if ack else "NACK")

    def encode(self, data_bytes, nack_index=None, stretch_index=None):
        self.events = []
        self.start_condition()

        address_byte = (self.address << 1) | self.rw
        self.send_byte(
            address_byte,
            ack=(nack_index != -1),
            stretch=(stretch_index == -1)
        )

        for index, byte in enumerate(data_bytes):
            should_ack = index != nack_index
            should_stretch = index == stretch_index
            self.send_byte(byte, ack=should_ack, stretch=should_stretch)

        self.stop_condition()
        return self.events


def pack_sensor_data(accel_x, accel_y, accel_z):
    return list(struct.pack(">hhh", accel_x, accel_y, accel_z))


def decode_i2c(events):
    sampled_bits = []

    for i in range(1, len(events) - 1):
        previous = events[i - 1]
        current = events[i]

        rising_edge = previous["SCL"] == 0 and current["SCL"] == 1
        if rising_edge and "CLOCK_STRETCH" not in current["label"]:
            sampled_bits.append(current["SDA"])

    bytes_out = []
    ack_bits = []

    for i in range(0, len(sampled_bits), 9):
        group = sampled_bits[i:i + 9]

        if len(group) < 9:
            continue

        byte_bits = group[:8]
        ack = group[8]

        value = 0
        for bit in byte_bits:
            value = (value << 1) | bit

        bytes_out.append(value)
        ack_bits.append(ack)

    if not bytes_out:
        return None

    address_byte = bytes_out[0]
    address = address_byte >> 1
    rw = address_byte & 1
    data = bytes_out[1:]

    return {
        "address": address,
        "rw": rw,
        "data": data,
        "ack_bits": ack_bits
    }


def plot_i2c(events):
    t = np.arange(len(events))
    scl = np.array([e["SCL"] for e in events])
    sda = np.array([e["SDA"] for e in events])

    plt.figure(figsize=(14, 6))

    plt.step(t, scl + 2, where="post", label="SCL")
    plt.step(t, sda, where="post", label="SDA")

    plt.yticks([0, 1, 2, 3], ["SDA LOW", "SDA HIGH", "SCL LOW", "SCL HIGH"])
    plt.xlabel("Time step")
    plt.title("I2C SDA/SCL Timing Diagram")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def animate_i2c(events):
    t = np.arange(len(events))
    scl = np.array([e["SCL"] for e in events])
    sda = np.array([e["SDA"] for e in events])

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.set_xlim(0, len(events))
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["SDA LOW", "SDA HIGH", "SCL LOW", "SCL HIGH"])
    ax.set_xlabel("Time step")
    ax.set_title("Animated I2C Bus: SDA and SCL")
    ax.grid(True)

    line_scl, = ax.step([], [], where="post", label="SCL")
    line_sda, = ax.step([], [], where="post", label="SDA")
    label_text = ax.text(0.02, 0.92, "", transform=ax.transAxes)

    ax.legend()

    def update(frame):
        line_scl.set_data(t[:frame], scl[:frame] + 2)
        line_sda.set_data(t[:frame], sda[:frame])

        if frame < len(events):
            label_text.set_text(f"Event: {events[frame]['label']}")

        return line_scl, line_sda, label_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(events),
        interval=80,
        blit=True,
        repeat=False
    )

    plt.show()


def main():
    address = 0x68

    accel_x = 1200
    accel_y = -450
    accel_z = 16384

    sensor_bytes = pack_sensor_data(accel_x, accel_y, accel_z)

    print("Original sensor values:")
    print("Accel X:", accel_x)
    print("Accel Y:", accel_y)
    print("Accel Z:", accel_z)

    print("\nEncoded data bytes:")
    print([hex(b) for b in sensor_bytes])

    i2c = I2CPacket(address=address, rw=0)

    events = i2c.encode(
        sensor_bytes,
        nack_index=2,
        stretch_index=1
    )

    decoded = decode_i2c(events)

    print("\nDecoded packet:")
    print(decoded)

    decoded_data = decoded["data"]

    if len(decoded_data) >= 6:
        recovered = struct.unpack(">hhh", bytes(decoded_data[:6]))

        print("\nRecovered sensor values:")
        print("Accel X:", recovered[0])
        print("Accel Y:", recovered[1])
        print("Accel Z:", recovered[2])

    print("\nACK/NACK bits:")
    for i, bit in enumerate(decoded["ack_bits"]):
        print(f"Byte {i}: {'ACK' if bit == 0 else 'NACK'}")

    print("\nShowing I2C waveform plot...")
    plot_i2c(events)

    print("Showing I2C animation...")
    animate_i2c(events)


if __name__ == "__main__":
    main()