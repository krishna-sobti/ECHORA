#!/usr/bin/env python3
import argparse
import json
import math
import os
import socket
import struct
import time
from datetime import datetime, timezone

CSI_MAGIC = 0xC5110001


def udp_payload(packet):
    # Ethernet II frame, including optional VLAN tags
    if len(packet) < 14:
        return None

    ethertype = struct.unpack_from("!H", packet, 12)[0]
    offset = 14

    while ethertype in (0x8100, 0x88A8, 0x9100):
        if len(packet) < offset + 4:
            return None
        ethertype = struct.unpack_from("!H", packet, offset + 2)[0]
        offset += 4

    # IPv4 + UDP
    if ethertype != 0x0800 or len(packet) < offset + 20:
        return None

    ip = packet[offset:]
    if ip[0] >> 4 != 4:
        return None

    ihl = (ip[0] & 0x0F) * 4
    if ihl < 20 or len(ip) < ihl + 8 or ip[9] != 17:
        return None

    frag = struct.unpack_from("!H", ip, 6)[0]
    if frag & 0x1FFF:  # ignore non-initial IP fragments
        return None

    udp = offset + ihl
    src_port, dst_port, udp_len = struct.unpack_from("!HHH", packet, udp)
    if udp_len < 8 or len(packet) < udp + udp_len:
        return None

    src_ip = socket.inet_ntoa(ip[12:16])
    dst_ip = socket.inet_ntoa(ip[16:20])
    return src_ip, dst_ip, src_port, dst_port, packet[udp + 8:udp + udp_len]


def parse_csi(data, src_ip):
    if len(data) < 20:
        return None

    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != CSI_MAGIC:
        return None

    node_id = data[4]
    antennas = data[5]
    subcarriers = struct.unpack_from("<H", data, 6)[0]
    freq_mhz = struct.unpack_from("<I", data, 8)[0]
    sequence = struct.unpack_from("<I", data, 12)[0]
    rssi = struct.unpack_from("b", data, 16)[0]
    noise_floor = struct.unpack_from("b", data, 17)[0]
    ppdu_type = data[18]
    flags = data[19]

    if not antennas or not subcarriers:
        return None

    iq_len = antennas * subcarriers * 2
    expected_len = 20 + iq_len
    if len(data) < expected_len:
        return None

    iq = data[20:expected_len]
    amplitudes = []
    for i in range(0, len(iq), 2):
        imag = struct.unpack_from("b", iq, i)[0]
        real = struct.unpack_from("b", iq, i + 1)[0]
        amplitudes.append(round(math.hypot(real, imag), 2))

    now_ns = time.time_ns()
    return {
        "type": "raw_csi",
        "timestamp": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z"),
        "ts_ns": now_ns,
        "source_ip": src_ip,
        "node_id": node_id,
        "sequence": sequence,
        "rssi": rssi,
        "noise_floor": noise_floor,
        "freq_mhz": freq_mhz,
        "ppdu_type": ppdu_type,
        "flags": flags,
        "antennas": antennas,
        "subcarriers": subcarriers,
        "amplitudes": amplitudes,
        "iq_hex": iq.hex(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Capture ESP32 CSI UDP frames without binding UDP port 5005"
    )
    parser.add_argument("--interface", default="wlan0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--output", default="data/recordings/echora_raw_test")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, f"csi-{int(time.time())}.csi.jsonl")

    # AF_PACKET sees packets on the interface without binding UDP port 5005.
    sock = socket.socket(
        socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003)
    )
    sock.bind((args.interface, 0))
    sock.settimeout(1.0)

    seen_udp = 0
    valid_csi = 0
    rejected_csi = 0
    nodes = set()
    start = time.monotonic()

    print(f"Interface: {args.interface} | UDP destination port: {args.port}")
    print(f"Duration: {args.duration}s | Output: {path}")
    print("Waiting for packets...")

    try:
        with open(path, "w", buffering=1) as out:
            while time.monotonic() - start < args.duration:
                try:
                    packet, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue

                decoded = udp_payload(packet)
                if not decoded:
                    continue

                src_ip, _, _, dst_port, payload = decoded
                if dst_port != args.port:
                    continue

                seen_udp += 1
                if len(payload) < 4:
                    continue

                magic = struct.unpack_from("<I", payload, 0)[0]
                if magic != CSI_MAGIC:
                    continue

                frame = parse_csi(payload, src_ip)
                if frame is None:
                    rejected_csi += 1
                    continue

                out.write(json.dumps(frame, separators=(",", ":")) + "\n")
                valid_csi += 1
                nodes.add(frame["node_id"])

                if valid_csi % 100 == 0:
                    print(
                        f"Valid CSI: {valid_csi} | UDP: {seen_udp} | "
                        f"Rejected CSI: {rejected_csi} | Nodes: {sorted(nodes)}"
                    )
    except KeyboardInterrupt:
        print("\nCapture stopped by user.")
    finally:
        sock.close()

    size = os.path.getsize(path)
    print("\n=== Capture result ===")
    print(f"UDP packets on port: {seen_udp}")
    print(f"Valid CSI frames saved: {valid_csi}")
    print(f"CSI-looking frames rejected: {rejected_csi}")
    print(f"Nodes seen: {sorted(nodes)}")
    print(f"File size: {size} bytes")
    print(f"Output: {path}")


if __name__ == "__main__":
    main()
