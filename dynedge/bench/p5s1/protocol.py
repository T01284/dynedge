"""P5S1 (星澜科创) voltage/current meter binary protocol v2.

Frame: AA 55 | Type | Len | Payload[Len] | CRC8(Type,Len,Payload)
CRC8: init 0x00, poly 0x07, MSB-first, no reflect, no final XOR.
All multi-byte fields little-endian.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, Optional

HEADER = b"\xAA\x55"
MAX_PAYLOAD_RX = 250
MAX_PAYLOAD_TX = 32

TYPE_START = 0x01
TYPE_STOP = 0x02
TYPE_CAL_PARAM = 0x03
TYPE_ZERO_CAL = 0x04
TYPE_SET_SAMPLE_RATE = 0x05
TYPE_SET_SD_RATE = 0x06
TYPE_SET_ALARM_CONFIG = 0x07
TYPE_INFO = 0x10
TYPE_DATA = 0x20

SAMPLE_RATES = (1, 10, 100)
SAMPLE_BYTES_V2 = 24  # t_ms u64 + v_nV i64 + i_nA i64

# Paper-wide P5S1 policy (product may list other shunts; serial has no range cmd).
PAPER_I_RANGE_A = 3.3
PAPER_I_RESOLUTION_MA = 0.01  # 10 µA; matches ±3.3 A shunt per manufacturer sheet


def crc8(data: bytes | bytearray) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def build_frame(ftype: int, payload: bytes = b"") -> bytes:
    if len(payload) > MAX_PAYLOAD_TX:
        raise ValueError(f"TX payload too long: {len(payload)} > {MAX_PAYLOAD_TX}")
    body = bytes([ftype & 0xFF, len(payload) & 0xFF]) + payload
    return HEADER + body + bytes([crc8(body)])


def cmd_start() -> bytes:
    return build_frame(TYPE_START)


def cmd_stop() -> bytes:
    return build_frame(TYPE_STOP)


def cmd_set_sample_rate(rate_hz: int) -> bytes:
    if rate_hz not in SAMPLE_RATES:
        raise ValueError(f"sample rate must be one of {SAMPLE_RATES}, got {rate_hz}")
    return build_frame(TYPE_SET_SAMPLE_RATE, bytes([rate_hz]))


def cmd_zero_cal() -> bytes:
    return build_frame(TYPE_ZERO_CAL)


@dataclass(frozen=True)
class Frame:
    type: int
    payload: bytes


@dataclass(frozen=True)
class DeviceInfo:
    model: str
    version: str
    voltage_range_v: float
    current_range_a: float
    digits: float
    sample_rate_hz: int
    alarm_mask: int
    alarm_thresholds: tuple[int, ...]  # 6 x i32


@dataclass(frozen=True)
class Sample:
    t_ms: int
    v_nv: int
    i_na: int

    @property
    def t_s(self) -> float:
        return self.t_ms / 1000.0

    @property
    def voltage_v(self) -> float:
        return self.v_nv / 1e9

    @property
    def current_a(self) -> float:
        return self.i_na / 1e9

    @property
    def power_w(self) -> float:
        return self.voltage_v * self.current_a


def parse_info(payload: bytes) -> DeviceInfo:
    if len(payload) < 9:
        raise ValueError(f"INFO payload too short: {len(payload)}")
    model_len, ver_len = payload[0], payload[1]
    v_range_mv, i_range_ma = struct.unpack_from("<HH", payload, 2)
    digits_x10 = payload[6]
    off = 7
    model = payload[off : off + model_len].decode("utf-8", errors="replace")
    off += model_len
    version = payload[off : off + ver_len].decode("utf-8", errors="replace")
    off += ver_len
    if off + 1 + 1 + 24 > len(payload):
        raise ValueError("INFO payload truncated")
    sample_rate_hz = payload[off]
    alarm_mask = payload[off + 1]
    thresholds = struct.unpack_from("<6i", payload, off + 2)
    return DeviceInfo(
        model=model,
        version=version,
        voltage_range_v=v_range_mv / 1000.0,
        current_range_a=i_range_ma / 1000.0,
        digits=digits_x10 / 10.0,
        sample_rate_hz=sample_rate_hz,
        alarm_mask=alarm_mask,
        alarm_thresholds=thresholds,
    )


def parse_data(payload: bytes) -> list[Sample]:
    if len(payload) < 2:
        raise ValueError("DATA payload too short")
    (n,) = struct.unpack_from("<H", payload, 0)
    need = 2 + n * SAMPLE_BYTES_V2
    if len(payload) < need:
        # Legacy 20-byte sample (u32 t_ms + i64 v + i64 i) — accept if it fits.
        legacy = 2 + n * 20
        if len(payload) >= legacy and n > 0:
            samples: list[Sample] = []
            for k in range(n):
                base = 2 + 20 * k
                t_ms = struct.unpack_from("<I", payload, base)[0]
                v_nv, i_na = struct.unpack_from("<qq", payload, base + 4)
                samples.append(Sample(t_ms=t_ms, v_nv=v_nv, i_na=i_na))
            return samples
        raise ValueError(f"DATA payload truncated: need {need}, got {len(payload)}")
    samples = []
    for k in range(n):
        base = 2 + SAMPLE_BYTES_V2 * k
        t_ms, v_nv, i_na = struct.unpack_from("<Qqq", payload, base)
        samples.append(Sample(t_ms=t_ms, v_nv=v_nv, i_na=i_na))
    return samples


class FrameParser:
    """Byte-stream reassembler keyed on AA 55 header + CRC."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> list[Frame]:
        self._buf.extend(chunk)
        out: list[Frame] = []
        while True:
            frame = self._try_pop()
            if frame is None:
                break
            out.append(frame)
        return out

    def _try_pop(self) -> Optional[Frame]:
        buf = self._buf
        # Resync to header
        while True:
            if len(buf) < 5:
                return None
            idx = buf.find(HEADER)
            if idx < 0:
                # Keep last byte in case it is 0xAA of a split header
                if buf and buf[-1] == HEADER[0]:
                    del buf[:-1]
                else:
                    buf.clear()
                return None
            if idx > 0:
                del buf[:idx]
            if len(buf) < 5:
                return None
            ftype = buf[2]
            plen = buf[3]
            if plen > MAX_PAYLOAD_RX:
                del buf[0]  # skip bad header byte, resync
                continue
            total = 5 + plen
            if len(buf) < total:
                return None
            body = bytes(buf[2 : 4 + plen])
            expect = buf[4 + plen]
            if crc8(body) != expect:
                del buf[0]
                continue
            payload = bytes(buf[4 : 4 + plen])
            del buf[:total]
            return Frame(type=ftype, payload=payload)


def known_cmd_hex() -> dict[str, str]:
    """Golden vectors from the v2 protocol PDF (for self-check)."""
    return {
        "START": cmd_start().hex(" ").upper(),
        "STOP": cmd_stop().hex(" ").upper(),
        "SET_RATE_10": cmd_set_sample_rate(10).hex(" ").upper(),
    }


def assert_golden_vectors() -> None:
    got = known_cmd_hex()
    expect = {
        "START": "AA 55 01 00 15",
        "STOP": "AA 55 02 00 2A",
        "SET_RATE_10": "AA 55 05 01 0A E3",
    }
    for k, v in expect.items():
        if got[k] != v:
            raise AssertionError(f"{k}: got {got[k]}, expect {v}")
