"""Session wrapper around a P5S1 serial / USB-CDC link."""
from __future__ import annotations

import time
from typing import Callable, Iterator, Optional

from bench.p5s1.protocol import (
    TYPE_DATA,
    TYPE_INFO,
    DeviceInfo,
    FrameParser,
    Sample,
    cmd_set_sample_rate,
    cmd_start,
    cmd_stop,
    parse_data,
    parse_info,
)

try:
    import serial
    from serial.tools import list_ports
except ImportError as e:
    raise SystemExit("pip install pyserial") from e


DEFAULT_BAUD = 921600


def list_serial_ports() -> list[str]:
    return [p.device for p in list_ports.comports()]


class Meter:
    def __init__(
        self,
        port: str,
        baud: int = DEFAULT_BAUD,
        timeout: float = 0.05,
    ) -> None:
        self.port = port
        self.baud = baud
        self._ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        self._parser = FrameParser()
        self.info: Optional[DeviceInfo] = None
        time.sleep(0.15)
        self._ser.reset_input_buffer()

    def close(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        try:
            self._ser.close()
        except Exception:
            pass

    def __enter__(self) -> "Meter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _write(self, frame: bytes) -> None:
        self._ser.write(frame)
        self._ser.flush()

    def start(self) -> None:
        """Lock active link (UART/CDC) and begin streaming."""
        self._write(cmd_start())

    def stop(self) -> None:
        self._write(cmd_stop())

    def set_sample_rate(self, rate_hz: int) -> None:
        self._write(cmd_set_sample_rate(rate_hz))

    def drain(self, max_chunks: int = 64) -> list:
        frames = []
        for _ in range(max_chunks):
            chunk = self._ser.read(4096)
            if not chunk:
                break
            frames.extend(self._parser.feed(chunk))
        return frames

    def wait_info(self, timeout_s: float = 3.0) -> DeviceInfo:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for fr in self.drain():
                if fr.type == TYPE_INFO:
                    self.info = parse_info(fr.payload)
                    return self.info
            time.sleep(0.01)
        raise TimeoutError(f"no INFO frame from {self.port} within {timeout_s}s")

    def iter_samples(
        self,
        duration_s: Optional[float] = None,
        on_info: Optional[Callable[[DeviceInfo], None]] = None,
    ) -> Iterator[tuple[float, Sample]]:
        """Yield (host_monotonic_s, sample). Stops on duration or StopIteration."""
        t0 = time.monotonic()
        while True:
            if duration_s is not None and (time.monotonic() - t0) >= duration_s:
                return
            chunk = self._ser.read(4096)
            if not chunk:
                continue
            for fr in self._parser.feed(chunk):
                if fr.type == TYPE_INFO:
                    info = parse_info(fr.payload)
                    self.info = info
                    if on_info is not None:
                        on_info(info)
                elif fr.type == TYPE_DATA:
                    host_t = time.monotonic()
                    for s in parse_data(fr.payload):
                        yield host_t, s
