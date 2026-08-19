"""Tiny UDP flap sender — the game's input protocol is one ASCII "1" per flap."""

from __future__ import annotations

import socket

FLAP_MESSAGE = b"1"


class UDPSender:
    def __init__(self, host: str = "127.0.0.1", port: int = 5005):
        self.host = host
        self.port = int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_flap(self) -> None:
        self._sock.sendto(FLAP_MESSAGE, (self.host, self.port))

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> "UDPSender":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
