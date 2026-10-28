"""Explicit cache-disabled file reader for capstone's upstream integration.

Provides only the disabled MCClient interface used by the pinned dataset reader.
It never implements or pretends to provide a Memcached service.
"""
from pathlib import Path

class MCClient:
    def __init__(self,enabled=False,servers=None):
        if enabled:
            raise ValueError('This capstone adapter requires pin_memory=False; Memcached is unavailable')
    def get(self,path):
        return Path(path).read_bytes()
