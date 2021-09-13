import os
from typing import Final

APPNAME = "pyffstream"

try:
    _cpu_count = len(os.sched_getaffinity(0))
except AttributeError:
    if (_cpu := os.cpu_count()) is not None:
        _cpu_count = _cpu
    else:
        _cpu_count = 4

MAX_JOBS: Final[int] = _cpu_count
MAX_IO_JOBS: Final[int] = min(MAX_JOBS, 8)
