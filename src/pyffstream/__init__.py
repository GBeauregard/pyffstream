"""pyffstream.

Script and tools to enable streaming files or OBS with ffmpeg to a
remote server. Specializes in sending over SRT.
"""
from typing import Final

# import os

APPNAME: Final = "pyffstream"

# try:
#     _cpu_count = len(os.sched_getaffinity(0))
# except AttributeError:
#     if (_cpu := os.cpu_count()) is not None:
#         _cpu_count = _cpu
#     else:
#         _cpu_count = 4

# MAX_JOBS: Final[int] = _cpu_count
