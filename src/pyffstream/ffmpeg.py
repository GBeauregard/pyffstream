"""Helper functions for interacting with ffmpeg.

Functions include mimicking number/duration processing, probing files
with ffprobe, and a utility class to deal with filters.

Attributes:
    ff_bin (FFBin): A tuple that is the recommended way to specify the
        path to ffmpeg utilities. Can be overridden.
"""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import copy
import enum
import functools
import json
import logging
import os
import pathlib
import queue
import re
import shlex
import socket
import subprocess
import threading
import typing
from collections.abc import Iterable, Mapping, MutableSequence, Sequence
from typing import Any, Final, Literal, NamedTuple, TypedDict, Union, overload

logger = logging.getLogger(__name__)


class ProbeType(enum.Enum):
    RAW = enum.auto()
    STREAM = enum.auto()
    TAGS = enum.auto()
    DISPOSITION = enum.auto()
    FORMAT = enum.auto()


# TODO: use typing.TypeAlias in 3.10
StrProbetype = Literal[
    ProbeType.STREAM, ProbeType.TAGS, ProbeType.DISPOSITION, ProbeType.FORMAT
]
StreamQueryTuple = tuple[Iterable[str], Iterable[str], Iterable[str]]
# TODO: 3.10 | union syntax
InitTuple = Union[StreamQueryTuple, Iterable[str]]


class FFProbeJSON(TypedDict, total=False):
    # https://github.com/FFmpeg/FFmpeg/blob/master/doc/ffprobe.xsd
    streams: Sequence[Mapping[str, Any]]
    packets: Sequence[Mapping[str, Any]]
    library_versions: Sequence[Mapping[str, Any]]
    side_data_list: Sequence[Any]
    format: Mapping[str, str]
    program_version: Mapping[str, str]


@functools.total_ordering
class FFVersion:
    """Holds a ffmpeg component version."""

    def __init__(self, *args: str | int | FFVersion):
        """Construct version.

        Args:
            *args: String of type ``(N.)*N`` or an int N or FFVersion to
                be appended to the version when constructing.

        Raises:
            ValueError: Invalid type passed to the constructor.
        """
        self._version: list[int] = []
        for arg in args:
            if isinstance(arg, FFVersion):
                self._version += arg._version
            elif isinstance(arg, str):
                self._version += map(int, arg.split("."))
            elif isinstance(arg, int):
                self._version.append(int(arg))
            else:
                raise ValueError(f"Invalid arg {arg!r} passed to FFVersion")

    def __repr__(self) -> str:
        return ".".join(map(str, self._version))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (FFVersion, str)):
            return NotImplemented
        return self._version == FFVersion(other)._version

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, (FFVersion, str)):
            return NotImplemented
        for me, them in zip(self._version, FFVersion(other)._version):
            if me != them:
                return me < them
        return False


class FFBanner(NamedTuple):
    ffversion: str
    ffconfig: list[str]
    versions: dict[str, FFVersion]


class FFEncoders(NamedTuple):
    vencoders: set[str]
    aencoders: set[str]
    sencoders: set[str]


class FFProtocols(NamedTuple):
    inputs: set[str]
    outputs: set[str]


class FFBin:
    """An instance of a path to an ffmpeg binary.

    Provides various helper properties to get information about the
    capabilities of the ffmpeg binary. Note that just because something
    was compiled into the binary it doesn't mean it's usable at runtime.
    This is the case for hardware support (e.g. nvidia) in particular.

    Attributes:
        ffmpeg (str): Path to ffmpeg binary.
        ffprobe (str): Path to ffprobe binary.
        env (dict[str,str]): Environmental variables to use with ffmpeg.
    """

    def __init__(
        self,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        env: dict[str, str] | None = None,
    ):
        """Inits new FFBin instance.

        Args:
            ffmpeg: Path to ffmpeg binary.
            ffprobe: Path to ffprobe binary.
            env: Environmental variables to use with ffmpeg.
        """
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.env = env if env is not None else os.environ.copy()

    _QUERY_PREFIX: Final = {
        ProbeType.STREAM: "stream=",
        ProbeType.TAGS: "stream_tags=",
        ProbeType.DISPOSITION: "stream_disposition=",
        ProbeType.FORMAT: "format=",
        ProbeType.RAW: "",
    }

    @overload
    def probe(
        self,
        entries: str,
        fileargs: str | Sequence[str] | None,
        streamtype: str | None,
        probetype: StrProbetype,
        deep_probe: bool = ...,
        extraargs: str | Sequence[str] | None = ...,
    ) -> str | None:
        ...

    @overload
    def probe(
        self,
        entries: str,
        fileargs: str | Sequence[str] | None,
        streamtype: str | None,
        probetype: Literal[ProbeType.RAW],
        deep_probe: bool = ...,
        extraargs: str | Sequence[str] | None = ...,
    ) -> FFProbeJSON | None:
        ...

    def probe(
        self,
        entries: str,
        fileargs: str | Sequence[str] | None,
        streamtype: str | None = None,
        probetype: ProbeType = ProbeType.STREAM,
        deep_probe: bool = False,
        extraargs: str | Sequence[str] | None = None,
    ) -> str | FFProbeJSON | None:
        """Probes a media file with ffprobe and returns results.

        Generic function for probing a media file for information using
        ffmpeg's `ffprobe` utility. Capable of returning both individual
        values and the raw deserialized JSON. Users may be interested in
        querying many values at once with JSON ouput in order to avoid
        the delays from repeatedly calling ffprobe.

        Args:
            entries:
                Argument passed to the ``-show_entries`` flag in
                ffprobe. If a non-raw streamtype is specified, then the
                argument may be the type field you want to query, for
                example the duration.
            fileargs:
                String of the file you want to analyze. If additional
                args are needed to specify the input, accepts a list of
                args to pass on.
            streamtype:
                Optional; Argument to pass on to the ``-select_streams``
                flag in ffprobe. Argument not passed if None.
            probetype:
                Optional; If one of STREAM, TAGS, DISPOSITION, FORMAT,
                query file for metadata of selected probetype and
                streamtype and return the requested entries of the first
                matching stream. If RAW, query file as before, but
                return the raw JSON corresponding to the full entries.
            deep_probe:
                Optional; Pass extra arguments to ffprobe in order to
                probe the file more deeply. This is useful for
                containers that can't be lightly inspected.
            extraargs:
                Optional; A list of additional arguments to past to
                ffprobe during runtime. Can be used for example to
                request ``-sexagesimal`` formatting of duration fields.

        Returns:
            None: The query failed or returned "unknown" or "N/A".

            str: For non-raw probetype returns the value of the
            requested query.

            deserialized JSON: For raw probetype, the JSON returned
            after deserialization.

        Raises:
            ValueError: Invalid probetype was passed.
        """
        # fmt: off
        probeargs = [
            self.ffprobe,
            *(["-analyzeduration", "100M", "-probesize", "100M"] if deep_probe else []),
            "-v", "0",
            "-of", "json=c=1",
            "-noprivate",
            *((extraargs,) if isinstance(extraargs, str) else extraargs or ()),
            *(("-select_streams", streamtype) if streamtype is not None else ()),
            "-show_entries", self._QUERY_PREFIX[probetype] + entries,
            *((fileargs,) if isinstance(fileargs, str) else fileargs or ()),
        ]
        # fmt: on
        result = subprocess.run(
            probeargs,
            capture_output=True,
            env=self.env,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode != 0:
            return None
        jsonout: FFProbeJSON = json.loads(result.stdout)
        if probetype is ProbeType.RAW:
            return jsonout
        try:
            # TODO: change to match case in 3.10
            returnval: str
            if probetype is ProbeType.STREAM:
                returnval = jsonout["streams"][0][entries]
            elif probetype is ProbeType.TAGS:
                returnval = jsonout["streams"][0]["tags"][entries]
            elif probetype is ProbeType.DISPOSITION:
                returnval = jsonout["streams"][0]["disposition"][entries]
            elif probetype is ProbeType.FORMAT:
                returnval = jsonout["format"][entries]
            else:
                raise ValueError("invalid probe type query")
        except (KeyError, IndexError):
            # TODO: remove parentheses in 3.10
            return None
        return None if returnval in {"N/A", "unknown"} else returnval

    def make_playlist(
        self,
        pathlist: Iterable[pathlib.Path],
        directory: pathlib.Path,
        add_duration: bool,
        deep_probe: bool = False,
        name: str = "streamplaylist.txt",
    ) -> pathlib.Path:
        """Construct ffconcat playlist from path list.

        Paths created are absolute, and the files are probed in parallel
        if needed to determine the duration.

        Args:
            pathlist: Ordered pathlist to construct ffconcat with.
            directory: Directory to put the constructed playlist in.
            add_duration: Whether or not to include duration metadata.
            deep_probe: Whether to probe file deeply.
            name: Optional; name of playlist file.

        Returns:
            Path to created ffconcat playlist.
        """
        durfutures = None
        if add_duration:
            probe_args = [
                ("duration", str(fpath), None, ProbeType.FORMAT, deep_probe)
                for fpath in pathlist
            ]
            executor = concurrent.futures.ThreadPoolExecutor()
            durfutures = executor.map(lambda p: self.probe(*p), probe_args)
            executor.shutdown(wait=False)
        playlistpath = directory / name
        with playlistpath.open(mode="x", newline="\n", encoding="utf-8") as f:
            print("ffconcat version 1.0", file=f)
            for fpath in pathlist:
                print("file " + shlex.quote(str(fpath.resolve())), file=f)
                if durfutures is not None and (fdur := next(durfutures)) is not None:
                    print("duration " + fdur, file=f)
        return playlistpath

    @functools.cached_property
    def _header(self) -> FFBanner:
        """Gets properties readable from the CLI ffmpeg header.

        Returns:
            FFBanner: a named tuple of those properties.
        """
        ffversion = "0.0.0"
        ffconfig: list[str] = []
        version_dict: dict[str, FFVersion] = collections.defaultdict(
            lambda: FFVersion(0, 0, 0)
        )
        jsonout = self.probe(
            format_probe(
                ("program_version", {"version", "configuration"}),
                ("library_versions", ()),
                allow_empty=True,
            ),
            None,
            None,
            probetype=ProbeType.RAW,
        )
        if jsonout is None:
            return FFBanner(ffversion, ffconfig, version_dict)
        with contextlib.suppress(KeyError):
            ffversion = jsonout["program_version"]["version"]
        with contextlib.suppress(KeyError):
            ffconfig = jsonout["program_version"]["configuration"].split(" ")
        libraries = jsonout.get("library_versions", [])
        for library in libraries:
            with contextlib.suppress(KeyError):
                version_dict[library["name"]] = FFVersion(
                    library["major"], library["minor"], library["micro"]
                )
        return FFBanner(ffversion, ffconfig, version_dict)

    @property
    def ffversion(self) -> str:
        """String representing ffmpeg release version.

        Note this might not always be sensible to use for simple
        comparison, for example in the case of versions compiled from
        git.
        """
        return self._header.ffversion

    @property
    def build_config(self) -> list[str]:
        """List of build config options."""
        return self._header.ffconfig

    @property
    def version(self) -> dict[str, FFVersion]:
        """Dict of FFVersions indexed by component name."""
        return self._header.versions

    @functools.cached_property
    def _encoders(self) -> FFEncoders:
        """All encoders compiled in."""
        encoders: dict[str, set[str]] = {"V": set(), "A": set(), "S": set()}
        encoderargs = [self.ffmpeg, "-hide_banner", "-v", "0", "-encoders"]
        result = subprocess.run(
            encoderargs,
            capture_output=True,
            env=self.env,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode != 0:
            return FFEncoders({"libx264"}, {"aac"}, {"srt"})
        encoder_regex = re.compile(
            r"""
            ^\s*
            (?P<type>[VAS])[A-Z\.]+
            \s*
            (?P<encoder>[\w-]+)
            """,
            flags=re.VERBOSE | re.MULTILINE,
        )
        for match in encoder_regex.finditer(result.stdout):
            if (encoder_type := match.group("type")) and encoder_type in encoders:
                encoders[encoder_type].add(match.group("encoder"))
        return FFEncoders(*encoders.values())

    @property
    def vencoders(self) -> set[str]:
        """Set of compiled-in video encoders."""
        return self._encoders.vencoders

    @property
    def aencoders(self) -> set[str]:
        """Set of compiled-in audio encoders."""
        return self._encoders.aencoders

    @property
    def sencoders(self) -> set[str]:
        """Set of compiled-in subtitle encoders."""
        return self._encoders.sencoders

    @functools.cached_property
    def filters(self) -> set[str]:
        """Set of filters compiled into ffmpeg instance."""
        filters = set()
        versionargs = [self.ffmpeg, "-hide_banner", "-v", "0", "-filters"]
        result = subprocess.run(
            versionargs,
            capture_output=True,
            env=self.env,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode != 0:
            return set()
        filter_regex = re.compile(
            r"""
            ^\s*
            [A-Z.]+
            \s+
            (?P<filter>[\w-]+)
            \s*
            (?P<src>[AVSN|]+)->(?P<dest>[AVSN|]+)
            """,
            flags=re.VERBOSE | re.MULTILINE,
        )
        for match in filter_regex.finditer(result.stdout):
            if filt := match.group("filter"):
                filters.add(filt)
        return filters

    @functools.cached_property
    def hwaccels(self) -> set[str]:
        """Set of hwaccels compiled into ffmpeg version."""
        hwaccels = set()
        hwaccelargs = [self.ffmpeg, "-hide_banner", "-v", "0", "-hwaccels"]
        result = subprocess.run(
            hwaccelargs,
            capture_output=True,
            env=self.env,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode != 0:
            return set()
        hwaccel_regex = re.compile(r"^\s*(?P<accel>[\w-]+)$", flags=re.MULTILINE)
        for match in hwaccel_regex.finditer(result.stdout):
            if accel := match.group("accel"):
                hwaccels.add(accel)
        return hwaccels

    @functools.cached_property
    def protocols(self) -> FFProtocols:
        """A FFProtocols of compiled in input/output protocols."""
        in_protocols: set[str] = set()
        out_protocols: set[str] = set()
        protocol_args = [self.ffmpeg, "-hide_banner", "-v", "0", "-protocols"]
        result = subprocess.run(
            protocol_args,
            capture_output=True,
            env=self.env,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode != 0:
            return FFProtocols(in_protocols, {"rtmp"})
        protocol_regex = re.compile(
            r"""
            ^\s*
            (?:
                (?P<input>[\w-]+)(?=[^:]*:)
                |
                (?P<output>[\w-]+)(?!.*:)
            )$
            """,
            flags=re.VERBOSE | re.MULTILINE,
        )
        for match in protocol_regex.finditer(result.stdout):
            if inp := match.group("input"):
                in_protocols.add(inp)
            if out := match.group("output"):
                out_protocols.add(out)
        return FFProtocols(in_protocols, out_protocols)


ff_bin = FFBin()


def format_probe(*queries: tuple[str, Iterable[str]], allow_empty: bool = False) -> str:
    return ":".join(
        f"{sect[0]}{'=' + ','.join(sect[1]) if sect[1] else ''}"
        for sect in queries
        if sect[1] or allow_empty
    )


def format_q_tuple(init_tuple: InitTuple | None, is_stream: bool) -> str:
    """Format the entries arg for raw JSON queries to the probefile.

    This corresponds to the ``-show_entries`` flag in ffprobe and can be
    used generically to format arguments to it.

    Args:
        init_tuple: If querying a stream, a 3-tuple of the stream
            values, stream tags, and stream dispositions to query
            ffprobe for. If querying a format, a list of the format
            values to query. Can be None.
        is_stream: A boolean indicating whether or not init_tuple is a
            3-tuple for a stream or a list for a format.

    Returns:
        A formatted query for `-show_entries` in ffprobe. An empty
        string if init_tuple is None.
    """
    if init_tuple is None:
        return ""
    if is_stream:
        assert isinstance(init_tuple, tuple)
        return format_probe(
            *zip(("stream", "stream_tags", "stream_disposition"), init_tuple)
        )
    else:
        assert not isinstance(init_tuple, tuple)
        return format_probe(("format", init_tuple))


_SI_PREFIXES: Final[dict[str, float]] = {
    "Y": 8,
    "Z": 7,
    "E": 6,
    "P": 5,
    "T": 4,
    "G": 3,
    "M": 2,
    "K": 1,
    "k": 1,
    "h": 2 / 3,
    "": 0,
    "d": -1 / 3,
    "c": -2 / 3,
    "m": -1,
    "u": -2,
    "n": -3,
    "p": -4,
    "f": -5,
    "a": -6,
    "z": -7,
    "y": -8,
}


def num(val: str | int | float) -> float:
    """Process input into float in a way that mimics ffmpeg.

    Method follows ffmpeg's `numerical options`_. All whitespace is
    stripped, then valid input is a number followed optionally with an
    SI prefix that may be appended with an `i` modifier that indicates
    the SI prefix is in powers of 1024 instead of 1000. Finally, the
    number may end in a `B` indicating it is to be multiplied by 8. The
    optional ffmpeg utility ``ffeval`` may be used to validate the
    output of this function.

    .. _numerical options:
       https://ffmpeg.org/ffmpeg.html#Options
    """
    val = str(val)
    val = re.sub(r"\s+", "", val)
    input_regex = re.compile(
        r"""
        (?P<number>-?(?:\d+\.?\d*|\.\d+))
        (?:
            (?P<siprefix>[YZEPTGMKkhdcmunpfazy])
            (?P<binary>i)?
        )?
        (?P<byte>B)?
        """,
        flags=re.VERBOSE | re.ASCII,
    )
    if (match := input_regex.fullmatch(val)) is None:
        raise ValueError(f"Invalid ffmpeg number string: {val!r}")
    if not (basenum := match.group("number")):
        raise ValueError(f"Match number not found: {val!r}")
    prefix = _SI_PREFIXES[si_prefix] if (si_prefix := match.group("siprefix")) else 0
    power = 1024 if match.group("binary") == "i" else 1000
    byte = 8 if match.group("byte") == "B" else 1
    return float(basenum) * power ** prefix * byte


def duration(timestamp: str | float | int) -> float:
    """Processes ffmpeg duration string into time in seconds.

    https://ffmpeg.org/ffmpeg-utils.html#Time-duration
    """
    timestamp = str(timestamp)
    if (
        re.fullmatch(r"(\d+:)?\d?\d:\d\d(\.\d*)?", timestamp, flags=re.ASCII)
        is not None
    ):
        return sum(
            s * float(t) for s, t in zip([1, 60, 3600], reversed(timestamp.split(":")))
        )
    if (
        match := re.fullmatch(
            r"(-?(?:\d+\.?\d*|\.\d+))([mu]?s)?", timestamp, flags=re.ASCII
        )
    ) is not None:
        val = float(match.group(1))
        if match.group(2) == "ms":
            val /= 1000
        elif match.group(2) == "us":
            val /= 1_000_000
        return val
    raise ValueError(f"Invalid ffmpeg duration: {timestamp!r}")


class Filter:
    """A single ffmpeg filter.

    Collects helper methods for constructing and rendering out ffmpeg
    filter strings for use with the CLI.
    """

    def __init__(
        self,
        filt: str | Filter,
        *filtopts: str,
        src: Sequence[str | int | None] | None = None,
        dst: Sequence[str | int | None] | None = None,
    ) -> None:
        if src is None:
            src = [None]
        if dst is None:
            dst = [None]
        self.basefilter: str
        self.opts: MutableSequence[str]
        self.src: list[str | int | None]
        self.dst: list[str | int | None]
        if isinstance(filt, Filter):
            self.basefilter = filt.basefilter
            self.opts = filt.opts
            self.src = list(src) if src != [None] else filt.src
            self.dst = list(dst) if dst != [None] else filt.dst
        else:
            self.src = list(src)
            self.dst = list(dst)
            self.basefilter = filt
            self.opts = []
            if filtopts:
                if all(
                    isinstance(list_, Sequence) and not isinstance(list_, str)
                    for list_ in filtopts
                ):
                    for list_ in filtopts:
                        self.opts += list_
                else:
                    self.opts = list(filtopts)

    @staticmethod
    def escape_val(val: str) -> str:
        trans = str.maketrans({char: "\\" + char for char in ":\\'"})
        return val.translate(trans)

    @staticmethod
    def escape_graph(val: str) -> str:
        trans = str.maketrans({char: "\\" + char for char in "\\'[],;"})
        return val.translate(trans)

    @classmethod
    def full_escape(cls, val: str) -> str:
        """Do full escaping needed for complex filter graph in ffmpeg.

        https://ffmpeg.org/ffmpeg-filters.html#Notes-on-filtergraph-escaping
        """
        return cls.escape_graph(cls.escape_val(val))

    @classmethod
    def complex_join(
        cls,
        filterlist: Sequence[Filter | str],
        startkey: str = "v",
        endkey: str = "b",
        basekey: str = "c",
    ) -> str:
        filts = copy.deepcopy(list(map(cls, filterlist)))
        filts[0].src = [key or startkey for key in filts[0].src]
        filts[-1].dst = [key or endkey for key in filts[-1].dst]
        output = ""
        produced: list[int | str] = []
        for i, f in enumerate(filts):
            if f.src == [None]:
                output += ","
            else:
                if i != 0:
                    output += ";"
                for key in f.src:
                    if isinstance(key, int) or key is None:
                        output += f"[{produced[key or 0]}]"
                    else:
                        output += f"[{key}]"
                produced = [key for idx, key in enumerate(produced) if idx not in f.src]
            output += str(f)
            if f.dst != [None] or filts[i + 1].src != [None]:
                for key in f.dst:
                    if isinstance(key, int) or key is None:
                        key = basekey
                        produced.append(basekey)
                        basekey = chr(ord(basekey) + 1)
                    output += f"[{key}]"
        return output

    @staticmethod
    def vf_join(filterlist: Sequence[str | Filter]) -> str:
        return ",".join(map(str, filterlist))

    # @classmethod
    # def parse_filtstring(cls, filtstr: str) -> Filter:
    #     return cls(filtstr)

    def __repr__(self) -> str:
        if self.opts:
            return self.basefilter + "=" + ":".join(self.opts)
        else:
            return self.basefilter


class Progress:
    """Assists in monitoring the progress output of an ffmpeg encode."""

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("localhost", 0))
        self._sock.listen(1)
        self._packet_avail = threading.Event()
        self._packet_lock = threading.Lock()
        self._progress_packet: list[str] = []
        self._loglevel: int
        self._futures: list[concurrent.futures.Future[Any]] = []
        self.progress_avail = threading.Event()
        self.finished = threading.Event()
        self.output: collections.deque[str]
        self.output_que: queue.Queue[str | None]
        self._make_queue: bool
        self.status: dict[str, str] = {
            "frame": "0",
            "fps": "0",
            "stream_0_0_q": "0.0",
            "bitrate": "0kbit/s",
            "total_size": "0",
            "out_time_us": "0",
            "out_time_ms": "0",
            "out_time": "00:00:00",
            "dup_frames": "0",
            "drop_frames": "0",
            "speed": "0x",
            "progress": "continue",
        }

    def flags(self, update_period: float = 0.5) -> list[str]:
        return [
            "-nostats",
            "-nostdin",
            "-progress",
            "tcp://" + ":".join(map(str, self._sock.getsockname())),
            "-stats_period",
            f"{update_period:.6f}",
        ]

    @property
    def time_us(self) -> int:
        try:
            return int(self.status["out_time_us"])
        except ValueError:
            return 0

    @property
    def time_s(self) -> float:
        return self.time_us / 1_000_000

    def monitor_progress(
        self,
        result: subprocess.Popen[str],
        outstream: typing.IO[str],
        make_queue: bool = False,
        maxlen: int | None = None,
        loglevel: int = logging.DEBUG,
    ) -> None:
        self._loglevel = loglevel
        self._make_queue = make_queue
        if self._make_queue:
            self.output_que = queue.Queue()
        self.output = collections.deque(maxlen=maxlen)
        executor = concurrent.futures.ThreadPoolExecutor()
        self._futures.append(executor.submit(self._read_outstream, result, outstream))
        self._futures.append(executor.submit(self._read_progress, result))
        self._futures.append(executor.submit(self._parse_progress))
        executor.shutdown(wait=False)

    def _read_outstream(
        self, result: subprocess.Popen[str], outstream: typing.IO[str]
    ) -> None:
        try:
            while result.poll() is None:
                self._parse_outputline(outstream.readline().rstrip())
            for line in outstream:
                self._parse_outputline(line.rstrip())
        except ValueError:
            pass
        finally:
            if self._make_queue:
                self.output_que.put(None)
            self.finished.set()
            self._packet_avail.set()
            self.progress_avail.set()

    def _parse_outputline(self, line: str) -> None:
        if not line:
            return
        logger.log(self._loglevel, line)
        self.output.append(line)
        if self._make_queue:
            self.output_que.put(line)

    def _read_progress(self, result: subprocess.Popen[str]) -> None:
        conn, _ = self._sock.accept()
        with contextlib.closing(conn), contextlib.closing(self._sock), conn.makefile(
            encoding="utf-8"
        ) as f:
            while result.poll() is None:
                packet = []
                while line := f.readline():
                    if line.startswith("progress="):
                        break
                    if line := line.rstrip():
                        packet.append(line)
                if packet:
                    with self._packet_lock:
                        self._progress_packet = packet
                    self._packet_avail.set()

    def _parse_progress(self) -> None:
        self._packet_avail.wait()
        while not self.finished.is_set():
            with self._packet_lock:
                packet = self._progress_packet.copy()
            for line in packet:
                split = line.split("=", 1)
                try:
                    self.status[split[0]] = split[1]
                except IndexError:
                    logger.error(f"unexpected line in progress: {line!r}")
            self.progress_avail.set()
            self._packet_avail.wait()
            self._packet_avail.clear()

    def __del__(self) -> None:
        self._sock.close()
        concurrent.futures.wait(self._futures)
        for future in self._futures:
            if exception := future.exception(5):
                raise exception
