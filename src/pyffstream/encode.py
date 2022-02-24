"""Classes and functions for ffmpeg encoding."""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import enum
import fractions
import json
import logging
import math
import pathlib
import re
import subprocess
import threading
from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from typing import Any, ClassVar, Final, NamedTuple, TypeVar, cast

from . import ffmpeg

logger = logging.getLogger(__name__)


class FileOpts(NamedTuple):
    main: MutableSequence[str]
    subtitle: MutableSequence[str]
    fpath: pathlib.Path
    sfpath: pathlib.Path
    allpaths: Sequence[pathlib.Path]


class EncodeSession:
    VSTREAMS: Final = {
        "index",
        "width",
        "height",
        "start_time",
        "r_frame_rate",
        "time_base",
        "pix_fmt",
        "codec_name",
        "bit_rate",
    }
    VSTREAM_TAGS: Final = {
        "BPS",
        "BPS-eng",
    }
    ASTREAMS: Final = {
        "index",
        "start_time",
        "codec_name",
        "sample_rate",
        "bit_rate",
        "channel_layout",
    }
    ASTREAM_TAGS: Final = {
        "BPS",
        "BPS-eng",
    }
    SSTREAMS: Final = {
        "index",
        "codec_type",
        "codec_name",
    }
    SSTREAM_TAGS: Final = set[str]()
    VDISPOSITIONS: Final = set[str]()
    ADISPOSITIONS: Final = set[str]()
    SDISPOSITIONS: Final = set[str]()
    FORMAT_IDS: Final = {
        "bit_rate",
        "duration",
    }
    VIDEO_DEF: Final = {
        "start_time": "0",
        "time_base": "1/1000",
        "pix_fmt": "yuv420p",
        "codec_name": "h264",
        "r_frame_rate": "24/1",
    }
    AUDIO_DEF: Final = {
        "start_time": "0",
        "sample_rate": "48000",
        "codec_name": "aac",
    }
    SUBTITLE_DEF: Final[dict[str, str]] = {}
    FORMAT_DEF: Final = {
        "duration": "1200",
    }

    def __init__(self, fopts: FileOpts, ev: StaticEncodeVars) -> None:
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.fopts = fopts
        self.ev = ev
        vstreamvals = self.executor.submit(
            FileStreamVals,
            self.ev,
            f"v:{self.ev.vindex}",
            self.fopts.main,
            (self.VSTREAMS, self.VSTREAM_TAGS, self.VDISPOSITIONS),
            self.VIDEO_DEF,
        )
        astreamvals = self.executor.submit(
            FileStreamVals,
            self.ev,
            f"a:{self.ev.aindex}",
            self.fopts.main,
            (self.ASTREAMS, self.ASTREAM_TAGS, self.ADISPOSITIONS),
            self.AUDIO_DEF,
        )
        sstreamvals = self.executor.submit(
            FileStreamVals,
            self.ev,
            f"s:{self.ev.sindex}",
            self.fopts.subtitle,
            (self.SSTREAMS, self.SSTREAM_TAGS, self.SDISPOSITIONS),
            self.SUBTITLE_DEF,
        )
        fstreamvals = self.executor.submit(
            FileStreamVals,
            self.ev,
            "f",
            self.fopts.main,
            self.FORMAT_IDS,
            self.FORMAT_DEF,
        )
        self.streamvals = {
            "v": vstreamvals.result,
            "a": astreamvals.result,
            "s": sstreamvals.result,
            "f": fstreamvals.result,
        }
        self.filts = FilterList()
        self.statuses: list[StatusThread] = []
        self.update_avail = threading.Event()
        if self.ev.crop:
            self.crop = StatusThread(self, "crop")
            self.statuses.append(self.crop)
        if self.ev.anormalize:
            self.norm = StatusThread(self, "normalize")
            self.statuses.append(self.norm)
        if self.ev.subs:
            self.subs = StatusThread(self, "subtitles")
            self.statuses.append(self.subs)

    def __del__(self) -> None:
        self.executor.shutdown()

    def v(self, stype: str, key: str, ptype: ffmpeg.ProbeType | None = None) -> str:
        """Get file val (with default fallback)."""
        return self.streamvals[stype]().getval(key, ptype)

    def fv(
        self, stype: str, key: str, ptype: ffmpeg.ProbeType | None = None
    ) -> str | None:
        """Get file val (without default fallback)."""
        return self.streamvals[stype]().getfileval(key, ptype)

    def dv(
        self, stype: str, key: str, ptype: ffmpeg.ProbeType | None = None
    ) -> str | None:
        """Get default val."""
        return self.streamvals[stype]().getdefault(key, ptype)

    def sdv(
        self, stype: str, key: str, val: str, ptype: ffmpeg.ProbeType | None = None
    ) -> str | None:
        """Set default val."""
        return self.streamvals[stype]().setdefault(key, val, ptype)

    def sdvs(self, stype: str, vals: Mapping[str, str | Mapping[str, str]]) -> None:
        """Set default vals by specifying dict to merge in."""
        self.streamvals[stype]().setdefaults(vals)


class FileStreamVals:
    def __init__(
        self,
        ev: StaticEncodeVars,
        selector: str,
        fileargs: Sequence[str],
        init_tuple: ffmpeg.InitTuple | None = None,
        defaults: Mapping[str, str | Mapping[str, str]] | None = None,
    ) -> None:
        if defaults is None:
            defaults = {}
        self.__lock = threading.RLock()
        self.selector, self.fileargs = selector, list(fileargs).copy()
        self.fileargs.pop(-2)
        self.dont_probe = ev.live or (not ev.subs and self.selector[0] == "s")
        self.deep_probe = ev.deep_probe
        emptydict: dict[str, Any] = {}
        self.default_probetype: ffmpeg.ProbeType
        if self.selector[0] in {"v", "a", "s"}:
            is_stream = True
            self.default_probetype = ffmpeg.ProbeType.STREAM
            emptydict = {"disposition": {}, "tags": {}}
        elif self.selector[0] == "f":
            is_stream = False
            self.default_probetype = ffmpeg.ProbeType.FORMAT
            emptydict = {}
        else:
            raise ValueError(f"invalid selector: {self.selector!r}")
        self.filevals = emptydict.copy()
        self.defaultvals = emptydict.copy()
        self.defaultvals |= defaults
        if probestr := ffmpeg.format_q_tuple(init_tuple, is_stream):
            assert init_tuple is not None  # implied by check passed above
            if not self.dont_probe:
                outjson = ffmpeg.ff_bin.probe_json(
                    probestr,
                    self.fileargs,
                    None if self.selector[0] == "f" else self.selector,
                    deep_probe=self.deep_probe,
                )
                if outjson is not None:
                    if is_stream:
                        if streams := outjson.get("streams"):
                            self.filevals = dict(streams[0])
                    elif formats := outjson.get("format"):
                        self.filevals = dict(formats)

            def initval(
                valdict: MutableMapping[str, str | None], keylist: Iterable[str]
            ) -> None:
                for val in keylist:
                    if val not in valdict or valdict[val] in {"N/A", "unknown"}:
                        valdict[val] = None

            if is_stream:
                assert isinstance(init_tuple, tuple)
                initval(self.filevals, init_tuple[0])
                if "tags" not in self.filevals:
                    self.filevals["tags"] = cast(dict[str, Any], {})
                if "disposition" not in self.filevals:
                    self.filevals["disposition"] = cast(dict[str, Any], {})
                initval(self.filevals["tags"], init_tuple[1])
                initval(self.filevals["disposition"], init_tuple[2])
            else:
                assert not isinstance(init_tuple, tuple)
                initval(self.filevals, init_tuple)

    def get_t_valdict(
        self, t: ffmpeg.ProbeType | None = None
    ) -> tuple[ffmpeg.ProbeType, dict[str, str | None], dict[str, str | None]]:
        with self.__lock:
            probetype = self.default_probetype if t is None else t
            # TODO: 3.10 match case
            if probetype in {ffmpeg.ProbeType.STREAM, ffmpeg.ProbeType.FORMAT}:
                file_dict = self.filevals
                default_dict = self.defaultvals
            elif probetype is ffmpeg.ProbeType.TAGS:
                file_dict = self.filevals["tags"]
                default_dict = self.defaultvals["tags"]
            elif probetype is ffmpeg.ProbeType.DISPOSITION:
                file_dict = self.filevals["disposition"]
                default_dict = self.defaultvals["disposition"]
            else:
                raise ValueError(f"Invalid streamtype {t!r} passed to get_t_valdict")
            return probetype, file_dict, default_dict

    def getval(self, key: str, t: ffmpeg.ProbeType | None = None) -> str:
        with self.__lock:
            if (fileval := self.getfileval(key, t)) is not None or (
                fileval := self.getdefault(key, t)
            ) is not None:
                return fileval
            else:
                raise ValueError(
                    f"Key {key!r} of type {t!r} failed to return value with getval"
                    " function."
                )

    def getfileval(self, key: str, t: ffmpeg.ProbeType | None = None) -> str | None:
        with self.__lock:
            probetype, valdict, _ = self.get_t_valdict(t)
            try:
                return valdict[key]
            except KeyError:
                logger.warning(
                    "File probed for uncached val %r of type %r for %r",
                    key,
                    probetype,
                    self.selector,
                )
                if self.dont_probe:
                    readval = None
                else:
                    readval = ffmpeg.ff_bin.probe_val(
                        key,
                        self.fileargs,
                        self.selector,
                        probetype=probetype,
                        deep_probe=self.deep_probe,
                    )
                valdict[key] = readval
                return readval

    def getdefault(self, key: str, t: ffmpeg.ProbeType | None = None) -> str | None:
        with self.__lock:
            return self.get_t_valdict(t)[2].get(key)

    def setdefault(self, key: str, val: str, t: ffmpeg.ProbeType | None = None) -> str:
        with self.__lock:
            self.get_t_valdict(t)[2][key] = val
            return val

    def setdefaults(self, vals: Mapping[str, str | Mapping[str, str]]) -> None:
        with self.__lock:
            self.defaultvals |= vals


class StatusCode(enum.Enum):
    FAILED = enum.auto()
    FINISHED = enum.auto()
    NOT_STARTED = enum.auto()
    RUNNING = enum.auto()
    OTHER = enum.auto()


class StatusThread:
    def __init__(self, fv: EncodeSession, name: str) -> None:
        self.__lock = threading.RLock()
        self.fv = fv
        self.name = name
        self.status: StatusCode = StatusCode.NOT_STARTED
        self.long_status = "starting"
        self.progress = 0.0

    def setstatus(self, status: StatusCode, long_status: str) -> None:
        with self.__lock:
            self.status = status
            self.long_status = long_status
            self.fv.update_avail.set()

    def setprogress(self, progress: float) -> None:
        with self.__lock:
            self.progress = progress
            self.fv.update_avail.set()


class FilterList:
    """Thread-safe class for storing and getting ffmpeg filters."""

    def __init__(self) -> None:
        self.__lock = threading.Lock()
        self.filts: dict[str, ffmpeg.Filter] = {}

    def __getitem__(self, key: str) -> str:
        """Return requested filter as a string."""
        with self.__lock:
            return str(self.filts[key])

    def get(self, k: str, default: ffmpeg.Filter | str | None = None) -> str | None:
        """An implementation of the get method as in dicts."""
        with self.__lock:
            out = self.filts.get(k, default)
            return None if out is None else str(out)

    def __setitem__(self, key: str, val: str | ffmpeg.Filter | Sequence[str]) -> None:
        """Set filter.

        val may be the filter string, a filter itself, or a sequence to
        pass to the filter constructor.
        """
        with self.__lock:
            if isinstance(val, Sequence) and not isinstance(val, (str, ffmpeg.Filter)):
                self.filts[key] = ffmpeg.Filter(*val)
            else:
                self.filts[key] = ffmpeg.Filter(val)

    def if_exists(self, key: str) -> tuple[str] | tuple[()]:
        """Return 1-tuple of filter at key if it exists, else 0-tuple.

        The intent is to use this function starred in-line like
        ``*if_exists(key)`` when defining a list to simplify creation of
        filter lists.
        """
        with self.__lock:
            if key in self.filts:
                return (str(self.filts[key]),)
            else:
                return ()


def min_version(
    args: str | Sequence[str], version: tuple[str, str | ffmpeg.FFVersion]
) -> Sequence[str] | tuple[()]:
    if ffmpeg.ff_bin.version[version[0]] < version[1]:
        return ()
    if isinstance(args, str):
        return (args,)
    else:
        return args


@dataclasses.dataclass
class StaticEncodeVars:
    """Class holding general encoding parameters.

    Needs to be passed to an encode session to initialize it.
    """

    STREAM_PROTOCOLS: ClassVar[set[str]] = {"srt", "rtmp"}

    H264_ENCODERS: ClassVar[set[str]] = {"h264_nvenc", "libx264"}
    HEVC_ENCODERS: ClassVar[set[str]] = {"hevc_nvenc", "libx265"}
    NVIDIA_ENCODERS: ClassVar[set[str]] = {"hevc_nvenc", "h264_nvenc"}
    SW_ENCODERS: ClassVar[set[str]] = {"libx264", "libx265"}
    TENBIT_ENCODERS: ClassVar[set[str]] = HEVC_ENCODERS
    MULTIPASS_ENCODERS: ClassVar[set[str]] = {"libx264", "libx265"}
    VIDEO_ENCODERS: ClassVar[set[str]] = H264_ENCODERS | HEVC_ENCODERS
    ALLOWED_PRESETS: ClassVar[list[str]] = [
        "placebo",
        "veryslow",
        "slower",
        "slow",
        "medium",
        "fast",
        "faster",
        "veryfast",
        "superfast",
        "ultrafast",
    ]

    AAC_ENCODERS: ClassVar[set[str]] = {"aac", "libfdk_aac"}
    OPUS_ENCODERS: ClassVar[set[str]] = {"libopus"}
    AUDIO_ENCODERS: ClassVar[set[str]] = AAC_ENCODERS | OPUS_ENCODERS

    tempdir: pathlib.Path

    # TODO: 3.10 _: KW_ONLY

    verbosity: int = 0
    api_url: str = ""
    api_key: str = ""
    # samplerate for passing on to server
    samplerate: str = "48000"
    endpoint: str = "127.0.0.1:9998"
    target_w: str = "1920"
    target_h: str = "1080"
    framerate: str = "24/1"
    framerate_multiplier: fractions.Fraction = fractions.Fraction(1)
    scale_w: str = target_w
    scale_h: str = target_h
    bound_w: str = "1920"
    bound_h: str = "1080"
    kf_target_sec: float = 5.0
    clip_length: str | None = None
    vencoder: str = "libx264"
    aencoder: str = "aac"
    x26X_preset: str = "medium"
    x26X_tune: str | None = None
    vstandard: str = "h264"
    astandard: str = "aac"
    protocol: str = "srt"
    # these strings are processed into int by argparse
    vbitrate: str = "6M"
    max_vbitrate: str = "6M"
    abitrate: str = "256k"
    chlayout: str = "stereo"
    start_delay: str = "30"
    end_pad: bool = True
    end_delay: str = "600"
    timestamp: str | None = None
    suboffset: str | None = None
    cleanborders: list[int] | None = None
    # timestamp to attempt crop at (can be ffmpeg timestamp)
    crop_ts: str = "600"
    # crop length (ditto)
    crop_len: str = "60"

    # loudnorm parameters
    target_i: str = "-19"
    target_lra: str = "11.0"
    target_tp: str = "-1.0"

    pix_fmt: str = "yuv420p"
    subfile_provided: bool = False
    text_subs: bool = True
    subfilter_list: Sequence[ffmpeg.Filter] = dataclasses.field(default_factory=list)
    kf_int: str = "0"
    min_kf_int: str = "0"
    bufsize: str = "0"
    kf_sec: str = "0"
    latency_target: str = "0"
    afilters: str = ""
    filter_complex: str = ""
    input_flags: MutableSequence[str] = dataclasses.field(default_factory=list)
    encode_flags: MutableSequence[str] = dataclasses.field(default_factory=list)
    filter_flags: MutableSequence[str] = dataclasses.field(default_factory=list)
    output_flags: MutableSequence[str] = dataclasses.field(default_factory=list)
    shader_list: MutableSequence[str] = dataclasses.field(default_factory=list)
    ff_flags: MutableSequence[str] = dataclasses.field(default_factory=list)
    srt_passphrase: str = ""
    srt_latency: float = 5.0
    ff_verbosity_flags: Sequence[str] = dataclasses.field(default_factory=list)
    ff_deepprobe_flags: Sequence[str] = dataclasses.field(default_factory=list)
    placebo_opts: Sequence[str] = dataclasses.field(default_factory=list)
    sw_filters: Sequence[str] = dataclasses.field(default_factory=list)
    vencoder_params: Sequence[str] = dataclasses.field(default_factory=list)
    copy_audio: bool = False
    copy_video: bool = False
    use_timeline: bool = False
    hwaccel: bool = True
    subs: bool = False
    deep_probe: bool = False
    vindex: int = 0
    aindex: int = 0
    sindex: int | None = None
    outfile: pathlib.Path | None = None
    npass: int | None = None
    passfile: pathlib.Path | None = None
    wait: bool = False
    fifo: bool = False
    soxr: bool = False
    zscale: bool = False
    slowseek: bool = False
    live: bool = False
    obs: bool = False
    vulkan: bool = False
    vgop: bool = False
    trust_vulkan: bool = False
    vulkan_device: int = -1
    decimate_target: str = "24/1"
    is_playlist: bool = False
    eightbit: bool = False
    cropsecond: bool = False
    subcropfirst: bool = False
    picsubscale: str = "bicubic"
    delay_start: bool = False
    deinterlace: bool = False
    crop: bool = False
    upscale: bool = False
    anormalize: bool = False
    normfile: pathlib.Path | None = None
    dynamicnorm: bool = False
    fix_start_time: bool = True
    pyffserver: bool = False
    ffprogress: ffmpeg.Progress[str] = dataclasses.field(
        default_factory=ffmpeg.Progress
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> StaticEncodeVars:
        """Constructor to make StaticEncodeVars from passed args."""
        evars = cls(tempdir=args.tempdir)
        evars.pyffserver = args.pyffserver
        evars.protocol = args.protocol
        evars.srt_passphrase = args.srt_passphrase
        evars.srt_latency = args.srt_latency
        evars.vencoder = args.vencoder
        if evars.vencoder in cls.H264_ENCODERS:
            evars.vstandard = "h264"
        elif evars.vencoder in cls.HEVC_ENCODERS:
            evars.vstandard = "hevc"
        evars.aencoder = args.aencoder
        if evars.aencoder in cls.AAC_ENCODERS:
            evars.astandard = "aac"
        elif evars.aencoder in cls.OPUS_ENCODERS:
            evars.astandard = "opus"
        evars.x26X_preset = args.preset
        evars.x26X_tune = args.tune
        evars.fix_start_time = args.fix_start_time
        evars.dynamicnorm = args.dynamicnorm
        evars.normfile = args.normfile
        evars.anormalize = args.anormalize
        evars.soxr = args.soxr
        evars.zscale = args.zscale
        evars.vgop = args.vgop
        evars.vulkan = args.vulkan
        evars.trust_vulkan = args.trust_vulkan
        evars.vulkan_device = args.vulkan_device
        evars.placebo_opts = args.placebo_opts or []
        evars.sw_filters = args.sw_filters or []
        evars.vencoder_params = args.vencoder_params or []
        evars.upscale = args.upscale
        evars.crop = args.crop
        evars.crop_ts = args.croptime
        evars.crop_len = args.croplength
        evars.deinterlace = args.deinterlace
        evars.cropsecond = args.cropsecond
        evars.subcropfirst = args.subfirst
        evars.picsubscale = args.picsubscale
        evars.delay_start = args.startdelay
        evars.end_pad = args.endpad
        evars.is_playlist = args.playlist
        evars.hwaccel = args.hwaccel
        if args.obs:
            if args.sixtyfps:
                evars.decimate_target = "60/1"
            elif args.paldecimate:
                evars.decimate_target = "25/1"
            elif args.nodecimate:
                evars.decimate_target = "30/1"
            else:
                evars.decimate_target = "24/1"
        evars.obs = args.obs
        evars.wait = args.wait
        evars.fifo = args.fifo
        evars.slowseek = args.slowseek
        evars.live = args.live
        evars.outfile = args.outfile
        evars.npass = args.npass
        evars.passfile = args.passfile
        if args.npass in (1, 3):
            evars.outfile = pathlib.Path("-")
        evars.vindex = args.vindex
        evars.aindex = args.aindex
        evars.sindex = args.sindex
        evars.subs = args.subs
        evars.deep_probe = args.deep_probe
        evars.copy_audio = args.copy_audio
        evars.copy_video = args.copy_video
        evars.endpoint = args.endpoint
        evars.api_url = args.api_url
        evars.api_key = args.api_key
        evars.target_w = str(math.ceil(args.height * 16 / 9))
        evars.target_h = str(args.height)
        evars.framerate_multiplier = args.framerate_multiplier
        evars.kf_target_sec = args.keyframe_target_sec
        evars.clip_length = args.cliplength
        evars.verbosity = args.verbose
        evars.vbitrate = str(args.vbitrate)
        evars.max_vbitrate = (
            str(args.max_vbitrate) if args.max_vbitrate else evars.vbitrate
        )
        evars.abitrate = str(int(args.abitrate))
        evars.shader_list = args.shaders
        evars.chlayout = "stereo" if not args.mono else "mono"
        evars.timestamp = args.timestamp
        evars.suboffset = args.suboffset
        evars.cleanborders = args.cleanborders
        evars.eightbit = args.eightbit
        evars.pix_fmt = (
            ("yuv420p10le" if args.vencoder in cls.SW_ENCODERS else "p010le")
            if args.vencoder in cls.TENBIT_ENCODERS and not args.eightbit
            else "yuv420p"
        )
        evars.eightbit = evars.pix_fmt == "yuv420p"
        evars.subfile_provided = args.subfile is not None
        evars.ff_deepprobe_flags = (
            ["-analyzeduration", "100M", "-probesize", "100M"]
            if args.deep_probe
            else []
        )
        if evars.verbosity >= 4:
            evars.ff_verbosity_flags = ["-loglevel", "trace", "-debug_ts"]
        if evars.verbosity >= 3:
            evars.ff_verbosity_flags = ["-loglevel", "debug", "-debug_ts"]
        elif evars.verbosity >= 2:
            evars.ff_verbosity_flags = ["-loglevel", "debug"]
        elif evars.verbosity >= 1:
            evars.ff_verbosity_flags = ["-loglevel", "verbose"]
        else:
            evars.ff_verbosity_flags = ["-hide_banner"]
        return evars


T = TypeVar("T")


def percentile(data: Sequence[T], perc: float) -> T:
    return data[math.ceil((len(data) * perc) / 100) - 1]


def divide_off(num: int, divisor: int) -> int:
    while not math.remainder(num, divisor):
        num //= divisor
    return num


def do_framerate_calcs(fv: EncodeSession) -> None:
    if fv.ev.obs:
        fv.sdv("v", "r_frame_rate", fv.ev.decimate_target)
    framerate = fractions.Fraction(fv.v("v", "r_frame_rate"))
    framerate *= fv.ev.framerate_multiplier
    fv.ev.framerate = str(framerate)
    if fv.ev.copy_video:
        if fv.ev.outfile is not None:
            return
        fv.ev.use_timeline = True
        frame_json = ffmpeg.ff_bin.probe_json(
            "packet=pts,flags",
            fv.streamvals["v"]().fileargs,
            fv.streamvals["v"]().selector,
            deep_probe=fv.ev.deep_probe,
        )
        if frame_json:
            pts_list = [
                int(pts)
                for packet in frame_json.get("packets", [])
                if str(pts := packet.get("pts", "")).isdecimal()
                and packet.get("flags", "__")[0] == "K"
            ]
            if len(pts_list) >= 2:
                time_diffs = sorted(y - x for x, y in zip(pts_list, pts_list[1:]))
                min_diff = percentile(time_diffs, 30)
                max_diff = percentile(time_diffs, 99.5)
                timebase = fractions.Fraction(fv.v("v", "time_base"))
                fv.ev.kf_int = str(int(min_diff * timebase * framerate))
                fv.ev.min_kf_int = fv.ev.kf_int
                fv.ev.kf_sec = (
                    f"{float(min_diff*timebase):.7f}"[:-1].rstrip("0").rstrip(".")
                )
                fv.ev.latency_target = (
                    f"{max(4*float(fv.ev.kf_sec), float(2*max_diff*timebase)):.8g}"
                )
                logger.debug("keyframe interval: %s", fv.ev.kf_int)
                logger.debug("keyframe seconds: %s", fv.ev.kf_sec)
                logger.debug("latency target: %s", fv.ev.latency_target)
                return
    # see https://trac.ffmpeg.org/ticket/9440
    ideal_gop = fv.ev.kf_target_sec * framerate
    num = framerate.numerator
    num = divide_off(num, 2)
    num = divide_off(num, 5)
    gop_size = math.ceil(ideal_gop / num) * num
    seg_length = gop_size / framerate
    if (
        not gop_size
        or seg_length.denominator > 1_000_000
        or abs(seg_length - fv.ev.kf_target_sec) > 0.4
    ):
        fv.ev.use_timeline = True
        gop_size = math.ceil(ideal_gop)
        seg_length = gop_size / framerate
    if not fv.ev.vgop:
        fv.ev.kf_int = str(gop_size)
        fv.ev.min_kf_int = fv.ev.kf_int
    else:
        fv.ev.min_kf_int = str(int(framerate))
        fv.ev.kf_int = str(max(gop_size, int(fv.ev.min_kf_int)))
    fv.ev.kf_sec = f"{float(seg_length):.7f}"[:-1].rstrip("0").rstrip(".")
    fv.ev.bufsize = f"{2*int(fv.ev.vbitrate)}"
    fv.ev.latency_target = f"{4*float(fv.ev.kf_sec):.8g}"
    if fv.ev.vgop:
        fv.ev.kf_sec = f"{float(fv.ev.kf_sec)/2:.7f}"[:-1].rstrip("0").rstrip(".")
    logger.debug("min keyframe interval: %s", fv.ev.min_kf_int)
    logger.debug("keyframe interval: %s", fv.ev.kf_int)
    logger.debug("keyframe seconds: %s", fv.ev.kf_sec)
    logger.debug("latency target: %s", fv.ev.latency_target)


def determine_autocrop(fv: EncodeSession) -> None:
    crop_ts_num = ffmpeg.duration(fv.ev.crop_ts)
    crop_len_num = ffmpeg.duration(fv.ev.crop_len)
    flength = ffmpeg.duration(fv.v("f", "duration"))
    crop_ts_string = str(fv.ev.crop_ts)
    crop_len_string = str(fv.ev.crop_len)
    if flength is not None:
        if flength < crop_ts_num + crop_len_num:
            crop_ts_num = min(flength / 2, max(flength - crop_len_num, 0))
            crop_ts_string = f"{crop_ts_num:.3f}".rstrip("0").rstrip(".")
        if flength - crop_ts_num < crop_len_num:
            crop_len_num = flength - crop_ts_num
            crop_len_string = f"{crop_len_num:.3f}".rstrip("0").rstrip(".")
    ffprogress = ffmpeg.Progress[str]()
    prescale = [
        ffmpeg.Filter(
            "scale",
            f"w='{fv.ev.scale_w}'",
            f"h='{fv.ev.scale_h}'",
            "force_original_aspect_ratio=decrease",
            "force_divisible_by=2",
        )
    ]
    detect_filters: list[ffmpeg.Filter | str] = [
        *(prescale if not fv.ev.subcropfirst else ()),
        "cropdetect=round=2",
    ]
    # fmt: off
    cropargs = [
        ffmpeg.ff_bin.ffmpeg,
        *ffprogress.flags(0.1),
        *fv.ev.ff_deepprobe_flags,
        "-hide_banner",
        "-nostats",
        *(("-hwaccel", "auto") if fv.ev.hwaccel else ()),
        *(("-ss", crop_ts_string) if not fv.ev.slowseek else ()),
        *fv.fopts.main,
        *(("-ss", crop_ts_string) if fv.ev.slowseek else ()),
        "-an", "-sn",
        "-t", crop_len_string,
        "-sws_flags", "bilinear",
        "-vf", ffmpeg.Filter.vf_join(detect_filters),
        "-map", f"0:v:{fv.ev.vindex}",
        "-f", "null", "-",
    ]
    # fmt: on
    with subprocess.Popen(
        cropargs,
        text=True,
        encoding="utf-8",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=ffmpeg.ff_bin.env,
    ) as result:
        fv.crop.setstatus(StatusCode.RUNNING, "calculating")

        def consume_output() -> str | None:
            cropfilt = None
            crop_regex = re.compile(
                r"t:\s*(?P<time>[\d\.]*)\s+(?P<filter>crop=\S+)",
                flags=re.ASCII | re.MULTILINE,
            )
            for line in iter(ffprogress.output_que.get, None):
                if match := crop_regex.search(line):
                    cropfilt = match.group("filter")
            return cropfilt

        assert result.stderr is not None
        ffprogress.monitor_progress(
            result, result.stderr, make_queue=True, maxlen=50, loglevel=logging.DEBUG
        )
        future = fv.executor.submit(consume_output)
        ffprogress.progress_avail.wait()

        while not ffprogress.finished.is_set():
            fv.crop.setprogress(min(ffprogress.time_s / crop_len_num, 1))
            ffprogress.progress_avail.wait()
            ffprogress.progress_avail.clear()

        cropfilt = future.result()

    if result.returncode == 0 and cropfilt:
        fv.filts["vcrop"] = cropfilt
        logger.info("determined crop filter: %s", cropfilt)
        fv.crop.setstatus(StatusCode.FINISHED, "success")
    else:
        logger.warning("crop failed")
        fv.crop.setstatus(StatusCode.FAILED, "[red]failed")


def determine_afilters(fv: EncodeSession) -> None:
    if fv.fv("a", "channel_layout") == "mono":
        fv.ev.chlayout = "mono"
    if fv.fv("a", "channel_layout") != fv.ev.chlayout:
        fv.filts["adownmix"] = [
            "aresample",
            *(("resampler=soxr",) if fv.ev.soxr else ()),
            f"ocl={fv.ev.chlayout}",
        ]
    futures: list[concurrent.futures.Future[Any]] = []
    if fv.ev.anormalize:
        futures.append(fv.executor.submit(determine_anormalize, fv))

    max_samplerate = 48000
    if fv.ev.astandard == "opus":
        samplerate = "48000"
    else:
        if int(fv.v("a", "sample_rate")) < max_samplerate:
            samplerate = fv.v("a", "sample_rate")
        else:
            samplerate = str(max_samplerate)
    if fv.ev.anormalize or fv.fv("a", "sample_rate") != samplerate:
        fv.filts["aresample"] = [
            "aresample",
            *(("resampler=soxr", "precision=28") if fv.ev.soxr else ()),
            f"osr={samplerate}",
        ]
    fv.ev.samplerate = str(samplerate)
    if fv.ev.delay_start and fv.ev.timestamp is None:
        fv.filts["adelay"] = ["adelay", f"{fv.ev.start_delay}s", "all=1"]
    close_futures(futures)
    audiofilters = [
        *fv.filts.if_exists("adelay"),
        *fv.filts.if_exists("adownmix"),
        *fv.filts.if_exists("anormalize"),
        *fv.filts.if_exists("aresample"),
    ]
    fv.ev.afilters = ffmpeg.Filter.vf_join(audiofilters)
    logger.info("Determined afilters:\n%r", fv.ev.afilters)


def determine_anormalize(fv: EncodeSession) -> None:
    if fv.ev.dynamicnorm:
        fv.filts["anormalize"] = [
            "loudnorm",
            f"i={fv.ev.target_i}",
            f"lra={fv.ev.target_lra}",
            f"tp={fv.ev.target_tp}",
            *(("dual_mono=true",) if fv.ev.chlayout == "mono" else ()),
            "linear=false",
        ]
        fv.norm.setstatus(StatusCode.FINISHED, "success")
    else:

        def json_to_normfilt(json_map: Mapping[str, str]) -> None:
            logger.info(json_map)
            try:
                fv.filts["anormalize"] = [
                    "loudnorm",
                    f"i={fv.ev.target_i}",
                    f"lra={fv.ev.target_lra}",
                    f"tp={fv.ev.target_tp}",
                    *(("dual_mono=true",) if fv.ev.chlayout == "mono" else ()),
                    "linear=true",
                    f'measured_i={json_map["input_i"]}',
                    f'measured_lra={json_map["input_lra"]}',
                    f'measured_tp={json_map["input_tp"]}',
                    f'measured_thresh={json_map["input_thresh"]}',
                    f'offset={json_map["target_offset"]}',
                ]
            except KeyError as e:
                fv.norm.setstatus(StatusCode.FAILED, "[red]failed")
                logger.error("normalization json didn't contain expected keys")
                raise e
            else:
                fv.norm.setstatus(StatusCode.FINISHED, "success")

        normfile = fv.ev.normfile
        if normfile is None or not normfile.exists():
            normanalyze = [
                *fv.filts.if_exists("adownmix"),
                ffmpeg.Filter(
                    "loudnorm",
                    f"i={fv.ev.target_i}",
                    f"lra={fv.ev.target_lra}",
                    f"tp={fv.ev.target_tp}",
                    *(("dual_mono=true",) if fv.ev.chlayout == "mono" else ()),
                    "print_format=json",
                ),
            ]
            ffprogress = ffmpeg.Progress[str]()
            # fmt: off
            normargs = [
                ffmpeg.ff_bin.ffmpeg,
                *ffprogress.flags(0.25),
                *fv.ev.ff_deepprobe_flags,
                "-hide_banner",
                *(("-hwaccel", "auto") if fv.ev.hwaccel else ()),
                *fv.fopts.main,
                "-vn",
                "-sn",
                "-map", f"0:a:{fv.ev.aindex}",
                "-af", ffmpeg.Filter.vf_join(normanalyze),
                "-f", "null",
                "-",
            ]
            # fmt: on
            with subprocess.Popen(
                normargs,
                text=True,
                encoding="utf-8",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=ffmpeg.ff_bin.env,
            ) as result:
                length = ffmpeg.duration(fv.v("f", "duration")) or 1
                code = (
                    StatusCode.RUNNING
                    if fv.fv("f", "duration") is not None
                    else StatusCode.OTHER
                )
                fv.norm.setstatus(code, "calculating")
                assert result.stderr is not None
                ffprogress.monitor_progress(
                    result, result.stderr, maxlen=50, loglevel=logging.DEBUG
                )
                ffprogress.progress_avail.wait()
                while not ffprogress.finished.is_set():
                    fv.norm.setprogress(min(ffprogress.time_s / length, 1))
                    ffprogress.progress_avail.wait()
                    ffprogress.progress_avail.clear()

            if result.returncode == 0 and (
                jsonmatch := re.search(
                    r"(?P<json>{[^{]+})[^}]*$", "\n".join(ffprogress.output)
                )
            ):
                jsonout = json.loads(jsonmatch.group("json"))
                if normfile is not None:
                    normfile.write_text(
                        json.dumps(jsonout, separators=(",", ":")), encoding="utf-8"
                    )
                fv.norm.setstatus(StatusCode.OTHER, "read json")
                json_to_normfilt(jsonout)
            else:
                logger.info("\n".join(ffprogress.output))
                logger.warning("normalization failed")
                fv.norm.setstatus(StatusCode.FAILED, "[red]failed")
        elif normfile.is_file():
            fv.norm.setstatus(StatusCode.OTHER, "opening")
            jsonout = json.loads(normfile.read_text(encoding="utf-8"))
            fv.norm.setstatus(StatusCode.OTHER, "read file")
            json_to_normfilt(jsonout)
        else:
            fv.norm.setstatus(StatusCode.FAILED, "[red]failed")
            raise ValueError("provided normalization file isn't a file")
    if fv.norm.status is StatusCode.FINISHED:
        logger.info("anormalize filter:\n%s", fv.filts["anormalize"])


def determine_timeseek(fv: EncodeSession) -> None:
    if not fv.ev.live and fv.ev.timestamp is None and fv.ev.fix_start_time:
        vstart = ffmpeg.duration(fv.v("v", "start_time"))
        astart = ffmpeg.duration(fv.v("a", "start_time"))
        if vstart != astart:
            if fv.ev.copy_video:
                fv.ev.slowseek = True
            else:
                fv.ev.timestamp = f"{max(vstart, astart):f}"


def determine_bounds(fv: EncodeSession) -> None:
    fv.sdv("v", "width", fv.ev.target_w)
    fv.sdv("v", "height", fv.ev.target_h)
    if fv.ev.copy_video:
        fv.ev.bound_w = fv.v("v", "width")
        fv.ev.bound_h = fv.v("v", "height")
    elif fv.ev.upscale:
        fv.ev.bound_w = fv.ev.target_w
        fv.ev.bound_h = fv.ev.target_h
    else:
        fv.ev.bound_w = str(min(int(fv.v("v", "width")), int(fv.ev.target_w)))
        fv.ev.bound_h = str(min(int(fv.v("v", "height")), int(fv.ev.target_h)))


_PICSUB_NAMES: Final = {"dvb_subtitle", "dvd_subtitle", "hdmv_pgs_subtitle", "xsub"}


def determine_subtitles(fv: EncodeSession) -> None:
    if fv.fv("s", "codec_type") != "subtitle":
        fv.subs.setstatus(StatusCode.FAILED, "[red]failed")
        logger.warning(
            "No subtitles detected in file, but subtitles were marked as enabled."
        )
        return
    if fv.fv("s", "codec_name") in _PICSUB_NAMES:
        fv.ev.text_subs = False
        fv.ev.subfilter_list = get_picsub_list(fv)
    else:
        fv.ev.text_subs = True
        fv.ev.subfilter_list = get_textsub_list(fv)


class StyleFile(NamedTuple):
    names: MutableSequence[str]
    lines: MutableSequence[str]
    insert_index: int


def parse_stylelines(ass_text: Sequence[str]) -> StyleFile | None:
    section_index = next(
        (
            i
            for i, line in enumerate(ass_text)
            if re.match(r"\[.*?styles\+?", line.strip(), flags=re.IGNORECASE)
        ),
        None,
    )
    if section_index is None:
        return None
    format_index = next(
        (
            i
            for i, line in enumerate(ass_text[section_index + 1 :])
            if line.strip().lower().startswith("format:")
        ),
        None,
    )
    if format_index is None:
        return None
    insert_index = section_index + format_index + 2
    last_index = next(
        (
            i
            for i, line in enumerate(ass_text[insert_index:])
            if re.match(r"\[.*\]", line.strip())
        ),
        None,
    )
    if last_index is None:
        return None
    last_index += insert_index
    lines = [
        line
        for line in ass_text[insert_index:last_index]
        if line.strip().lower().startswith("style:")
    ]
    format_line = ass_text[section_index + format_index + 1]
    format_list = format_line.strip().lower().removeprefix("format:").split(",")
    name_index = next(
        (i for i, field in enumerate(format_list) if field.strip() == "name"), None
    )
    if name_index is None:
        return None
    names = [line.strip()[6:].strip().split(",")[name_index] for line in lines]
    return StyleFile(names=names, lines=lines, insert_index=insert_index)


def extract_style(
    file: pathlib.Path, sindex: int, deep_probe: bool = False
) -> StyleFile | None:
    extradata = ffmpeg.ff_bin.probe_val(
        "extradata",
        str(file),
        f"s:{sindex}",
        ffmpeg.ProbeType.STREAM,
        deep_probe,
        "-show_data",
    )
    if extradata is None:
        return None
    header = bytes.fromhex(
        "".join(line[10:49] for line in extradata.splitlines())
    ).decode("utf-8", errors="surrogateescape")
    return parse_stylelines(header.splitlines())


def extract_styles(
    files: Sequence[pathlib.Path], sindex: int, deep_probe: bool = False
) -> Iterator[StyleFile]:
    executor = concurrent.futures.ThreadPoolExecutor()
    try:
        futures = [
            executor.submit(extract_style, file, sindex, deep_probe) for file in files
        ]
    finally:
        executor.shutdown(wait=False)

    def style_iterator() -> Iterator[StyleFile]:
        for future in concurrent.futures.as_completed(futures):
            if (result := future.result()) is not None:
                yield result

    return style_iterator()


def get_textsub_list(fv: EncodeSession) -> list[ffmpeg.Filter]:
    subpath = fv.fopts.sfpath
    subindex = fv.ev.sindex
    if not fv.ev.subfile_provided or fv.ev.timestamp is not None:
        CONVERT_SUBS: Final = {"eia_608"}
        container_map: dict[str | None, str] = {k: "mkv" for k in CONVERT_SUBS}
        container_map |= {"mov_text": "mp4"}
        sub_container = container_map.get(fv.fv("s", "codec_name"), "mkv")
        subpath = fv.ev.tempdir / f"subs.{sub_container}"
        subindex = 0
        ffprogress = ffmpeg.Progress[str]()
        # fmt: off
        subargs = [
            ffmpeg.ff_bin.ffmpeg,
            *ffprogress.flags(0.1),
            *fv.ev.ff_deepprobe_flags,
            "-hide_banner",
            *(("-hwaccel", "auto") if fv.ev.hwaccel else ()),
            "-fix_sub_duration",
            *(("-itsoffset", fv.ev.suboffset) if fv.ev.suboffset is not None else ()),
            *fv.fopts.subtitle,
            *(
                ("-ss", "0")
                if fv.ev.timestamp is None
                else ("-ss", fv.ev.timestamp)
            ),
            "-vn", "-an",
            "-c:s", "copy" if fv.fv("s", "codec_name") not in CONVERT_SUBS else "ass",
            "-map", f"0:s:{fv.ev.sindex}",
            "-map", "0:t?",
            "-y",
            str(subpath),
        ]
        # fmt: on
        # TODO: does -an -vn make a difference if playlist isn't duration'd?
        with subprocess.Popen(
            subargs,
            text=True,
            encoding="utf-8",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=ffmpeg.ff_bin.env,
        ) as result:
            fv.subs.setstatus(StatusCode.RUNNING, "extracting")
            length = ffmpeg.duration(fv.v("f", "duration"))
            assert result.stderr is not None
            ffprogress.monitor_progress(
                result, result.stderr, maxlen=50, loglevel=logging.DEBUG
            )
            ffprogress.progress_avail.wait()
            while not ffprogress.finished.is_set():
                fv.subs.setprogress(min(ffprogress.time_s / length, 1))
                ffprogress.progress_avail.wait()
                ffprogress.progress_avail.clear()

        if result.returncode != 0:
            fv.subs.setstatus(StatusCode.FAILED, "[red]failed")
            logger.warning(
                "extracting subtitles failed with exit code %s", result.returncode
            )
            return []
        if fv.ev.is_playlist and fv.fv("s", "codec_name") == "ass":
            fv.subs.setstatus(StatusCode.OTHER, "merging")
            stylegen = extract_styles(
                fv.fopts.allpaths, cast(int, fv.ev.sindex), fv.ev.deep_probe
            )
            subass = fv.ev.tempdir / "subs.ass"
            # fmt: off
            subextract_args = [
                ffmpeg.ff_bin.ffmpeg,
                "-hide_banner",
                "-i", str(subpath),
                "-vn", "-an",
                "-c:s", "copy",
                "-map", f"0:s:{subindex}",
                str(subass),
            ]
            # fmt: on
            extract_result = subprocess.run(
                subextract_args,
                env=ffmpeg.ff_bin.env,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if extract_result.returncode != 0:
                fv.subs.setstatus(StatusCode.FAILED, "[red]failed")
                logger.warning(
                    "extracting merged subtitles failed with exit code %s",
                    result.returncode,
                )
                return []
            sublines = subass.read_text(
                encoding="utf-8-sig", errors="surrogateescape"
            ).splitlines()
            mainstyle = parse_stylelines(sublines)
            if mainstyle is not None:
                for style in stylegen:
                    for i, name in enumerate(style.names):
                        if name not in mainstyle.names:
                            sublines.insert(mainstyle.insert_index, style.lines[i])
                            mainstyle.names.append(name)
                with subass.open(
                    mode="w", encoding="utf-8", errors="surrogateescape"
                ) as f:
                    f.writelines(line + "\n" for line in sublines)
            in_flags: list[str] = []
            attach_flags: list[str] = []
            for i, file in enumerate(fv.fopts.allpaths):
                in_flags += ["-i", str(file)]
                attach_flags += ["-map", f"{i+1}:t?"]
            # fmt: off
            submerge_args = [
                ffmpeg.ff_bin.ffmpeg,
                "-hide_banner",
                "-i", str(subass),
                *in_flags,
                "-vn", "-an",
                "-c:s", "copy",
                "-map", "0:s:0",
                *attach_flags,
                "-y",
                str(subpath),
            ]
            # fmt: on
            merge_result = subprocess.run(
                submerge_args,
                env=ffmpeg.ff_bin.env,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subass.unlink()
            if merge_result.returncode != 0:
                fv.subs.setstatus(StatusCode.FAILED, "[red]failed")
                logger.warning(
                    "merging subtitles with attachments failed with exit code %s",
                    result.returncode,
                )
                return []
    subfilter = ffmpeg.Filter(
        "subtitles", ffmpeg.Filter.full_escape(str(subpath)), f"si={subindex}"
    )
    fv.subs.setstatus(StatusCode.FINISHED, "success")
    return [subfilter]


def get_picsub_list(fv: EncodeSession) -> list[ffmpeg.Filter]:
    if fv.ev.subcropfirst or not fv.ev.vulkan:
        if re.fullmatch(r"yuv[ja]?4[0-4]{2}p", fv.v("v", "pix_fmt")):
            overlay_fmt = "yuv420"
        else:
            overlay_fmt = "yuv420p10"
    else:
        overlay_fmt = "yuv420" if fv.ev.eightbit else "yuv420p10"
    subfilter_list = [
        ffmpeg.Filter(
            "premultiply",
            "inplace=1",
            src=[f"{int(fv.ev.subfile_provided)}:s:{fv.ev.sindex}"],
        ),
        *(
            (
                ffmpeg.Filter(
                    "setpts", f"PTS{ffmpeg.duration(fv.ev.suboffset):+0.6f}/TB"
                ),
            )
            if fv.ev.suboffset is not None
            else ()
        ),
        ffmpeg.Filter(
            "scale2ref",
            "w='trunc(min(ih*mdar,iw))'",
            "h='trunc(min(ih,iw/mdar))'",
            f"flags={fv.ev.picsubscale}",
            src=[1, 0],
            dst=[0, 1],
        ),
        ffmpeg.Filter("overlay", "(W-w)/2", "H-h", f"format={overlay_fmt}", src=[1, 0]),
        ffmpeg.Filter(f"format={fv.ev.pix_fmt}"),
    ]
    fv.subs.setstatus(StatusCode.FINISHED, "success")
    return subfilter_list


def determine_scale(fv: EncodeSession) -> None:
    if fv.ev.upscale:
        softflags = "lanczos"
        fv.ev.scale_w = fv.ev.target_w
        fv.ev.scale_h = fv.ev.target_h
    else:
        softflags = "spline"
        fv.ev.scale_w = f"min({fv.ev.target_w},iw)"
        fv.ev.scale_h = f"min({fv.ev.target_h},ih)"
    if fv.ev.vulkan:
        libplacebo = [
            "libplacebo",
            f"w='{fv.ev.scale_w}'",
            f"h='{fv.ev.scale_h}'",
            "force_original_aspect_ratio=decrease",
            "force_divisible_by=2",
            "pad_crop_ratio=1",
            "upscaler=ewa_lanczos",
            "downscaler=mitchell",
            "colorspace=bt709",
            "color_primaries=bt709",
            "color_trc=bt709",
            "range=tv",
            *fv.ev.placebo_opts,
            f"format={fv.ev.pix_fmt}",
        ]
        custom_shaders = [
            pathlib.Path(s) for s in fv.ev.shader_list if pathlib.Path(s).is_file()
        ]
        shader_names = {s.stem.lower() for s in custom_shaders}
        if not fv.ev.upscale and "ssimdownscaler" in shader_names:
            libplacebo.append("disable_linear=1")
        if len(custom_shaders) < len(fv.ev.shader_list):
            logger.warning("Could not find all specified shaders in list.")
        if custom_shaders:
            shader_concat = fv.ev.tempdir / "customshaders.glsl"
            with shader_concat.open(mode="w", encoding="utf-8") as f:
                for shader in custom_shaders:
                    f.write(shader.read_text(encoding="utf-8"))
            libplacebo.append(
                f"custom_shader_path={ffmpeg.Filter.full_escape(str(shader_concat))}"
            )
        if fv.ev.trust_vulkan:
            fv.filts["scale"] = libplacebo
        else:
            fv.filts["scale"] = f"scale_vulkan,{ffmpeg.Filter(*libplacebo)}"
    elif fv.ev.zscale:
        if fv.ev.upscale:
            # TODO: implement zscale upscaling
            logger.error(
                "Upscale specified while using zscale. Video will not be upscaled."
            )
        fv.filts["scale"] = [
            "zscale",
            f"w='trunc(min(1,min({fv.ev.target_w}/iw,{fv.ev.target_h}/ih))*iw/2)*2'",
            f"h='trunc(min(1,min({fv.ev.target_w}/iw,{fv.ev.target_h}/ih))*ih/2)*2'",
            "filter=spline36",
        ]
    else:
        fv.filts["scale"] = [
            "scale",
            f"w='{fv.ev.scale_w}'",
            f"h='{fv.ev.scale_h}'",
            "force_original_aspect_ratio=decrease",
            f"force_divisible_by=2:flags={softflags}",
        ]


def close_futures(futures: Iterable[concurrent.futures.Future[Any]]) -> None:
    """Wait on futures in list, then raise any exceptions in them."""
    concurrent.futures.wait(futures)
    for future in futures:
        if exception := future.exception(5):
            raise exception


def determine_decimation(_: EncodeSession) -> None:
    raise NotImplementedError("video decimation is not currently implemented")


def get_hwtransfer(
    fv: EncodeSession,
) -> tuple[list[ffmpeg.Filter], list[ffmpeg.Filter], list[ffmpeg.Filter]]:
    prescale_transfer: list[ffmpeg.Filter] = []
    sw_transfer: list[ffmpeg.Filter] = []
    encode_transfer: list[ffmpeg.Filter] = []

    if not fv.ev.subs and not fv.ev.crop and not fv.ev.cleanborders:
        fv.ev.subcropfirst = True

    need_postscale_sw = not fv.ev.subcropfirst

    if fv.ev.vulkan:
        if not fv.ev.trust_vulkan:
            prescale_transfer += [ffmpeg.Filter("hwupload")]
        prescale_transfer += [ffmpeg.Filter("hwupload", "derive_device=vulkan")]

    if fv.ev.vulkan and need_postscale_sw:
        sw_transfer += [
            ffmpeg.Filter("hwdownload"),
            ffmpeg.Filter("format", fv.ev.pix_fmt),
        ]

    if fv.ev.vulkan and not need_postscale_sw:
        if fv.ev.vencoder in fv.ev.SW_ENCODERS:
            encode_transfer += [
                ffmpeg.Filter("hwdownload"),
                ffmpeg.Filter("format", fv.ev.pix_fmt),
            ]
        elif fv.ev.vencoder in fv.ev.NVIDIA_ENCODERS:
            encode_transfer += [ffmpeg.Filter("hwupload", "derive_device=cuda")]
        else:
            raise ValueError(f"No valid transfer found for encoder {fv.ev.vencoder!r}.")
    else:
        encode_transfer += [ffmpeg.Filter("format", fv.ev.pix_fmt)]
        if fv.ev.vencoder in fv.ev.NVIDIA_ENCODERS:
            encode_transfer += [ffmpeg.Filter("hwupload", "derive_device=cuda")]

    return prescale_transfer, sw_transfer, encode_transfer


def determine_vfilters(fv: EncodeSession) -> None:
    futures: list[concurrent.futures.Future[Any]] = []
    if fv.ev.subs:
        futures.append(fv.executor.submit(determine_subtitles, fv))
    determine_scale(fv)
    if fv.ev.crop:
        futures.append(fv.executor.submit(determine_autocrop, fv))
    if fv.ev.obs:
        determine_decimation(fv)
    if fv.ev.deinterlace:
        fv.filts["deinterlace"] = ["bwdif", "0"]
    if fv.ev.cleanborders:
        fv.filts["cleanborders"] = ["fillborders", *map(str, fv.ev.cleanborders)]
    if fv.ev.end_pad and not fv.ev.live and fv.ev.outfile is None:
        fv.filts["endpadfilt"] = ["tpad", f"stop_duration={fv.ev.end_delay}"]
    if fv.ev.delay_start:
        fv.filts["startpadfilt"] = ["tpad", f"start_duration={fv.ev.start_delay}"]
    prescale_transfer, sw_transfer, encode_transfer = get_hwtransfer(fv)
    close_futures(futures)
    crop_and_clean = [*fv.filts.if_exists("vcrop"), *fv.filts.if_exists("cleanborders")]
    sub_and_crop = [
        *(crop_and_clean if not fv.ev.cropsecond else ()),
        *fv.ev.subfilter_list,
        *(crop_and_clean if fv.ev.cropsecond else ()),
    ]
    vfilter_list = [
        *fv.filts.if_exists("deinterlace"),
        *fv.ev.sw_filters,
        *fv.filts.if_exists("startpadfilt"),
        *fv.filts.if_exists("endpadfilt"),
        *(sub_and_crop if fv.ev.subcropfirst else ()),
        *prescale_transfer,
        fv.filts["scale"],
        *sw_transfer,
        *(sub_and_crop if not fv.ev.subcropfirst else ()),
        *encode_transfer,
    ]
    fv.ev.filter_complex = ffmpeg.Filter.complex_join(
        vfilter_list, startkey=f"0:v:{fv.ev.vindex}", endkey="b"
    )
    logger.info("Determined vfilters:\n%r", fv.ev.filter_complex)


def get_x264_flags(fv: EncodeSession) -> list[str]:
    x264_params = ":".join(
        [
            "open-gop=0",
            *(("scenecut=0",) if not fv.ev.vgop else ()),
            *fv.ev.vencoder_params,
        ]
    )
    # fmt: off
    flags = [
        "-c:v", "libx264",
        "-profile:v", "high",
        "-preset:v", fv.ev.x26X_preset,
        "-g:v", f"{fv.ev.kf_int}",
        "-keyint_min:v", f"{fv.ev.min_kf_int}",
        *(("-sc_threshold:v", "0") if not fv.ev.vgop else ()),
        "-forced-idr:v", "1",
        *(("-x264-params:v", x264_params) if x264_params else ()),
        *(("-pass", f"{fv.ev.npass}") if fv.ev.npass is not None else ()),
        *(("-passlogfile", f"{fv.ev.passfile}") if fv.ev.passfile is not None else ()),
        *(("-tune", f"{fv.ev.x26X_tune}") if fv.ev.x26X_tune is not None else ()),

        "-b:v", f"{fv.ev.vbitrate}",
        "-maxrate:v", f"{fv.ev.max_vbitrate}",
        "-bufsize:v", f"{fv.ev.bufsize}",
    ]
    # fmt: on
    return flags


def get_x265_flags(fv: EncodeSession) -> list[str]:
    x265_params = ":".join(
        [
            "open-gop=0",
            *(("scenecut=0",) if not fv.ev.vgop else ()),
            *((f"pass={fv.ev.npass}",) if fv.ev.npass is not None else ()),
            *(
                (f"stats={ffmpeg.single_quote(fv.ev.passfile)}",)
                if fv.ev.passfile is not None
                else ()
            ),
            *fv.ev.vencoder_params,
        ]
    )
    # fmt: off
    flags = [
        "-c:v", "libx265",
        "-profile:v", *(("main",) if fv.ev.eightbit else ("main10",)),
        "-preset:v", fv.ev.x26X_preset,
        "-g:v", f"{fv.ev.kf_int}",
        "-keyint_min:v", f"{fv.ev.min_kf_int}",
        "-forced-idr:v", "1",
        *(("-x265-params:v", x265_params) if x265_params else ()),
        *(("-tune", f"{fv.ev.x26X_tune}") if fv.ev.x26X_tune is not None else ()),

        "-tag:v", "hvc1",
        "-b:v", f"{fv.ev.vbitrate}",
        "-maxrate:v", f"{fv.ev.max_vbitrate}",
        "-bufsize:v", f"{fv.ev.bufsize}",
    ]
    # fmt: on
    return flags


# TODO: enable SEI when nvidia fixes their driver (495 series)
# TODO: make workaround and encode options dependent on ffmpeg version
def get_nvenc_hevc_flags(fv: EncodeSession) -> list[str]:
    # fmt: off
    flags = [
        "-c:v", "hevc_nvenc",
        "-profile:v", *(("main",) if fv.ev.eightbit else ("main10",)),
        "-threads:v", "3",
        "-g:v", f"{fv.ev.kf_int}",
        "-keyint_min:v", f"{fv.ev.min_kf_int}",
        "-forced-idr:v", "1",
        *(("-no-scenecut:v", "1") if not fv.ev.vgop else ()),
        "-spatial-aq:v", "1",
        "-temporal-aq:v", "1",

        "-preset:v", "p7",
        "-tune:v", "hq",
        "-rc:v", "cbr",
        "-multipass:v", "fullres",
        "-bf:v", "3",
        "-b_ref_mode:v", "each",
        "-rc-lookahead:v", "32",
        *min_version(("-s12m_tc:v", "0"), ("libavcodec", "59.1.101")),
        *min_version(("-extra_sei:v", "0"), ("libavcodec", "59.1.101")),

        "-tag:v", "hvc1",
        "-b:v", f"{fv.ev.vbitrate}",
        "-maxrate:v", f"{fv.ev.max_vbitrate}",
        "-bufsize:v", f"{fv.ev.bufsize}",
    ]
    # fmt: on
    return flags


# TODO: enable SEI when nvidia fixes their driver (495 series)
# TODO: make workaround and encode options dependent on ffmpeg version
def get_nvenc_h264_flags(fv: EncodeSession) -> list[str]:
    # fmt: off
    flags = [
        "-c:v", "h264_nvenc",
        "-profile:v", "high",
        "-threads:v", "3",
        "-g:v", f"{fv.ev.kf_int}",
        "-keyint_min:v", f"{fv.ev.min_kf_int}",
        "-forced-idr:v", "1",
        *(("-no-scenecut:v", "1") if not fv.ev.vgop else ()),
        "-rc-lookahead:v", "32",
        "-coder:v", "cabac",
        "-bf:v", "3",
        "-b_ref_mode:v", "middle",
        "-spatial-aq:v", "1",
        "-temporal-aq:v", "1",
        # "-strict_gop", "1",

        "-preset:v", "p7",
        "-tune:v", "hq",
        "-rc:v", "cbr",
        "-multipass:v", "fullres",
        *min_version(("-extra_sei:v", "0"), ("libavcodec", "59.1.101")),

        "-b:v", f"{fv.ev.vbitrate}",
        "-maxrate:v", f"{fv.ev.max_vbitrate}",
        "-bufsize:v", f"{fv.ev.bufsize}",
    ]
    # fmt: on
    return flags


def get_aflags(fv: EncodeSession) -> list[str]:
    aflags: list[str] = []
    if fv.ev.npass in (1, 3):
        return ["-an"]
    if fv.ev.copy_audio:
        fv.ev.astandard = fv.v("a", "codec_name")
        fv.ev.samplerate = fv.fv("a", "sample_rate") or "unknown"
        fv.ev.chlayout = fv.fv("a", "channel_layout") or "unknown"
        # better expressed with PEP 505
        fv.ev.abitrate = (
            fv.fv("a", "bit_rate")
            or fv.fv("a", "BPS", ffmpeg.ProbeType.TAGS)
            or fv.fv("a", "BPS-eng", ffmpeg.ProbeType.TAGS)
            or fv.ev.abitrate
        )
        aflags += ["-c:a", "copy"]
        logger.debug("Using abitrate for copy: %s", fv.ev.abitrate)
        return aflags
    # TODO: change to match case in 3.10
    if fv.ev.aencoder == "aac":
        # fmt: off
        aflags += [
            "-c:a", "aac",
            "-b:a", f"{fv.ev.abitrate}",
            "-cutoff:a", "19000",
        ]
        # fmt: on
        return aflags
    elif fv.ev.aencoder == "libfdk_aac":
        # fmt: off
        aflags += [
            "-c:a", "libfdk_aac",
            "-b:a", f"{fv.ev.abitrate}",
            "-cutoff:a", "19000",
        ]
        # fmt: on
        return aflags
    elif fv.ev.aencoder == "libopus":
        aflags += ["-c:a", "libopus", "-b:a", f"{fv.ev.abitrate}"]
        return aflags
    raise ValueError("No valid audio encoder parameter selection found.")


def get_vflags(fv: EncodeSession) -> list[str]:
    vflags: list[str] = []
    if fv.ev.copy_video:
        fv.ev.vstandard = fv.v("v", "codec_name")
        fv.ev.pix_fmt = fv.fv("v", "pix_fmt") or "unknown"
        # better expressed with PEP 505: or -> ??
        fv.ev.vbitrate = (
            fv.fv("v", "bit_rate")
            or fv.fv("v", "BPS", ffmpeg.ProbeType.TAGS)
            or fv.fv("v", "BPS-eng", ffmpeg.ProbeType.TAGS)
            or fv.fv("f", "bit_rate")
            or fv.ev.vbitrate
        )
        vflags += ["-c:v", "copy"]
        logger.debug("Using vbitrate for copy: %s", fv.ev.vbitrate)
        return vflags
    # TODO: match case in 3.10
    if fv.ev.vencoder == "libx264":
        return get_x264_flags(fv)
    elif fv.ev.vencoder == "libx265":
        return get_x265_flags(fv)
    elif fv.ev.vencoder == "hevc_nvenc":
        return get_nvenc_hevc_flags(fv)
    elif fv.ev.vencoder == "h264_nvenc":
        return get_nvenc_h264_flags(fv)
    raise ValueError(
        f"No valid video encoder parameter selection found: {fv.ev.vencoder!r}"
    )


def set_input_flags(fv: EncodeSession) -> None:
    hwaccel_flags = []
    if fv.ev.hwaccel and fv.ev.vencoder in fv.ev.NVIDIA_ENCODERS:
        # fmt: off
        hwaccel_flags = [
            "-hwaccel", "cuda",
            "-init_hw_device", "cuda=cud",
            "-hwaccel_device", "cud",
            "-filter_hw_device", "cud",
        ]
        # fmt: on
    elif fv.ev.hwaccel:
        hwaccel_flags = ["-hwaccel", "auto"]
    if fv.ev.vulkan and "vulkan" in ffmpeg.ff_bin.hwaccels:
        device_str = "" if fv.ev.vulkan_device == -1 else f":{fv.ev.vulkan_device}"
        hwaccel_flags += ["-init_hw_device", f"vulkan=vulk{device_str}"]
    input_flags = [
        *fv.ev.ffprogress.flags(0.25),
        *fv.ev.ff_verbosity_flags,
        *fv.ev.ff_deepprobe_flags,
        *hwaccel_flags,
        "-threads:v",
        "0",
    ]
    if fv.ev.obs:
        # TODO: implement OBS
        fv.ev.input_flags = input_flags
        return
    input_flags += [
        *(("-thread_queue_size", "4096") if not fv.ev.live else ()),
        *(("-noaccurate_seek",) if fv.ev.timestamp and fv.ev.copy_video else ()),
        *(("-ss", fv.ev.timestamp) if fv.ev.timestamp and not fv.ev.slowseek else ()),
        *(("-re",) if not fv.ev.live and fv.ev.outfile is None else ()),
        *fv.fopts.main,
        *(("-ss", fv.ev.timestamp) if fv.ev.timestamp and fv.ev.slowseek else ()),
        *(("-t", fv.ev.clip_length) if fv.ev.clip_length else ()),
    ]
    if fv.ev.subfile_provided and not fv.ev.text_subs:
        input_flags += [
            *fv.ev.ff_deepprobe_flags,
            *(("-hwaccel", "auto") if fv.ev.hwaccel else ()),
            "-thread_queue_size",
            "4096",
            *fv.fopts.subtitle,
        ]
    fv.ev.input_flags = input_flags


def get_fifo_flags(fifo_format: str) -> list[str]:
    # fmt: off
    return [
        "-f", "fifo",
        "-fifo_format", fifo_format,
        "-attempt_recovery", "1",
        "-recovery_wait_time", "1",
        "-queue_size", "200",
    ]
    # fmt: on


def set_output_flags(fv: EncodeSession) -> None:
    if fv.ev.outfile is not None:
        if fv.ev.npass in (1, 3):
            fv.ev.output_flags = ["-f", "null", "-"]
        else:
            fv.ev.output_flags = [str(fv.ev.outfile)]
        return
    if fv.ev.protocol == "srt":
        set_srt_flags(fv)
    elif fv.ev.protocol == "rtmp":
        rtmp_flags: list[str] = []
        if fv.ev.fifo:
            rtmp_flags += get_fifo_flags("flv")
        rtmp_flags += [f"rtmp://{fv.ev.endpoint}"]
        fv.ev.output_flags = rtmp_flags
    else:
        raise ValueError(f"Invalid stream protocol passed: {fv.ev.protocol!r}")


def set_srt_flags(fv: EncodeSession) -> None:
    RTT_SEC: Final = 0.3
    BPS_MULTIPLIER: Final = 5.0
    BPS: Final[int] = int(BPS_MULTIPLIER * int(fv.ev.vbitrate) + int(fv.ev.abitrate))
    IPV4_MAX_HEADER: Final[int] = 20
    UDP_SIZE: Final[int] = 8
    UDPHDR_SIZE: Final = IPV4_MAX_HEADER + UDP_SIZE
    SRT_HEADER: Final[int] = 16
    HEADER_SIZE = UDPHDR_SIZE + SRT_HEADER
    MTU: Final[int] = 1500
    # alternatively payload can be 1316 for mpegts
    PAYLOAD_SIZE: Final[int] = MTU - HEADER_SIZE
    MSS: Final[int] = PAYLOAD_SIZE + HEADER_SIZE
    LATENCY_SEC: Final = fv.ev.srt_latency
    LATENCY_USEC: Final[int] = int(LATENCY_SEC * 1_000_000)
    FULL_LATENCY_SEC: Final = LATENCY_SEC + RTT_SEC / 2
    TARGET_PAYLOAD_BYTES: Final = FULL_LATENCY_SEC * BPS / 8
    FC_WINDOW: Final[int] = math.ceil(TARGET_PAYLOAD_BYTES / PAYLOAD_SIZE)
    RCVBUF: Final[int] = math.ceil(FC_WINDOW * (MSS - UDPHDR_SIZE))
    srt_options = [
        "mode=caller",
        "transtype=live",
        "tlpktdrop=0",
        *((f"passphrase={fv.ev.srt_passphrase}",) if fv.ev.srt_passphrase else ()),
        "pbkeylen=32",
        f"mss={MSS}",
        f"payload_size={PAYLOAD_SIZE}",
        "connect_timeout=8000",
        f"rcvlatency={LATENCY_USEC}",
        f"peerlatency={LATENCY_USEC}",
        f"ffs={FC_WINDOW}",
        f"rcvbuf={RCVBUF}",
    ]
    srt_opts = "&".join(srt_options)
    srt_flags = ["-flush_packets", "0"]
    srt_format = "matroska" if fv.ev.pyffserver else "mpegts"
    if fv.ev.fifo:
        srt_flags += get_fifo_flags(srt_format)
    else:
        srt_flags += ["-f", srt_format]
    srt_flags += [f"srt://{fv.ev.endpoint}?{srt_opts}"]
    fv.ev.output_flags = srt_flags


def set_filter_flags(fv: EncodeSession) -> None:
    filter_flags: list[str] = []
    if fv.ev.copy_video:
        filter_flags += ["-map", f"0:v:{fv.ev.vindex}"]
    else:
        # fmt: off
        filter_flags += [
            "-sws_flags", "lanczos+accurate_rnd+full_chroma_int",
            "-filter_complex", fv.ev.filter_complex,
            "-map", "[b]",
        ]
        # fmt: on
    if fv.ev.npass not in (1, 3):
        if fv.ev.copy_audio:
            filter_flags += ["-map", f"0:a:{fv.ev.aindex}?"]
        else:
            filter_flags += [
                "-map",
                f"0:a:{fv.ev.aindex}?",
                *(("-resampler", "soxr") if fv.ev.soxr else ()),
                *(("-af", fv.ev.afilters) if fv.ev.afilters else ()),
            ]
    filter_flags += ["-map_metadata", "-1", "-map_chapters", "-1"]
    fv.ev.filter_flags = filter_flags


def set_ffmpeg_flags(fv: EncodeSession) -> None:
    ff_flags = [
        ffmpeg.ff_bin.ffmpeg,
        *fv.ev.input_flags,
        *fv.ev.filter_flags,
        *fv.ev.encode_flags,
        *fv.ev.output_flags,
    ]
    logger.debug("Encode command:\n%r", ff_flags)
    logger.info("Encode command:\n%s", " ".join(ff_flags))
    fv.ev.ff_flags = ff_flags
