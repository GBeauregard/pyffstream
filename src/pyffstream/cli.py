"""CLI frontend for encoding and streaming."""
from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import configparser
import copy
import dataclasses
import itertools
import logging
import logging.handlers
import os
import pathlib
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Iterable, Sequence
from typing import Any, Final, NamedTuple

import platformdirs
import requests
import rich.box
import rich.columns
import rich.console
import rich.live
import rich.logging
import rich.markup
import rich.progress
import rich.prompt
import rich.table
import rich.text

from . import APPNAME, encode, ffmpeg

# import rich.traceback
# rich.traceback.install(console=console)

logger = logging.getLogger(__name__)
logging.getLogger("requests").propagate = False
logging.getLogger("urllib3").propagate = False

console = rich.console.Console()


def get_stream_list(
    streamtype: str,
    q_tuple: ffmpeg.StreamQueryTuple,
    myfileargs: Sequence[str],
    deep_probe: bool = False,
) -> list[list[tuple[str, str]]]:
    """Make and return tuples of (key,val) pairs for each stream."""
    outjson = ffmpeg.ff_bin.probe(
        ffmpeg.format_q_tuple(q_tuple, True),
        myfileargs,
        streamtype,
        probetype=ffmpeg.ProbeType.RAW,
        extraargs="-pretty",
        deep_probe=deep_probe,
    )
    if outjson is None:
        logger.error(f"getting info for stream {streamtype} failed")
        return []
    if not (allstreams := outjson.get("streams")):
        return []
    stream_list = []
    BAD_VALS: Final[set[str | int]] = {"N/A", "unknown"}
    for s in allstreams:
        val_list = []
        ituple = (
            ("", s, q_tuple[0], BAD_VALS),
            ("tags: ", s.get("tags", {}), q_tuple[1], BAD_VALS),
            ("disposition: ", s.get("disposition", {}), q_tuple[2], BAD_VALS | {0}),
        )
        for prefix, sdict, valid_keys, forbidden in ituple:
            val_list += [
                (prefix + key, str(val))
                for key, val in sdict.items()
                if key in valid_keys and val not in forbidden
            ]
        stream_list.append(val_list)
    return stream_list


def highlight_path(path: os.PathLike[Any]) -> str:
    pl_path = pathlib.Path(path)
    name = str(pl_path.name)
    parent = str(pl_path).removesuffix(name)
    parent = "[magenta]" + rich.markup.escape(parent) if parent else ""
    name = "[bright_magenta]" + rich.markup.escape(name)
    return parent + name


class InfoKeys:
    VSTREAMS: Final = {
        "codec_name",
        "width",
        "height",
        "r_frame_rate",
        "field_order",
        "pix_fmt",
        "color_space",
        "color_transfer",
        "color_primaries",
        "color_range",
    }
    VSTREAM_TAGS: Final = {
        "title",
        "language",
    }
    ASTREAMS: Final = {
        "codec_name",
        "channels",
        "channel_layout",
        "sample_fmt",
        "bits_per_raw_sample",
        "sample_rate",
        "bit_rate",
    }
    ASTREAM_TAGS: Final = {
        "title",
        "language",
    }
    SSTREAMS: Final = {
        "codec_name",
    }
    SSTREAM_TAGS: Final = {
        "title",
        "language",
    }
    DISPOSITIONS: Final = {
        "default",
        "forced",
        "dub",
        "original",
        "comment",
        "lyrics",
        "karaoke",
        "hearing_impaired",
        "visual_impaired",
        "clean_effects",
        "attached_pic",
        "timed_thumbnails",
        "captions",
        "descriptions",
        "metadata",
        "dependent",
        "still_image",
    }


def print_info(fopts: encode.FileOpts, deep_probe: bool = False) -> None:
    """Prints to console formatted information about the input file.

    Output is nicely formatted for console usage using rich tables.

    Args:
        fopts: The file to print information about.
        deep_probe: Whether or not to probe the file deeply.
    """
    probefargs = copy.copy(fopts.main)
    probesfargs = copy.copy(fopts.subtitle)
    probefargs.pop(-2)
    probesfargs.pop(-2)

    def make_columns(title: str, *args: Any) -> str:
        slist = get_stream_list(*args)
        table_list = []
        for subindex, vallist in enumerate(slist):
            table = rich.table.Table(
                title=f"{subindex}",
                show_header=False,
                title_style="dim bold",
                box=rich.box.ROUNDED,
                row_styles=["none", "dim"],
            )
            table.add_column(style="green", max_width=19, overflow="fold")
            table.add_column(style="blue", max_width=30, overflow="fold")
            for tup in vallist:
                # table.add_row(*map(rich.markup.escape, tup))
                # pre-calculate the line wrap so the table width is correct
                table.add_row(
                    rich.markup.escape(textwrap.fill(tup[0], 19)),
                    rich.markup.escape(textwrap.fill(tup[1], 30)),
                )
            table_list.append(table)
        with console.capture() as capture:
            console.print(
                rich.columns.Columns(
                    table_list,
                    title=rich.text.Text(title, style="bold italic"),
                )
            )
        return capture.get()

    console.print(f"file: {highlight_path(fopts.fpath)}", highlight=False)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        fileduration = executor.submit(
            ffmpeg.ff_bin.probe,
            "duration",
            probefargs,
            probetype=ffmpeg.ProbeType.FORMAT,
            extraargs="-pretty",
            deep_probe=deep_probe,
        )
        vid_fut = executor.submit(
            make_columns,
            "Video",
            "v",
            (InfoKeys.VSTREAMS, InfoKeys.VSTREAM_TAGS, InfoKeys.DISPOSITIONS),
            probefargs,
            deep_probe,
        )
        aud_fut = executor.submit(
            make_columns,
            "Audio",
            "a",
            (InfoKeys.ASTREAMS, InfoKeys.ASTREAM_TAGS, InfoKeys.DISPOSITIONS),
            probefargs,
            deep_probe,
        )
        sub_fut = executor.submit(
            make_columns,
            "Subtitles",
            "s",
            (InfoKeys.SSTREAMS, InfoKeys.SSTREAM_TAGS, InfoKeys.DISPOSITIONS),
            probesfargs,
            deep_probe,
        )
        if (duration := fileduration.result()) is not None:
            console.print("format_duration=[green]" + duration, highlight=False)
        if vid := vid_fut.result():
            console.out(vid, highlight=False)
        if aud := aud_fut.result():
            console.out(aud, highlight=False)
        if subs := sub_fut.result():
            console.out(subs, highlight=False)


def status_wait(
    fv: encode.EncodeSession, futures: Iterable[concurrent.futures.Future[Any]]
) -> None:
    """Wait on remaining background processes while showing status."""
    for future in concurrent.futures.wait(futures, 0).not_done:
        future.add_done_callback(lambda fut: fv.update_avail.set())
    if not concurrent.futures.wait(futures, 0).not_done and fv.ev.verbosity == 0:
        return
    if fv.ev.verbosity > 0:
        unfinished = fv.statuses
    else:
        unfinished = [
            s for s in fv.statuses if s.status is not encode.StatusThread.Code.FINISHED
        ]

    REFRESH_PER_SEC: Final = 10
    with rich.progress.Progress(
        "[progress.description]{task.description}",
        "•",
        "[green]{task.fields[status]}",
        rich.progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        rich.progress.TimeRemainingColumn(),
        console=console,
        refresh_per_second=REFRESH_PER_SEC,
    ) as progress:
        task_ids = [
            progress.add_task(
                status.name, status=status.long_status, start=False, total=1.0
            )
            for status in unfinished
        ]

        def update_tasks() -> None:
            for task_id, status in zip(task_ids, unfinished):
                fv.update_avail.clear()
                # TODO: 3.10 match case
                if status.status is encode.StatusThread.Code.RUNNING:
                    progress.start_task(task_id)
                    completed = status.progress
                elif status.status is encode.StatusThread.Code.FINISHED:
                    progress.stop_task(task_id)
                    completed = 1.0
                elif status.status is encode.StatusThread.Code.FAILED:
                    progress.stop_task(task_id)
                    completed = 0
                else:
                    progress.reset(task_id, start=False)
                    completed = 0
                progress.update(
                    task_id,
                    status=status.long_status,
                    completed=completed,
                )

        start_time = time.perf_counter()
        sleep_time = 1 / REFRESH_PER_SEC
        update_tasks()
        while concurrent.futures.wait(
            futures, sleep_time - (time.perf_counter() - start_time) % sleep_time
        ).not_done:
            fv.update_avail.wait()
            update_tasks()
        update_tasks()


def setup_pyffserver_stream(fv: encode.EncodeSession) -> None:
    """Communicate with a pyffserver API to set up encode session."""
    payload = {"key": fv.ev.api_key}
    json_payload = {
        "vstandard": fv.ev.vstandard,
        "pixfmt": fv.ev.pix_fmt,
        "astandard": fv.ev.astandard,
        "sample_rate": fv.ev.samplerate,
        "framerate": fv.v("v", "r_frame_rate"),
        "keyframe_int": fv.ev.kf_int,
        "keyframe_sec": fv.ev.kf_sec,
        "inputbitrate": fv.ev.vbitrate,
        "inputabitrate": fv.ev.abitrate,
        "use_timeline": str(fv.ev.use_timeline).lower(),
        "bound_w": fv.ev.bound_w,
        "bound_h": fv.ev.bound_h,
    }
    with requests.Session() as s:
        while True:
            try:
                req = s.get(fv.ev.api_url + "/status", params=payload)
                if req.status_code == 401:
                    console.print("API key rejected by server")
                    raise SystemExit(1)
                elif req.status_code != 200:
                    console.print(
                        f"API server returned error {req.status_code}: {req.reason}"
                    )
                    console.print("waiting and trying again")
                    time.sleep(1)
                    continue
                status = req.text.strip()
            except requests.exceptions.RequestException:
                status = "req_exception"
            # TODO: 3.10 match case
            if status == "ready":
                break
            elif status == "running":
                console.print("Server running, waiting and trying again")
                time.sleep(1)
            elif status == "req_exception":
                console.print(
                    "API request returned exception, waiting and trying again"
                )
                time.sleep(1)
            else:
                console.print("Unknown server status, waiting and trying again")
                time.sleep(1)
        try:
            r = s.post(fv.ev.api_url + "/stream", json=json_payload, params=payload)
        except requests.exceptions.RequestException as e:
            raise ValueError(
                "Server returned unexpected error while initiating stream."
            ) from e
        if r.status_code != 200:
            raise ValueError(f"Server returned error {r.status_code}: {r.reason}.")
        json = r.json()
        fv.ev.srt_passphrase = json.get("srt_passphrase")
        fv.ev.endpoint = json.get("endpoint")
        if not fv.ev.srt_passphrase or not fv.ev.endpoint:
            raise ValueError("server did not return needed parameters")


def start_stream(fv: encode.EncodeSession) -> None:
    """Start and track the actual encode."""
    with subprocess.Popen(
        fv.ev.ff_flags,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=ffmpeg.ff_bin.env,
    ) as result, rich.progress.Progress(
        "[progress.description]{task.description}",
        rich.progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        "[green]{task.fields[timestamp]}",
        "•",
        "[red]{task.fields[bitrate]}",
        "•",
        "[cyan]{task.fields[speed]:<6}",
        refresh_per_second=10 if fv.fv("f", "duration") is None else 4,
        console=console,
    ) as progress:
        assert result.stdout is not None

        if fv.ev.clip_length:
            length = ffmpeg.duration(fv.ev.clip_length)
        else:
            length = ffmpeg.duration(fv.v("f", "duration")) or 1
        if fv.ev.timestamp and not fv.ev.clip_length:
            ts_offset = ffmpeg.duration(fv.ev.timestamp)
        else:
            ts_offset = 0.0

        ts = 0.0
        ts += ts_offset

        def sec_to_str(seconds: float | int) -> str:
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

        length_str = "" if fv.fv("f", "duration") is None else f"/{sec_to_str(length)}"

        task_id = progress.add_task(
            "Encode",
            bitrate=fv.ev.ffprogress.status["bitrate"],
            speed=fv.ev.ffprogress.status["speed"],
            timestamp=f"{sec_to_str(ts)}{length_str}",
            start=False,
        )
        progress.update(task_id, total=length)
        if fv.fv("f", "duration") is not None:
            progress.start_task(task_id)

        fv.ev.ffprogress.monitor_progress(
            result, result.stdout, maxlen=50, loglevel=logging.INFO
        )

        fv.ev.ffprogress.progress_avail.wait()
        while not fv.ev.ffprogress.finished.is_set():
            ts = fv.ev.ffprogress.time_s + ts_offset
            progress.update(
                task_id,
                completed=ts,
                bitrate=fv.ev.ffprogress.status["bitrate"],
                speed=fv.ev.ffprogress.status["speed"],
                timestamp=f"{sec_to_str(ts)}{length_str}",
            )
            fv.ev.ffprogress.progress_avail.wait()
            fv.ev.ffprogress.progress_avail.clear()

        ts = length
        progress.update(
            task_id,
            refresh=True,
            completed=ts,
            bitrate=fv.ev.ffprogress.status["bitrate"],
            speed=fv.ev.ffprogress.status["speed"],
            timestamp=f"{sec_to_str(ts)}{length_str}",
        )

    if result.returncode != 0:
        logger.error("\n".join(fv.ev.ffprogress.output))
        logger.error(f"stream finished with exit code {result.returncode}")
    else:
        console.print("stream finished")


def stream_file(fopts: encode.FileOpts, args: argparse.Namespace) -> None:
    """Manage calculating of all stream parameters."""
    with console.status("calculating stream parameters"):
        console.print(f"starting: {highlight_path(fopts.fpath)}", highlight=False)
        fv = encode.EncodeSession(fopts, encode.StaticEncodeVars.from_args(args))
        futures = []
        encode.determine_timeseek(fv)
        if not fv.ev.copy_audio:
            futures.append(fv.executor.submit(encode.determine_afilters, fv))
        if not fv.ev.copy_video:
            futures.append(fv.executor.submit(encode.determine_vfilters, fv))
        encode.do_framerate_calcs(fv)
        encode.determine_bounds(fv)
        fv.ev.encode_flags = [*encode.get_vflags(fv), *encode.get_aflags(fv)]
    status_wait(fv, futures)
    encode.close_futures(futures)
    fv.executor.shutdown()
    encode.set_input_flags(fv)
    encode.set_filter_flags(fv)
    if fv.ev.wait:
        with rich.live.Live(
            rich.text.Text("Press Enter to continue...", end=""),
            auto_refresh=False,
            transient=True,
            console=console,
        ):
            console.input()

    if fv.ev.pyffserver and not fv.ev.outfile:
        with console.status("connecting to server"):
            setup_pyffserver_stream(fv)
    encode.set_output_flags(fv)
    encode.set_ffmpeg_flags(fv)
    start_stream(fv)


def process_file(
    fpath: pathlib.Path, args: argparse.Namespace, stream_flist: Sequence[pathlib.Path]
) -> None:
    """Format input arguments needed for a file and send to output."""
    infile_args = []
    if args.playlist:
        infile_args += ["-f", "concat", "-safe", "0"]
    file_string = str(fpath)
    if args.bluray:
        file_string = f"bluray:{file_string}"
    infile_args += ["-i", file_string]
    if args.subfile is None:
        insubfile_args, sfpath = infile_args, fpath
    else:
        insubfile_args, sfpath = ["-i", str(args.subfile)], args.subfile
    fopts = encode.FileOpts(infile_args, insubfile_args, fpath, sfpath, stream_flist)
    if args.print_info:
        print_info(fopts, args.deep_probe)
    else:
        stream_file(fopts, args)


def parse_files(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Process input arguments and send them off processing."""
    if args.bluray:
        if args.playlist:
            parser.error("--bluray cannot be used with a playlist")
        elif len(args.files) == 1 and args.files[0].is_dir():
            process_file(args.files[0], args, [args.files[0]])
        else:
            parser.error("--bluray must be used with a directory")
    elif args.files:
        stream_flist = []
        for path in args.files:
            if path.is_dir():
                stream_flist += [f for f in sorted(path.iterdir()) if f.is_file()]
            elif path.is_file():
                stream_flist += [path]
            else:
                parser.error("input files must be files or directories")
        if not stream_flist:
            parser.error("must supply a directory with at least one file")
        if args.files[-1].is_file():
            not_same = lambda f: not f.samefile(stream_flist[-1])  # noqa: E731
            stream_flist = (
                list(itertools.dropwhile(not_same, stream_flist))[:-1] or stream_flist
            )
        if args.playlist:
            with console.status("constructing playlist..."):
                add_duration = bool(
                    args.timestamp is not None
                    or args.print_info
                    or args.crop
                    or args.subs
                )
                playlist = ffmpeg.ff_bin.make_playlist(
                    stream_flist, args.tempdir, add_duration, args.deep_probe
                )
            process_file(playlist, args, stream_flist)
        else:
            for i, path in enumerate(stream_flist):
                process_file(path, args, stream_flist)
                if i < len(stream_flist) - 1:
                    console.print()
                    console.print()
    elif args.obs:
        pass


@dataclasses.dataclass
class DefaultConfig:
    """Holds config file values.

    Used to determine the default CLI parameters after processing config
    from files.
    """

    pyffserver: bool = encode.StaticEncodeVars.pyffserver
    protocol: str = encode.StaticEncodeVars.protocol
    vbitrate: str = encode.StaticEncodeVars.vbitrate
    abitrate: str = encode.StaticEncodeVars.abitrate
    astandard: str = encode.StaticEncodeVars.astandard
    vencoder: str = encode.StaticEncodeVars.vencoder
    endpoint: str = encode.StaticEncodeVars.endpoint
    api_url: str = encode.StaticEncodeVars.api_url
    api_key: str = encode.StaticEncodeVars.api_key
    soxr: bool = encode.StaticEncodeVars.soxr
    preset: str = encode.StaticEncodeVars.x264_preset
    zscale: bool = encode.StaticEncodeVars.zscale
    vulkan: bool = encode.StaticEncodeVars.vulkan
    fdk: bool = encode.StaticEncodeVars.fdk
    height: int = int(encode.StaticEncodeVars.target_h)
    shader_dir: pathlib.Path = encode.StaticEncodeVars.shader_dir
    shader_list: list[str] = dataclasses.field(default_factory=list)
    kf_target_sec: float = encode.StaticEncodeVars.kf_target_sec
    ffmpeg_bin: str = ffmpeg.ff_bin.ffmpeg
    ffprobe_bin: str = ffmpeg.ff_bin.ffprobe
    env: dict[str, str] = dataclasses.field(
        default_factory=lambda: copy.deepcopy(ffmpeg.ff_bin.env)
    )


def set_console_logger(verbosity: int) -> None:
    """Set loglevel."""
    root_logger = logging.getLogger("")
    if verbosity >= 2:
        root_logger.setLevel(logging.DEBUG)
    elif verbosity >= 1:
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.WARNING)

    rhandler = rich.logging.RichHandler(
        console=console, show_time=False, show_path=False
    )
    que: queue.Queue[Any] = queue.Queue()
    queue_handler = logging.handlers.QueueHandler(que)
    listener = logging.handlers.QueueListener(que, rhandler)
    root_logger.addHandler(queue_handler)
    listener.start()
    atexit.register(lambda x: x.stop(), listener)


def download_win_ffmpeg(dltype: str = "git") -> bool:
    """Download and install ffmpeg for windows in user_data_path.

    The current ffmpeg in the location is replaced if already there.
    User data path is determined from platformdirs.
    """
    console.print(
        f"Starting download of ffmpeg {dltype} from"
        " https://github.com/BtbN/FFmpeg-Builds/releases"
    )
    with tempfile.TemporaryDirectory(
        prefix=f"{APPNAME}-"
    ) as tempdir, requests.Session() as s:
        download = pathlib.Path(tempdir) / "ffmpeg.zip"
        try:
            r = s.get("https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest")
            if r.status_code != 200:
                logger.error(
                    f"Fetch failed, server returned error {r.status_code}: {r.reason}"
                )
                return False
        except requests.exceptions.RequestException:
            logger.error("Fetch failed, requests could not communicate with server.")
            return False

        assets = r.json().get("assets", {})
        if dltype == "git":
            download_regex = r"(?P<dir>ffmpeg-N-.*win64-gpl)\.zip"
        elif dltype == "stable":
            download_regex = r"(?P<dir>ffmpeg-[n\d\.]+-.*win64-gpl-[\d\.]+)\.zip"
        else:
            raise ValueError("Unrecognized download type requested")
        ff_url = ""
        dir_name = ""
        for ass in assets:
            if (match := re.fullmatch(download_regex, ass.get("name", ""))) is not None:
                dir_name = match.group("dir")
                ff_url = ass["browser_download_url"]
                break
        else:
            logger.error("Could not fetch ffmpeg download URL from github")
            return False
        try:
            with download.open("w+b") as f, rich.progress.Progress(
                rich.progress.TextColumn(
                    "[bold blue]{task.fields[filename]}", justify="right"
                ),
                rich.progress.BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                rich.progress.DownloadColumn(),
                "•",
                rich.progress.TransferSpeedColumn(),
                "•",
                rich.progress.TimeRemainingColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task(
                    "download", filename=ff_url.split("/")[-1], start=False
                )
                response = s.get(
                    ff_url, stream=True, headers={"Cache-Control": "no-cache"}
                )
                progress.update(
                    task_id, total=int(response.headers.get("Content-length", 0))
                )
                progress.start_task(task_id)
                for chunk in response.iter_content(chunk_size=4096):
                    f.write(chunk)
                    progress.update(task_id, advance=len(chunk))
                if response.status_code != 200:
                    logger.error(
                        f"Fetch failed, server returned error {response.status_code}:"
                        f" {response.reason}"
                    )
                    return False
        except requests.exceptions.RequestException:
            logger.error("Fetch failed, requests could not communicate with server.")
            return False
        with console.status("extracting archive..."):
            data_path = platformdirs.user_data_path(APPNAME)
            data_path.mkdir(parents=True, exist_ok=True)
            ffmpeg_dir = data_path / "ffmpeg"
            shutil.unpack_archive(download, tempdir)
            archive_dir = pathlib.Path(tempdir) / dir_name
            if ffmpeg_dir.exists():
                shutil.rmtree(ffmpeg_dir)
            shutil.move(archive_dir, ffmpeg_dir)
    return True


def win_set_local_ffmpeg(dltype: str, env: dict[str, str]) -> None:
    """Set the ffmpeg instance to the app-local Windows copy.

    If ffmpeg is not already available in user_data_path, offer to
    download it from a public repository.
    """
    local_bin = platformdirs.user_data_path(APPNAME) / "ffmpeg/bin"
    if (
        shutil.which(local_bin / "ffmpeg.exe") is None
        or shutil.which(local_bin / "ffprobe.exe") is None
    ):
        if not sys.__stdin__.isatty():
            console.print(f"ffmpeg required to run {APPNAME}")
            raise SystemExit(1)
        console.print(
            "No local ffmpeg found. We can download from"
            " https://github.com/BtbN/FFmpeg-Builds/releases"
        )
        if not rich.prompt.Confirm.ask("Would you like to download?", console=console):
            console.print(f"ffmpeg required to run {APPNAME}")
            raise SystemExit(1)
        if (
            not download_win_ffmpeg(dltype)
            or shutil.which(local_bin / "ffmpeg.exe") is None
            or shutil.which(local_bin / "ffprobe.exe") is None
        ):
            logger.error(
                "ffmpeg download failed, download manually from"
                " https://ffmpeg.org/download.html or try again later"
            )
            raise SystemExit(1)
    ffmpeg.ff_bin = ffmpeg.FFBin(
        ffmpeg=str(local_bin / "ffmpeg.exe"),
        ffprobe=str(local_bin / "ffprobe.exe"),
        env=env,
    )


def get_parserconfig(
    reproducible: bool = True,
) -> tuple[argparse.ArgumentParser, DefaultConfig]:
    """Return parser and config used."""
    config = DefaultConfig()

    conf_list: list[str | os.PathLike[Any]] = []

    if not reproducible:
        conf_list.append(platformdirs.site_config_path(APPNAME) / f"{APPNAME}.conf")
        conf_list.append(platformdirs.user_config_path(APPNAME) / f"{APPNAME}.conf")

    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument(
        "--config", help="Path to config file", type=pathlib.Path, metavar="FILE"
    )
    conf_parser.add_argument(
        "--show-config-dirs",
        help="Print out config search locations",
        action="store_true",
    )
    conf_args, _ = conf_parser.parse_known_args()
    conf_file = conf_args.config

    if conf_args.show_config_dirs:
        console.print("Searched config files in this order:", highlight=False)
        for confile in conf_list:
            console.print(confile)
        raise SystemExit(0)

    if conf_file is not None and conf_file.is_file():
        conf_list.append(conf_file)

    converters = {
        "list": lambda val: [
            line.lstrip() for line in val.splitlines() if line.strip()
        ],
        "path": lambda val: pathlib.Path(val),  # pylint: disable=W0108
    }
    parsed_config = configparser.ConfigParser(converters=converters)
    read_configs = parsed_config.read(conf_list, encoding="utf-8")

    if "pyffstream" in parsed_config.sections():
        main_conf = parsed_config["pyffstream"]
        config.pyffserver = main_conf.getboolean("pyffserver", config.pyffserver)
        config.protocol = main_conf.get("protocol", config.protocol)
        config.vbitrate = main_conf.get("vbitrate", config.vbitrate)
        config.abitrate = main_conf.get("abitrate", config.abitrate)
        config.astandard = main_conf.get("astandard", config.astandard)
        config.vencoder = main_conf.get("vencoder", config.vencoder)
        config.preset = main_conf.get("preset", config.preset)
        config.endpoint = main_conf.get("endpoint", config.endpoint)
        config.api_url = main_conf.get("api_url", config.api_url)
        config.api_key = main_conf.get("api_key", config.api_key)
        config.soxr = main_conf.getboolean("soxr", config.soxr)
        config.zscale = main_conf.getboolean("zscale", config.zscale)
        config.vulkan = main_conf.getboolean("vulkan", config.vulkan)
        config.fdk = main_conf.getboolean("fdk", config.fdk)
        config.height = main_conf.getint("height", config.height)
        config.shader_dir = main_conf.getpath("shader_dir", config.shader_dir)
        config.shader_list = main_conf.getlist("shader_list", config.shader_list)
        config.kf_target_sec = main_conf.getfloat("kf_target_sec", config.kf_target_sec)
        config.ffmpeg_bin = main_conf.get("ffmpeg_bin", config.ffmpeg_bin)
        config.ffprobe_bin = main_conf.get("ffprobe_bin", config.ffprobe_bin)

    if "env" in parsed_config.sections():
        # use case-sensitive keys for env values
        env_config = configparser.ConfigParser()
        # https://github.com/python/mypy/issues/2427
        env_config.optionxform = str  # type: ignore
        env_config.read(conf_list, encoding="utf-8")
        for key in env_config["env"]:
            config.env[key] = env_config["env"][key]

    parser = argparse.ArgumentParser(
        description="CLI frontend for streaming over SRT and RTMP.",
        parents=[conf_parser],
    )

    def int_ge_zero(value: str) -> int:
        ivalue = int(value.strip())
        if ivalue < 0:
            raise argparse.ArgumentTypeError(
                f"{value.strip()!r} not an integer greater than or equal to 0"
            )
        return ivalue

    def int_gt_zero(value: str) -> int:
        ivalue = int(value.strip())
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                f"{value.strip()!r} not an integer greater than 0"
            )
        return ivalue

    def float_gt_pointfive(value: str) -> float:
        fvalue = float(value.strip())
        if fvalue <= 0.5:
            raise argparse.ArgumentTypeError(
                f"{value.strip()!r} not a float greater than 0.5"
            )
        return fvalue

    def ffmpeg_number(value: str) -> str:
        ffmpeg.num(value)
        return value

    def ffmpeg_duration(value: str) -> str:
        ffmpeg.duration(value)
        return value

    def ffmpeg_astandard(value: str) -> str:
        lvalue = value.lower()
        if lvalue not in encode.StaticEncodeVars.AUDIO_STANDARDS:
            raise argparse.ArgumentTypeError(f"unsupported astandard value: {value!r}")
        return lvalue

    def ffmpeg_protocol(value: str) -> str:
        lvalue = value.lower()
        if lvalue not in encode.StaticEncodeVars.STREAM_PROTOCOLS:
            raise argparse.ArgumentTypeError(
                f"unsupported stream protocol value: {value!r}"
            )
        return lvalue

    def ffmpeg_preset(value: str) -> str:
        lvalue = value.lower()
        if lvalue not in encode.StaticEncodeVars.ALLOWED_PRESETS:
            raise argparse.ArgumentTypeError(f"unsupported preset value: {value!r}")
        return lvalue

    def ffmpeg_vencoder(value: str) -> str:
        lvalue = value.lower()
        if lvalue not in encode.StaticEncodeVars.VIDEO_ENCODERS:
            raise argparse.ArgumentTypeError(f"unsupported vencoder value: {value!r}")
        return lvalue

    input_parser = parser.add_argument_group("input arguments")
    video_parser = parser.add_argument_group("video arguments")
    audio_parser = parser.add_argument_group("audio arguments")
    subtitle_parser = parser.add_argument_group("subtitle arguments")
    output_parser = parser.add_argument_group("output arguments")

    input_group = input_parser.add_mutually_exclusive_group()
    ff_binary_group = parser.add_mutually_exclusive_group()
    aencoder_group = audio_parser.add_mutually_exclusive_group()
    vencoder_group = video_parser.add_mutually_exclusive_group()
    decimate_group = input_parser.add_mutually_exclusive_group()
    anorm_group = audio_parser.add_mutually_exclusive_group()
    resolution_group = video_parser.add_mutually_exclusive_group()

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase verbosity level",
    )
    video_parser.add_argument(
        "-b",
        "--vbitrate",
        type=ffmpeg_number,
        help="encoding video bitrate (ffmpeg num) (default: %(default)s)",
        default=config.vbitrate,
        metavar="BITRATE",
    )
    audio_parser.add_argument(
        "-A",
        "--abitrate",
        type=ffmpeg_number,
        help="encoding audio bitrate (ffmpeg num) (default: %(default)s)",
        default=config.abitrate,
        metavar="BITRATE",
    )
    aencoder_group.add_argument(
        "--astandard",
        type=ffmpeg_astandard,
        help="audio encoding standard to use (default: %(default)s)",
        default=config.astandard,
        choices=sorted(encode.StaticEncodeVars.AUDIO_STANDARDS),
    )
    input_group.add_argument(
        "files",
        type=pathlib.Path,
        nargs="*",
        default=[],
        metavar="INPUT_FILES",
        help=(
            "list of input files and directories; if last argument is file already"
            " contained in input list, start list from that file"
        ),
    )
    input_group.add_argument(
        "-o", "--obs", action="store_true", help="get input from OBS pipe"
    )
    parser.add_argument(
        "-p",
        "--playlist",
        help="make ffconcat playlist from input files",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--print-info",
        help="print information about input file(s) instead of streaming",
        action="store_true",
    )
    parser.add_argument(
        "-t", "--timestamp", type=ffmpeg_duration, help="timestamp to start stream from"
    )
    output_parser.add_argument(
        "--pyffserver",
        help="use pyffserver as an API to send to",
        action=argparse.BooleanOptionalAction,
        default=config.pyffserver,
    )
    output_parser.add_argument(
        "--srt-passphrase",
        help="optional passphrase to use for SRT when not streaming to a pyffserver",
        type=str,
        default="",
        metavar="PASSWORD",
    )
    aencoder_group.add_argument(
        "--protocol",
        type=ffmpeg_protocol,
        help="streaming protocol to use (default: %(default)s)",
        default=config.protocol,
        choices=sorted(encode.StaticEncodeVars.STREAM_PROTOCOLS),
    )
    output_parser.add_argument(
        "-U",
        "--api-url",
        help="pyffserver API URL to use (default: from config)",
        type=str,
        default=config.api_url,
        metavar="URL",
    )
    output_parser.add_argument(
        "-k",
        "--api-key",
        help="pyffserver API key to use (default: from config)",
        type=str,
        default=config.api_key,
        metavar="KEY",
    )
    output_parser.add_argument(
        "-E",
        "--endpoint",
        help="endpoint to stream to without protocol (default: from config))",
        type=str,
        default=config.endpoint,
        metavar="DOMAIN[:PORT][PATH]",
    )
    parser.add_argument(
        "-w",
        "--wait",
        help="wait for keypress before starting stream",
        action="store_true",
    )
    input_parser.add_argument(
        "-B", "--bluray", help="input directory is bluray", action="store_true"
    )
    subtitle_parser.add_argument(
        "-e", "--subs", help="enable subtitles", action="store_true"
    )
    subtitle_parser.add_argument(
        "--subfile",
        type=pathlib.Path,
        help="path to external subtitles",
        metavar="FILE",
    )
    subtitle_parser.add_argument(
        "-s",
        "--sindex",
        help="subindex of subtitle stream to use",
        type=int_ge_zero,
        metavar="N",
    )
    audio_parser.add_argument(
        "-a",
        "--aindex",
        help="subindex of audio stream to use (default: %(default)s)",
        type=int_ge_zero,
        default=encode.StaticEncodeVars.aindex,
        metavar="N",
    )
    video_parser.add_argument(
        "--vindex",
        help="subindex of video stream to use (default: %(default)s)",
        type=int_ge_zero,
        default=encode.StaticEncodeVars.vindex,
        metavar="N",
    )
    input_parser.add_argument(
        "--live", help="hint that input is live", action="store_true"
    )
    parser.add_argument(
        "--fix-start-time",
        help="Fix start_time of streams",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    audio_parser.add_argument(
        "--soxr",
        help="Use SoX resampler library instead of ffmpeg's avresample",
        action=argparse.BooleanOptionalAction,
        default=config.soxr,
    )
    output_parser.add_argument(
        "--fifo",
        help="Use FIFO to try to sustain and stabilize the connection.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    video_parser.add_argument(
        "--zscale",
        help="Use zimg library for scaling instead of ffmpeg's scale",
        action=argparse.BooleanOptionalAction,
        default=config.zscale,
    )
    audio_parser.add_argument(
        "--fdk",
        help="Use libfdk instead of ffmpeg's aac encoder",
        action=argparse.BooleanOptionalAction,
        default=config.fdk,
    )
    output_parser.add_argument(
        "-f",
        "--outfile",
        type=pathlib.Path,
        help="path to an output file to use instead of streaming",
        metavar="FILE",
    )
    output_parser.add_argument(
        "--cliplength",
        type=ffmpeg_duration,
        help="clip stream to this length",
        metavar="LENGTH",
    )
    input_parser.add_argument(
        "--deep-probe",
        help="pass extra args to probe input file deeper",
        action="store_true",
    )
    aencoder_group.add_argument(
        "--copy-audio", help="copy audio stream from input", action="store_true"
    )
    vencoder_group.add_argument(
        "--copy-video", help="copy video stream from input", action="store_true"
    )
    parser.add_argument(
        "-c", "--copy", help="pass a/v to copy audio/video", type=str, action="append"
    )
    vencoder_group.add_argument(
        "-H", "--hevc-nvenc", help="encode with NVENC HEVC", action="store_true"
    )
    video_parser.add_argument(
        "-8",
        "--eightbit",
        help="encode with 8-bit NVENC HEVC (default 10-bit)",
        action="store_true",
    )
    vencoder_group.add_argument(
        "--h264-nvenc", help="encode with NVENC H264", action="store_true"
    )
    vencoder_group.add_argument(
        "-x", "--x264", help="encode with x264", action="store_true"
    )
    video_parser.add_argument(
        "--preset",
        help="preset to use for x264 encoding (default: %(default)s)",
        type=ffmpeg_preset,
        default=config.preset,
        choices=encode.StaticEncodeVars.ALLOWED_PRESETS,
    )
    vencoder_group.add_argument(
        "--vencoder",
        type=ffmpeg_vencoder,
        help="video encoder to use (default: %(default)s)",
        default=config.vencoder,
        choices=sorted(encode.StaticEncodeVars.VIDEO_ENCODERS),
    )
    video_parser.add_argument(
        "-u",
        "--upscale",
        help="unconditionally scale video to target size",
        action="store_true",
    )
    parser.add_argument(
        "--slowseek", help="use slow ffmpeg seeking", action="store_true"
    )
    video_parser.add_argument(
        "--deinterlace", help="deinterlace video", action="store_true"
    )
    video_parser.add_argument(
        "-C", "--crop", help="automatically crop video", action="store_true"
    )
    video_parser.add_argument(
        "-V",
        "--vulkan",
        help="use vulkan processing path",
        action=argparse.BooleanOptionalAction,
        default=config.vulkan,
    )
    subtitle_parser.add_argument(
        "-z",
        "--cropsecond",
        help="crop after subtitles are rendered",
        action="store_true",
    )
    decimate_group.add_argument(
        "--nodecimate",
        help="don't decimate 30 fps obs input to 24 fps",
        action="store_true",
    )
    decimate_group.add_argument(
        "--paldecimate", help="decimate 30 fps obs input to 25 fps", action="store_true"
    )
    decimate_group.add_argument(
        "--sixtyfps", help="don't halve 60 fps obs input to 30 fps", action="store_true"
    )
    audio_parser.add_argument(
        "-n",
        "--anormalize",
        help="normalize audio (implied by -N and -Q)",
        action="store_true",
    )
    anorm_group.add_argument(
        "-N",
        "--normfile",
        type=pathlib.Path,
        help="path to file to store audio normalization data",
        metavar="FILE",
    )
    anorm_group.add_argument(
        "-Q",
        "--dynamicnorm",
        help="do one-pass audio normalization",
        action="store_true",
    )
    resolution_group.add_argument(
        "--height",
        type=int_gt_zero,
        help="target 16:9 bounding box encode height (default: %(default)s)",
        default=config.height,
    )
    resolution_group.add_argument(
        "-4", "--res2160", help="set 4k encoding resolution", action="store_true"
    )
    resolution_group.add_argument(
        "-2", "--res1440", help="set 1440p encoding resolution", action="store_true"
    )
    resolution_group.add_argument(
        "-7", "--res720", help="set 720p encoding resolution", action="store_true"
    )
    output_parser.add_argument(
        "--keyframe-target-sec",
        type=float_gt_pointfive,
        help="target keyframe interval in seconds (default: %(default)s)",
        default=config.kf_target_sec,
        metavar="SEC",
    )
    audio_parser.add_argument(
        "--mono", help="output audio in mono", action="store_true"
    )
    parser.add_argument(
        "--startdelay", help="delay stream start by 30 seconds", action="store_true"
    )
    parser.add_argument(
        "--tempdir",
        type=pathlib.Path,
        help="directory to use for storing temporary files",
        metavar="DIR",
    )
    parser.add_argument(
        "--shaderdir",
        type=pathlib.Path,
        help="directory for libplacebo shaders",
        default=config.shader_dir,
        metavar="DIR",
    )
    parser.add_argument(
        "--shaders",
        type=str,
        help=(
            "list of shaders to use (specify once for each shader to add) (default:"
            " %(default)s)"
        ),
        default=config.shader_list,
        action="append",
        metavar="SHADER",
    )
    ff_binary_group.add_argument(
        "--system-ffmpeg",
        help=(
            "use system ffmpeg binaries instead of configured (default is system if"
            " unconfigured)"
        ),
        action="store_true",
    )
    if platform.system() == "Windows" or reproducible:
        ff_binary_group.add_argument(
            "--downloaded-ffmpeg",
            help=(
                "Use downloaded local Windows ffmpeg instead of configured or system"
                " ffmpeg (default is to only use as a fallback)"
            ),
            action="store_true",
        )
        parser.add_argument(
            "--redownload",
            help="Redownload stored local Windows ffmpeg binaries",
            action="store_true",
        )
        parser.add_argument(
            "--dltype",
            help="Type of Windows ffmpeg binary to download (default: %(default)s)",
            type=str.lower,
            default="git",
            choices=["git", "stable"],
        )
    parser.add_argument(
        "--write",
        help="write chosen arguments as defaults to config if not already default",
        action="store_true",
    )

    if conf_file is not None and not conf_file.is_file():
        parser.error("Passed config file must be a file.")

    logger.info(f"parsed configs: {read_configs!r}")

    return parser, config


def main() -> None:
    """Process config and CLI arguments then send off for processing."""
    parser, config = get_parserconfig(False)
    args = parser.parse_args()

    set_console_logger(args.verbose)

    if not args.system_ffmpeg:
        ffmpeg.ff_bin = ffmpeg.FFBin(
            ffmpeg=config.ffmpeg_bin, ffprobe=config.ffprobe_bin, env=config.env
        )

    if platform.system() == "Windows":
        if args.redownload:
            download_win_ffmpeg(args.dltype)
            raise SystemExit(0)
        if args.downloaded_ffmpeg:
            win_set_local_ffmpeg(args.dltype, config.env)

    if (
        shutil.which(ffmpeg.ff_bin.ffmpeg) is None
        or shutil.which(ffmpeg.ff_bin.ffprobe) is None
    ):
        if platform.system() == "Windows":
            win_set_local_ffmpeg(args.dltype, config.env)
        else:
            console.print("Cannot find ffmpeg and ffprobe utilities in path")
            console.print(
                "Consider downloading ffmpeg from your package manager or at"
                " https://ffmpeg.org/download.html"
            )
            raise SystemExit(1)

    required_args = [args.files, args.obs, args.write]
    if sum(bool(i) for i in required_args) != 1:
        if args.write:
            parser.error(
                "--write cannot be used with an output argument or other action"
            )
        parser.error("Must specify at least one output argument")

    if args.hevc_nvenc:
        args.vencoder = "hevc_nvenc"
    elif args.h264_nvenc:
        args.vencoder = "h264_nvenc"
    elif args.x264:
        args.vencoder = "libx264"

    if not args.print_info:
        if args.vencoder not in (
            encode.StaticEncodeVars.VIDEO_ENCODERS & ffmpeg.ff_bin.vencoders
        ):
            parser.error(
                f"selected vencoder {args.vencoder!r} not supported by ffmpeg"
                " installation"
            )

        if args.soxr and "--enable-libsoxr" not in ffmpeg.ff_bin.build_config:
            parser.error("soxr specified, but using an ffmpeg build without support")

        if args.zscale and "zscale" not in ffmpeg.ff_bin.filters:
            parser.error("zscale specified, but using an ffmpeg build without support")

        if args.fdk and "libfdk_aac" not in ffmpeg.ff_bin.aencoders:
            parser.error(
                "fdk encoder specified, but using an ffmpeg build without support"
            )

    if args.obs:
        args.live = True

    if args.copy is not None:
        if next(
            (i for i in args.copy if not re.search("^([vV]?[aA]?|[aA]?[vV]?)$", i)),
            None,
        ):
            parser.error(
                "--copy must only contain at most one of the characters a and v each"
            )
        if next((i for i in args.copy if re.search("[vV]", i)), None):
            args.copy_video = True
        if next((i for i in args.copy if re.search("[aA]", i)), None):
            args.copy_audio = True

    if args.startdelay and (args.copy_video or args.copy_audio):
        parser.error("audio/video copying cannot be used with a start delay")
    if args.startdelay and args.timestamp is not None:
        parser.error("timestamp seeking cannot be used with a start delay")

    if args.eightbit and not args.hevc_nvenc:
        parser.error("8-bit HEVC encoding must be specified with HEVC")

    if args.eightbit:
        logger.warning(
            "8-bit NVENC encoding is vastly inferior to 10-bit. Consider using H264"
            " instead if you require 8-bit."
        )

    if args.res2160:
        args.height = 2160
    if args.res1440:
        args.height = 1440
    if args.res720:
        args.height = 720

    if args.sindex is not None:
        args.subs = True
    elif args.subs:
        args.sindex = 0
    elif args.subfile is not None:
        args.subs = True
        args.sindex = 0

    if args.copy_video and args.subs:
        parser.error(
            "video cannot be copied while subs are enabled because they can't be"
            " rendered"
        )

    if args.crop and args.live:
        parser.error("cannot autocrop live input")

    if args.normfile is not None or args.dynamicnorm:
        args.anormalize = True

    if args.normfile is not None:
        if args.normfile.exists() and not args.normfile.is_file():
            parser.error("passed normfile must be a file")

        if not args.normfile.exists() and args.normfile.suffix != ".json":
            parser.error("a new normfile must be written to a .json file")

    if (args.nodecimate or args.paldecimate or args.sixtyfps) and not args.obs:
        parser.error("obs fps arguments require using obs input")

    if args.sixtyfps:
        args.nodecimate = True

    if args.timestamp and args.live:
        parser.error("timestamp seeking can't be used with live input")

    if args.outfile is not None and args.outfile.exists():
        parser.error("output file can't already exist")

    if args.playlist and not args.files:
        parser.error("--playlist requires input files")
    if args.print_info and not args.files:
        parser.error("--print-info requires input files")

    if args.subfile is not None and not args.subfile.is_file():
        parser.error("input subfile must be a file")

    if args.srt_passphrase and not args.pyffserver:
        parser.error("srt passphrase can only be set when not using pyffserver")

    if args.protocol != "srt" and args.pyffserver:
        parser.error("the srt protocol must be used when streaming to a pyffserver")

    if args.write:
        config_dir = platformdirs.user_config_path(APPNAME)
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{APPNAME}.conf"
        new_config = configparser.ConfigParser()
        new_config.read(config_path, encoding="utf-8")
        # keep case for env section
        new_full_config = configparser.ConfigParser()
        new_full_config.optionxform = str  # type: ignore
        new_full_config.read(config_path, encoding="utf-8")

        if "pyffstream" not in new_config.sections():
            new_config.add_section("pyffstream")
        main_section = new_config["pyffstream"]

        class ConfName(NamedTuple):
            file_name: str
            arg_name: str

        conf_names: set[ConfName] = {
            ConfName("pyffserver", "pyffserver"),
            ConfName("protocol", "protocol"),
            ConfName("vbitrate", "vbitrate"),
            ConfName("abitrate", "abitrate"),
            ConfName("astandard", "astandard"),
            ConfName("vencoder", "vencoder"),
            ConfName("preset", "preset"),
            ConfName("endpoint", "endpoint"),
            ConfName("api_url", "api_url"),
            ConfName("api_key", "api_key"),
            ConfName("soxr", "soxr"),
            ConfName("zscale", "zscale"),
            ConfName("vulkan", "vulkan"),
            ConfName("fdk", "fdk"),
            ConfName("height", "height"),
            ConfName("shader_dir", "shaderdir"),
            ConfName("kf_target_sec", "keyframe_target_sec"),
        }
        for conf in conf_names:
            if getattr(config, conf.file_name) != getattr(
                args, conf.arg_name, getattr(config, conf.file_name)
            ):
                main_section[conf.file_name] = str(getattr(args, conf.arg_name))
        new_full_config["pyffstream"] = new_config["pyffstream"]

        console.print("Writing defaults to config path:")
        console.print(config_path)
        with config_path.open(mode="w", encoding="utf-8") as f:
            new_full_config.write(f)
        raise SystemExit(0)

    with tempfile.TemporaryDirectory(prefix=f"{APPNAME}-") as tempdir:
        if args.tempdir is None:
            args.tempdir = pathlib.Path(tempdir)
        try:
            parse_files(args, parser)
        except KeyboardInterrupt:
            console.print("\nquit requested")
