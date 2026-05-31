"""Download/manage structural-eval source data under the scratch dataset root."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_ROOT = Path("/gpfs/scratch/acw749/datasets/structure")
SALAMI_URL = "https://github.com/DDMAL/salami-data-public/archive/refs/heads/master.zip"
HARMONIX_URL = "https://github.com/urinieto/harmonixset/archive/refs/heads/main.zip"
MATCHING_SALAMI_URL = "https://github.com/jblsmith/matching-salami/archive/refs/heads/main.zip"


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=120) as response:
        with path.open("wb") as f:
            shutil.copyfileobj(response, f)


def _extract_zip(zip_path: Path, dest: Path, strip_components: int = 1) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            parts = Path(member.filename).parts[strip_components:]
            if not parts:
                continue
            out = dest.joinpath(*parts)
            if member.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _refresh_archive(url: str, dest: Path, cache_name: str, force: bool = False) -> None:
    marker = dest / ".download_complete"
    if marker.exists() and not force:
        return
    if dest.exists() and force:
        shutil.rmtree(dest)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / cache_name
        _download(url, zip_path)
        _extract_zip(zip_path, dest)
    marker.write_text(url + "\n", encoding="utf-8")


def ensure_sources(root: Path, force: bool = False) -> None:
    _refresh_archive(SALAMI_URL, root / "annotations" / "salami", "salami.zip", force=force)
    _refresh_archive(HARMONIX_URL, root / "annotations" / "harmonix", "harmonix.zip", force=force)
    _refresh_archive(MATCHING_SALAMI_URL, root / "sources" / "matching-salami", "matching-salami.zip", force=force)


def _first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    lowered = {k.lower().strip(): v for k, v in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value:
            return value.strip()
    return ""


def _youtube_url(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"https://www.youtube.com/watch?v={value}"


def _read_salami_pairings(pairings_csv: Path) -> list[dict[str, str]]:
    with pairings_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(f, dialect=dialect)
        rows = []
        for row in reader:
            salami_id = _first_present(row, ("salami_id", "SALAMI_ID", "song_id", "id"))
            youtube = _first_present(row, ("youtube_url", "url", "YouTube", "youtube_id", "video_id"))
            if not salami_id or not youtube:
                values = [v.strip() for v in row.values() if v and v.strip()]
                salami_id = salami_id or (values[0] if values else "")
                youtube = youtube or next((v for v in values if "youtu" in v or len(v) == 11), "")
            if salami_id and youtube:
                rows.append({
                    "salami_id": salami_id,
                    "youtube_url": _youtube_url(youtube),
                    "salami_length": _first_present(row, ("salami_length", "duration")),
                    "onset_in_youtube": _first_present(row, ("onset_in_youtube", "youtube_onset")),
                    "onset_in_salami": _first_present(row, ("onset_in_salami", "salami_onset")),
                })
        return rows


def _read_harmonix_scores(scores_csv: Path) -> dict[str, float]:
    if not scores_csv.exists():
        return {}
    with scores_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return {
            row["File"].strip(): float(row["score"])
            for row in csv.DictReader(f)
            if row.get("File") and row.get("score")
        }


def _read_harmonix_durations(metadata_csv: Path) -> dict[str, float]:
    if not metadata_csv.exists():
        return {}
    with metadata_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return {
            row["File"].strip(): float(row["Duration"])
            for row in csv.DictReader(f)
            if row.get("File") and row.get("Duration")
        }


def _read_harmonix_urls(root: Path, min_score: float | None = None) -> list[dict[str, str]]:
    dataset_dir = root / "annotations" / "harmonix" / "dataset"
    scores = _read_harmonix_scores(dataset_dir / "youtube_alignment_scores.csv")
    durations = _read_harmonix_durations(dataset_dir / "metadata.tsv")
    rows = []
    with (dataset_dir / "youtube_urls.csv").open("r", encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.DictReader(f):
            file_id = (row.get("File") or "").strip()
            url = (row.get("URL") or "").strip()
            if not file_id or not url:
                continue
            score = scores.get(file_id)
            if min_score is not None and score is not None and score < min_score:
                continue
            rows.append({
                "file_id": file_id,
                "youtube_url": url,
                "score": "" if score is None else str(score),
                "duration": "" if file_id not in durations else str(durations[file_id]),
            })
    return rows


def _ffmpeg_path() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _align_salami_audio(raw_wav: Path, final_wav: Path, row: dict[str, str], ffmpeg: str | None) -> None:
    if ffmpeg is None:
        shutil.copy2(raw_wav, final_wav)
        return

    salami_length = float(row.get("salami_length") or 0.0)
    onset_youtube = float(row.get("onset_in_youtube") or 0.0)
    onset_salami = float(row.get("onset_in_salami") or 0.0)
    trim_start = max(onset_youtube - onset_salami, 0.0)
    pad_start = max(onset_salami - onset_youtube, 0.0)
    tmp_wav = final_wav.with_suffix(".tmp.wav")
    final_wav.parent.mkdir(parents=True, exist_ok=True)
    if tmp_wav.exists():
        tmp_wav.unlink()

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if trim_start > 0:
        cmd.extend(["-ss", f"{trim_start:.6f}"])
    cmd.extend(["-i", str(raw_wav)])
    if pad_start > 0:
        delay_ms = int(round(pad_start * 1000.0))
        filters = [f"adelay={delay_ms}:all=1", "apad"]
        if salami_length > 0:
            filters.append(f"atrim=duration={salami_length:.6f}")
        cmd.extend(["-af", ",".join(filters)])
    if salami_length > 0 and pad_start == 0:
        cmd.extend(["-t", f"{salami_length:.6f}"])
    cmd.extend(["-ac", "1", "-ar", "44100", str(tmp_wav)])
    subprocess.run(cmd, check=True)
    tmp_wav.replace(final_wav)


def _trim_harmonix_audio(raw_wav: Path, final_wav: Path, row: dict[str, str], ffmpeg: str | None) -> None:
    if ffmpeg is None:
        shutil.copy2(raw_wav, final_wav)
        return
    duration = float(row.get("duration") or 0.0)
    tmp_wav = final_wav.with_suffix(".tmp.wav")
    final_wav.parent.mkdir(parents=True, exist_ok=True)
    if tmp_wav.exists():
        tmp_wav.unlink()
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_wav)]
    if duration > 0:
        cmd.extend(["-t", f"{duration:.6f}"])
    cmd.extend(["-ac", "1", "-ar", "44100", str(tmp_wav)])
    subprocess.run(cmd, check=True)
    tmp_wav.replace(final_wav)


def _cookies_path(value: str | None) -> str | None:
    if value:
        return value
    env_value = os.environ.get("COVERX_YOUTUBE_COOKIES")
    return env_value if env_value else None


def download_salami_audio(
    root: Path,
    limit: int | None = None,
    force: bool = False,
    cookies: str | None = None,
) -> None:
    for tools_path in (
        Path(os.environ.get("COVERX_TOOLS_PATH", "")),
        root / "tools" / "coverx_tools",
        Path("/tmp/coverx_tools"),
    ):
        if str(tools_path) and tools_path.exists():
            sys.path.insert(0, str(tools_path))
    try:
        import yt_dlp
    except Exception as exc:
        raise RuntimeError("yt-dlp is required. Install it into /tmp/coverx_tools first.") from exc

    pairings = _read_salami_pairings(root / "sources" / "matching-salami" / "salami_youtube_pairings.csv")
    if limit is not None:
        pairings = pairings[:limit]

    raw_dir = root / "audio_raw" / "salami_youtube"
    audio_dir = root / "audio" / "salami"
    log_dir = root / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = _ffmpeg_path()
    status_path = log_dir / "salami_youtube_downloads.jsonl"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(raw_dir / "%(id)s.%(ext)s"),
        "quiet": False,
        "noplaylist": True,
        "retries": 5,
    }
    if ffmpeg:
        ydl_opts["ffmpeg_location"] = ffmpeg
        ydl_opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }]
    cookiefile = _cookies_path(cookies)
    if cookiefile:
        ydl_opts["cookiefile"] = cookiefile

    with yt_dlp.YoutubeDL(ydl_opts) as ydl, status_path.open("a", encoding="utf-8") as status:
        for row in pairings:
            salami_id = row["salami_id"]
            final_wav = audio_dir / f"{salami_id}.wav"
            if final_wav.exists() and not force:
                continue
            record = {"salami_id": salami_id, "url": row["youtube_url"], "ok": False}
            try:
                info = ydl.extract_info(row["youtube_url"], download=True)
                video_id = info.get("id")
                downloaded_wav = raw_dir / f"{video_id}.wav"
                if downloaded_wav.exists():
                    _align_salami_audio(downloaded_wav, final_wav, row, ffmpeg)
                    record.update({"ok": True, "video_id": video_id, "audio_path": str(final_wav)})
                else:
                    record.update({"error": f"downloaded wav not found for video_id={video_id}"})
            except Exception as exc:
                record.update({"error": str(exc)})
            status.write(json.dumps(record, sort_keys=True) + "\n")
            status.flush()


def download_harmonix_audio(
    root: Path,
    limit: int | None = None,
    force: bool = False,
    min_score: float | None = None,
    cookies: str | None = None,
) -> None:
    for tools_path in (
        Path(os.environ.get("COVERX_TOOLS_PATH", "")),
        root / "tools" / "coverx_tools",
        Path("/tmp/coverx_tools"),
    ):
        if str(tools_path) and tools_path.exists():
            sys.path.insert(0, str(tools_path))
    try:
        import yt_dlp
    except Exception as exc:
        raise RuntimeError("yt-dlp is required. Install it into the CoverX tools path first.") from exc

    rows = _read_harmonix_urls(root, min_score=min_score)
    if limit is not None:
        rows = rows[:limit]

    raw_dir = root / "audio_raw" / "harmonix_youtube"
    audio_dir = root / "audio" / "harmonix"
    log_dir = root / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = _ffmpeg_path()
    status_path = log_dir / "harmonix_youtube_downloads.jsonl"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(raw_dir / "%(id)s.%(ext)s"),
        "quiet": False,
        "noplaylist": True,
        "retries": 5,
    }
    if ffmpeg:
        ydl_opts["ffmpeg_location"] = ffmpeg
        ydl_opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }]
    cookiefile = _cookies_path(cookies)
    if cookiefile:
        ydl_opts["cookiefile"] = cookiefile

    with yt_dlp.YoutubeDL(ydl_opts) as ydl, status_path.open("a", encoding="utf-8") as status:
        for row in rows:
            file_id = row["file_id"]
            final_wav = audio_dir / f"{file_id}.wav"
            if final_wav.exists() and not force:
                continue
            record = {
                "file_id": file_id,
                "url": row["youtube_url"],
                "score": row.get("score", ""),
                "ok": False,
            }
            try:
                info = ydl.extract_info(row["youtube_url"], download=True)
                video_id = info.get("id")
                downloaded_wav = raw_dir / f"{video_id}.wav"
                if downloaded_wav.exists():
                    _trim_harmonix_audio(downloaded_wav, final_wav, row, ffmpeg)
                    record.update({"ok": True, "video_id": video_id, "audio_path": str(final_wav)})
                else:
                    record.update({"error": f"downloaded wav not found for video_id={video_id}"})
            except Exception as exc:
                record.update({"error": str(exc)})
            status.write(json.dumps(record, sort_keys=True) + "\n")
            status.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--download-salami-audio", action="store_true")
    parser.add_argument("--download-harmonix-audio", action="store_true")
    parser.add_argument("--harmonix-min-score", type=float, default=None)
    parser.add_argument("--cookies", default=None, help="Path to a Netscape-format YouTube cookies.txt file")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    ensure_sources(args.root, force=args.force)
    if args.download_salami_audio and not args.sources_only:
        download_salami_audio(args.root, limit=args.limit, force=args.force, cookies=args.cookies)
    if args.download_harmonix_audio and not args.sources_only:
        download_harmonix_audio(
            args.root,
            limit=args.limit,
            force=args.force,
            min_score=args.harmonix_min_score,
            cookies=args.cookies,
        )
    print(f"Prepared structure data under {args.root}")


if __name__ == "__main__":
    main()
