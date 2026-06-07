#!/usr/bin/env python
"""Serve a live MSPF visualization browser while PDFs are being generated."""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MSPF Structure Viewer</title>
  <style>
    :root {
      --bg: #f5f7f8;
      --panel: rgba(255, 255, 255, .72);
      --ink: #171717;
      --muted: #666;
      --line: rgba(18, 24, 27, .14);
      --shadow: 0 18px 60px rgba(20, 24, 28, .12);
      --accent: #2563eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #fbfcfc 0%, var(--bg) 38%, #eef3f2 100%);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    .shell { min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
    header {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto minmax(260px, 460px);
      gap: 14px;
      align-items: center;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.68);
      backdrop-filter: blur(18px) saturate(1.25);
      box-shadow: 0 1px 0 rgba(255,255,255,.75) inset;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    .title { min-width: 0; display: flex; align-items: baseline; gap: 10px; white-space: nowrap; overflow: hidden; }
    .track { font-weight: 650; overflow: hidden; text-overflow: ellipsis; }
    .count, .status { color: var(--muted); font-size: 13px; }
    .nav { display: flex; gap: 8px; }
    button, select {
      height: 34px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,.72);
      backdrop-filter: blur(12px);
      color: var(--ink);
      border-radius: 6px;
      font: inherit;
      font-size: 14px;
    }
    button { min-width: 36px; padding: 0 10px; cursor: pointer; }
    button:hover, select:hover { border-color: #b9b9b1; }
    button:active { transform: translateY(1px); }
    select { width: 100%; padding: 0 10px; }
    main { padding: 14px 16px 16px; min-height: 0; }
    .frame {
      width: 100%;
      height: calc(100vh - 78px);
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      overflow: hidden;
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }
    object { width: 100%; height: 100%; display: block; }
    .empty {
      display: grid;
      place-items: center;
      height: 100%;
      color: var(--muted);
      font-size: 15px;
    }
    @media (max-width: 780px) {
      header { grid-template-columns: 1fr; gap: 10px; }
      .nav button { flex: 1; }
      .frame { height: calc(100vh - 150px); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="title">
        <span class="track" id="trackLabel">Waiting for PDFs</span>
        <span class="count" id="countLabel"></span>
        <span class="status" id="statusLabel"></span>
      </div>
      <div class="nav">
        <button id="prevTrack" title="Previous track">Prev</button>
        <button id="nextTrack" title="Next track">Next</button>
        <button id="refresh" title="Refresh now">Refresh</button>
      </div>
      <select id="variantSelect" aria-label="Variant"></select>
    </header>
    <main>
      <div class="frame" id="frame">
        <object id="pdfView" type="application/pdf"></object>
      </div>
    </main>
  </div>
  <script>
    const POLL_MS = __POLL_MS__;
    let data = {tracks: [], generated_at: 0, pdf_count: 0};
    let trackIndex = 0;
    let variantIndex = 0;
    let selectedPath = "";
    const trackLabel = document.getElementById("trackLabel");
    const countLabel = document.getElementById("countLabel");
    const statusLabel = document.getElementById("statusLabel");
    const variantSelect = document.getElementById("variantSelect");
    const pdfView = document.getElementById("pdfView");
    const frame = document.getElementById("frame");

    function currentTrackKey() {
      const track = data.tracks[trackIndex];
      return track ? `${track.dataset}/${track.track_id}` : "";
    }

    function restoreSelection(oldKey, oldPath) {
      if (!data.tracks.length) {
        trackIndex = 0;
        variantIndex = 0;
        return;
      }
      const nextTrack = data.tracks.findIndex(track => `${track.dataset}/${track.track_id}` === oldKey);
      trackIndex = nextTrack >= 0 ? nextTrack : Math.min(trackIndex, data.tracks.length - 1);
      const variants = data.tracks[trackIndex].variants;
      const nextVariant = variants.findIndex(variant => variant.path === oldPath);
      variantIndex = nextVariant >= 0 ? nextVariant : Math.min(variantIndex, variants.length - 1);
    }

    function render() {
      if (!data.tracks.length) {
        trackLabel.textContent = "Waiting for PDFs";
        countLabel.textContent = "";
        statusLabel.textContent = "scanning...";
        variantSelect.innerHTML = "";
        frame.innerHTML = '<div class="empty">No PDFs found yet. This page will update automatically.</div>';
        return;
      }
      if (!document.getElementById("pdfView")) {
        frame.innerHTML = '<object id="pdfView" type="application/pdf"></object>';
      }
      const view = document.getElementById("pdfView");
      const track = data.tracks[trackIndex];
      const variant = track.variants[variantIndex];
      selectedPath = variant.path;
      trackLabel.textContent = `${track.dataset} / ${track.track_id}`;
      countLabel.textContent = `${trackIndex + 1} / ${data.tracks.length}`;
      statusLabel.textContent = `${data.pdf_count} PDFs`;
      variantSelect.innerHTML = "";
      track.variants.forEach((item, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = item.variant;
        variantSelect.appendChild(option);
      });
      variantSelect.value = String(variantIndex);
      view.data = `${variant.url}?m=${variant.mtime_ns}`;
    }

    async function refreshManifest() {
      const oldKey = currentTrackKey();
      const oldPath = selectedPath;
      const response = await fetch(`/manifest.json?ts=${Date.now()}`, {cache: "no-store"});
      data = await response.json();
      restoreSelection(oldKey, oldPath);
      render();
    }

    document.getElementById("prevTrack").addEventListener("click", () => {
      if (!data.tracks.length) return;
      trackIndex = (trackIndex - 1 + data.tracks.length) % data.tracks.length;
      variantIndex = 0;
      render();
    });
    document.getElementById("nextTrack").addEventListener("click", () => {
      if (!data.tracks.length) return;
      trackIndex = (trackIndex + 1) % data.tracks.length;
      variantIndex = 0;
      render();
    });
    document.getElementById("refresh").addEventListener("click", refreshManifest);
    variantSelect.addEventListener("change", () => {
      variantIndex = Number(variantSelect.value);
      render();
    });
    window.addEventListener("keydown", event => {
      if (event.key === "ArrowLeft") document.getElementById("prevTrack").click();
      if (event.key === "ArrowRight") document.getElementById("nextTrack").click();
    });
    refreshManifest();
    setInterval(refreshManifest, POLL_MS);
  </script>
</body>
</html>
"""


def _variant_label(track_id: str, xp: str, path: Path) -> str:
    stem = path.stem
    prefix = f"track_{track_id}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return f"{xp} / {stem}"


def scan_manifest(root: Path) -> dict:
    tracks: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(root.glob("*/*/*/*.pdf")):
        rel = path.relative_to(root)
        if len(rel.parts) != 4:
            continue
        dataset, track_id, xp, _ = rel.parts
        stat = path.stat()
        item = {
            "xp": xp,
            "variant": _variant_label(track_id, xp, path),
            "path": rel.as_posix(),
            "url": quote(rel.as_posix()),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }
        tracks.setdefault((dataset, track_id), []).append(item)
    return {
        "generated_at": time.time(),
        "pdf_count": sum(len(items) for items in tracks.values()),
        "tracks": [
            {
                "dataset": dataset,
                "track_id": track_id,
                "variants": sorted(items, key=lambda item: (item["xp"], item["variant"])),
            }
            for (dataset, track_id), items in sorted(tracks.items())
        ],
    }


class LiveVizHandler(SimpleHTTPRequestHandler):
    root: Path
    poll_ms: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.root), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_html()
            return
        if parsed.path == "/manifest.json":
            self._send_manifest()
            return
        self.path = unquote(parsed.path)
        return super().do_GET()

    def _send_html(self) -> None:
        body = INDEX_HTML.replace("__POLL_MS__", str(self.poll_ms)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_manifest(self) -> None:
        body = json.dumps(scan_manifest(self.root), separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("viz"), help="Visualization output root.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--poll-ms", type=int, default=2500, help="Browser manifest refresh interval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    mimetypes.add_type("application/pdf", ".pdf")
    LiveVizHandler.root = args.root.resolve()
    LiveVizHandler.poll_ms = args.poll_ms
    server = ThreadingHTTPServer((args.host, args.port), LiveVizHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving live MSPF viewer at {url}")
    print(f"Scanning {LiveVizHandler.root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
