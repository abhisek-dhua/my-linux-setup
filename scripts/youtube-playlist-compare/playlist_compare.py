#!/usr/bin/env python3
"""
playlist_compare.py — compare two public YouTube playlists -> timestamped CSV

TWO WAYS TO RUN
  1. Interactive (default) — just run it and answer the two prompts:
         python3 playlist_compare.py
     You can paste a full playlist URL or just its ID.

  2. Command-line arguments — no prompts, useful in scripts:
         python3 playlist_compare.py --a "<link A>" --b "<link B>"

BOTH modes produce the same CSV next to this script.

What it does:
  1. extracts every video from both playlists (full pagination),
  2. resolves each video's real availability (yt-dlp -> player API -> oEmbed),
  3. writes a timestamped CSV with positions, overlap and status.

OUTPUT FILE
  playlist-comparison_YYYY-MM-DD_HHMMSS.csv  (e.g. playlist-comparison_2026-11-18_102530.csv)
  Every run creates a NEW file, so previous reports are never overwritten.
  Use --output FILE to choose a specific filename instead.

Requires yt-dlp (at ~/.local/bin/yt-dlp or on PATH).
Optional: --cookies /path/to/cookies.txt if YouTube bot-checks your IP.
See README.md for examples and a column reference.
"""

import argparse
import concurrent.futures as cf
import csv
import datetime
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def timestamped_output():
    """Build a unique, sortable filename so runs never overwrite each other."""
    base = datetime.datetime.now().strftime("playlist-comparison_%Y-%m-%d_%H%M%S")
    path = os.path.join(SCRIPT_DIR, base + ".csv")
    n = 2
    while os.path.exists(path):  # collision guard for two runs in the same second
        path = os.path.join(SCRIPT_DIR, f"{base}_{n}.csv")
        n += 1
    return path


YTAPI = "https://www.youtube.com/youtubei/v1/player"
OEMBED = "https://www.youtube.com/oembed?url=https%3A//www.youtube.com/watch%3Fv%3D{}&format=json"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
PLAYLIST_RE = re.compile(r"(?:list=|/playlist/)([A-Za-z0-9_-]{10,})")


# ---------------------------------------------------------------- helpers
def find_ytdlp():
    for c in (os.path.expanduser("~/.local/bin/yt-dlp"), "yt-dlp"):
        p = shutil.which(c)
        if p:
            return p
    raise SystemExit("ERROR: yt-dlp not found. Install it with: pip install -U yt-dlp")


def playlist_id(raw):
    m = PLAYLIST_RE.search(raw.strip())
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", raw.strip()):
        return raw.strip()
    raise ValueError(f"cannot find a playlist ID in: {raw!r}")


def ytdlp_run(args, cookies=None):
    cmd = [YTDLP, "--ignore-errors", "--no-warnings"]
    if cookies:
        cmd += ["--cookies", cookies]
    cmd += args
    r = subprocess_run(cmd)
    return r[0], r[1]


def subprocess_run(cmd):
    import subprocess
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout, p.stderr


def fetch_playlist(pid, cookies=None):
    out, err = ytdlp_run(["--flat-playlist", "--dump-single-json",
                           f"https://www.youtube.com/playlist?list={pid}"], cookies)
    try:
        d = json.loads(out)
    except json.JSONDecodeError:
        raise SystemExit(f"ERROR: could not read playlist {pid}\n{err[-600:]}")
    if not d.get("entries"):
        raise SystemExit(f"ERROR: playlist {pid} is empty, private, or unavailable")
    return d


# ---------------------------------------------------------------- availability
def innertube(vid):
    """Return (status, reason) from YouTube's player API, or (None, None)."""
    body = json.dumps({
        "context": {"client": {"clientName": "WEB",
                               "clientVersion": "2.20250101.00.00",
                               "hl": "en", "gl": "US"}},
        "videoId": vid,
    }).encode()
    req = urllib.request.Request(YTAPI, data=body, headers={
        "Content-Type": "application/json", "User-Agent": UA,
        "Origin": "https://www.youtube.com",
    })
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            j = json.loads(r.read().decode())
        p = j.get("playabilityStatus", {})
        return p.get("status"), p.get("reason", "")
    except Exception:
        return None, None


def oembed(vid):
    """Return (http_code, title, author)."""
    req = urllib.request.Request(OEMBED.format(vid), headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            j = json.loads(r.read().decode())
            return r.status, j.get("title", ""), j.get("author_name", "")
    except urllib.error.HTTPError as e:
        return e.code, "", ""
    except Exception:
        return None, "", ""


def classify(vid, listing_title, ytdlp, itube, oe):
    """Resolve a single status. Priority: yt-dlp > innertube > oEmbed > listing."""
    st, title, dur, ch, err = ytdlp

    if st == "public":
        return "AVAILABLE", "Available", title, dur, ch

    msg = (err or "").lower()
    if "private video" in msg:
        return "PRIVATE", "Private video", None, None, None
    if "age" in msg and "sign in" in msg:
        return "AGE_RESTRICTED", "Age-restricted", None, None, None
    if "not available" in msg or "video unavailable" in msg:
        return "REMOVED", "Removed / not available", None, None, None

    # yt-dlp failed without a usable reason (usually a bot-check) -> innertube
    istatus, ireason = itube
    ir = (ireason or "").lower()
    if istatus == "UNPLAYABLE" and ("not available" in ir or "video unavailable" in ir):
        return "REMOVED", "Removed / not available", None, None, None
    if istatus == "LOGIN_REQUIRED" and "age" in ir:
        return "AGE_RESTRICTED", "Age-restricted", None, None, None
    if istatus == "LOGIN_REQUIRED" and ("private" in ir or "sign in" in ir):
        return "PRIVATE", "Private video", None, None, None
    if istatus == "OK":
        return "AVAILABLE", "Available", title, dur, ch

    # last resort: oEmbed (403 = private, 404 = deleted, 401 = embed-disabled)
    code, ot, oa = oe
    if code == 403:
        return "PRIVATE", "Private video", None, None, None
    if code == 404:
        return "REMOVED", "Removed / not available", None, None, None
    if listing_title in ("[Private video]", "[Deleted video]"):
        if "Private" in listing_title:
            return "PRIVATE", "Private video", None, None, None
        return "REMOVED", "Removed / not available", None, None, None
    # HTTP 200 or 401 -> the video exists (401 only means embedding is disabled)
    return "AVAILABLE", "Available", title or ot, dur, ch or oa


def resolve_one(vid, listing_title, cookies):
    out, err = ytdlp_run(["--skip-download", "--print-json",
                           f"https://www.youtube.com/watch?v={vid}"], cookies)
    line = next((l for l in out.splitlines() if l.lstrip().startswith("{")), None)
    if line:
        try:
            j = json.loads(line)
            return ("public", j.get("title"), j.get("duration"),
                    j.get("channel") or j.get("uploader"), None), None, None
        except json.JSONDecodeError:
            pass
    m = re.search(r"ERROR:\s*\[youtube\]\s*\S+:\s*(.+)", err)
    return ("error", None, None, None, m.group(1) if m else ""), innertube(vid), oembed(vid)


# ---------------------------------------------------------------- prompt
def prompt_for_playlist(label):
    """Ask until the user enters a valid playlist link or ID."""
    while True:
        try:
            raw = input(f"{label} link (or playlist ID): ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nCancelled.")
        if not raw:
            print("  Nothing entered — please paste a playlist link, or press Ctrl+C to quit.")
            continue
        try:
            return playlist_id(raw)
        except ValueError as e:
            print(f"  {e}")
            print('  Example: https://www.youtube.com/playlist?list=PLyh3SzBjNbIChSTpTzMFkosQjNR_0sl26')


BANNER = """
==============================================================
  YouTube Playlist Comparator
==============================================================
  Enter TWO public playlist links when prompted.
  Full URL or just the playlist ID both work.

  Prefer command-line instead? Run:
      python3 playlist_compare.py --a "<link A>" --b "<link B>"

  Output: playlist-comparison_<date>_<time>.csv (new file each run)
  Press Ctrl+C at any time to cancel.
==============================================================
"""


# ---------------------------------------------------------------- main
def main():
    global YTDLP
    YTDLP = find_ytdlp()

    ap = argparse.ArgumentParser(
        prog="playlist_compare.py",
        description=("Compare two public YouTube playlists and write a timestamped CSV. "
                     "Run with no arguments for interactive prompts, or pass --a and --b."),
        epilog=("EXAMPLES\n"
                "  Interactive (asks for both links):\n"
                "      python3 playlist_compare.py\n\n"
                "  Command line (no prompts):\n"
                "      python3 playlist_compare.py --a 'https://youtube.com/playlist?list=PLxxxx' "
                "--b 'https://youtube.com/playlist?list=PLyyyy'\n\n"
                "  With cookies (if YouTube bot-checks you):\n"
                "      python3 playlist_compare.py --cookies cookies.txt\n\n"
                "  Faster / gentler lookups:\n"
                "      python3 playlist_compare.py --jobs 2\n\n"
                "Choose the filename: --output my-report.csv\n\n"
                "Default output: playlist-comparison_YYYY-MM-DD_HHMMSS.csv\n"
                "A NEW file is created on every run — nothing is overwritten.\n"
                "See README.md for the column reference."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", help="Playlist A link — skips prompt A")
    ap.add_argument("--b", help="Playlist B link — skips prompt B")
    ap.add_argument("--cookies", help="cookies.txt to reduce YouTube bot-checks")
    ap.add_argument("--jobs", type=int, default=4, help="parallel lookups (default 4)")
    ap.add_argument("--output", help="exact output filename (default: timestamped CSV)")
    args = ap.parse_args()

    if not args.a and not args.b:
        print(BANNER)
        pa = prompt_for_playlist("Playlist A")
        pb = prompt_for_playlist("Playlist B")
    else:
        # If only one side was given, prompt for the other one.
        try:
            pa = playlist_id(args.a) if args.a else prompt_for_playlist("Playlist A")
            pb = playlist_id(args.b) if args.b else prompt_for_playlist("Playlist B")
        except ValueError as e:
            raise SystemExit(f"ERROR: {e}")

    print(f"\nPlaylist A: https://youtube.com/playlist?list={pa}")
    print(f"Playlist B: https://youtube.com/playlist?list={pb}\n")

    A = fetch_playlist(pa, args.cookies)
    B = fetch_playlist(pb, args.cookies)
    print(f"A: {len(A['entries'])} videos — {A.get('title')} by {A.get('uploader')}")
    print(f"B: {len(B['entries'])} videos — {B.get('title')} by {B.get('uploader')}")

    listing = {}
    for d in (A, B):
        for e in d["entries"]:
            if e.get("id"):
                listing.setdefault(e["id"], e.get("title"))

    ids = list(listing)
    print(f"\nResolving {len(ids)} distinct videos ...")
    resolved = {}
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(resolve_one, v, listing[v], args.cookies): v for v in ids}
        for i, f in enumerate(cf.as_completed(futs), 1):
            vid = futs[f]
            try:
                yt, it, oe = f.result()
            except Exception as e:
                yt, it, oe = ("error", None, None, None, str(e)), (None, None), (None, "", "")
            resolved[vid] = classify(vid, listing[vid], yt, it, oe)
            if i % 25 == 0 or i == len(ids):
                print(f"  {i}/{len(ids)}")

    # ---------- build rows ----------
    liveA = [e["id"] for e in A["entries"] if e.get("id")]
    liveB = [e["id"] for e in B["entries"] if e.get("id")]
    setA, setB = set(liveA), set(liveB)

    def mmss(s):
        try:
            s = int(s)
            return f"{s // 60}:{s % 60:02d}"
        except (TypeError, ValueError):
            return "—"

    rows = []
    for vid in sorted(setA | setB,
                      key=lambda v: (liveA.index(v) if v in setA else 10**6,
                                     liveB.index(v) if v in setB else 10**6)):
        status, detail, title, dur, ch = resolved[vid]
        entries = [e for d in (A, B) for e in d["entries"] if e.get("id") == vid]
        base = entries[0] if entries else {}
        if status == "PRIVATE":
            title, ch, dur = "Private video", "", ""
        rows.append({
            "video_id": vid,
            "title": title or base.get("title") or "",
            "channel": ch or base.get("channel") or base.get("uploader") or "",
            "duration_seconds": dur if dur is not None else (base.get("duration") or ""),
            "duration": mmss(dur if dur is not None else base.get("duration")),
            "status": status,
            "status_detail": detail,
            "in_playlist_A": "YES" if vid in setA else "NO",
            "pos_A": liveA.index(vid) + 1 if vid in setA else "",
            "in_playlist_B": "YES" if vid in setB else "NO",
            "pos_B": liveB.index(vid) + 1 if vid in setB else "",
            "presence": "BOTH" if vid in setA & setB else ("ONLY_A" if vid in setA else "ONLY_B"),
            "embeddable_oembed": "",
            "url": f"https://www.youtube.com/watch?v={vid}",
        })

    # oEmbed embeddability for every row (parallel, used for the CSV column)
    def oe_code(v):
        return v, oembed(v)[0]
    codes = {}
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for vid, code in ex.map(oe_code, [r["video_id"] for r in rows]):
            codes[vid] = code or ""
    for r in rows:
        r["embeddable_oembed"] = codes.get(r["video_id"], "")

    # ---------- verdict ----------
    from collections import Counter
    both = sum(1 for r in rows if r["presence"] == "BOTH")
    oa = [r for r in rows if r["presence"] == "ONLY_A"]
    ob = [r for r in rows if r["presence"] == "ONLY_B"]
    synced = not oa and not ob
    order_match = all(r["pos_A"] == r["pos_B"] for r in rows if r["presence"] == "BOTH")
    verdict = "FULLY SYNCED" if synced else "NOT SYNCED"
    order_verdict = "MATCH" if order_match else "DIFFERENT"

    output = args.output or timestamped_output()
    if not os.path.isabs(output):
        output = os.path.join(SCRIPT_DIR, output)
    if os.path.exists(output) and args.output:
        raise SystemExit(f"ERROR: {output} already exists. Choose a different --output name.")
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        # one blank line, then a sync summary block (see README)
        f.write("\n")
        sw = csv.writer(f)
        sw.writerow(["SYNC STATUS", verdict])
        sw.writerow(["PLAYLIST A VIDEOS", len(liveA)])
        sw.writerow(["PLAYLIST B VIDEOS", len(liveB)])
        sw.writerow(["COMMON VIDEOS", both])
        sw.writerow(["ONLY IN A", len(oa)])
        sw.writerow(["ONLY IN B", len(ob)])
        sw.writerow(["ORDER", order_verdict])
        sw.writerow(["UNAVAILABLE", sum(1 for r in rows if r["status"] != "AVAILABLE")])
        sw.writerow(["GENERATED", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    print(f"\nWrote {output}")
    print(f"  SYNC STATUS: {verdict}  |  order: {order_verdict}")
    print(f"  rows={len(rows)}  both={both}  only_A={len(oa)}  only_B={len(ob)}")
    print("  status:", dict(Counter(r["status"] for r in rows)))
    if oa:
        print("\n  Only in A:")
        for r in oa:
            print(f"    A@{r['pos_A']:<4} {r['status']:<16} {r['title'][:52]}")
    if ob:
        print("\n  Only in B:")
        for r in ob:
            print(f"    B@{r['pos_B']:<4} {r['status']:<16} {r['title'][:52]}")
    bad = [r for r in rows if r["status"] != "AVAILABLE"]
    if bad:
        print("\n  Not available:")
        for r in bad:
            print(f"    {r['status']:<16} A@{r['pos_A'] or '-':<5} B@{r['pos_B'] or '-':<5} {r['title'][:48]}")


if __name__ == "__main__":
    main()
