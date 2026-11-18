# YouTube Playlist Comparator

`playlist_compare.py` compares **two public YouTube playlists** and writes a detailed,
timestamped CSV. Every run creates a **new file**, so previous reports are never overwritten.
It works across accounts — no login or Google Cloud project needed.

---

## Two ways to run it

### 1. Interactive (default)

```bash
python3 playlist_compare.py
```

It prints a banner and asks for two links:

```
==============================================================
  YouTube Playlist Comparator
==============================================================
  Enter TWO public playlist links when prompted.
  Full URL or just the playlist ID both work.

  Prefer command-line instead? Run:
      python3 playlist_compare.py --a "<link A>" --b "<link B>"

  Output: playlist-comparison.csv (saved beside this script)
  Press Ctrl+C at any time to cancel.
==============================================================
Playlist A link (or playlist ID):
Playlist B link (or playlist ID):
```

You can paste any of these:

- `https://www.youtube.com/playlist?list=PLyh3SzBjNbIChSTpTzMFkosQjNR_0sl26`
- `https://youtube.com/playlist?list=PLyh3SzBjNbIChSTpTzMFkosQjNR_0sl26&si=some-share-id`
- `PLyh3SzBjNbIChSTpTzMFkosQjNR_0sl26` (just the ID)

Empty input re-prompts. Press **Ctrl+C** to cancel at any point.

### 2. Command-line arguments (no prompts)

```bash
python3 playlist_compare.py --a "<link A>" --b "<link B>"
```

If you supply only one of `--a` / `--b`, it prompts for the other.

**Both modes generate the same CSV, each with a unique timestamped filename.**

Example output:
```
Wrote /home/developer/Documents/AI/Projects/demo/playlist-comparison_2026-11-18_104021.csv
```

The pattern is `playlist-comparison_YYYY-MM-DD_HHMMSS.csv`. If you run twice in the same
second, the second file gets a `_2` suffix. Use `--output my-report.csv` if you explicitly
want a fixed filename (the script refuses to overwrite an existing file in that case).

---

## Options

| Flag | Description |
|---|---|
| `--a LINK` | Playlist A link. Skips prompt A. |
| `--b LINK` | Playlist B link. Skips prompt B. |
| `--cookies FILE` | Use a Netscape `cookies.txt` when YouTube bot-checks your IP. |
| `--jobs N` | Parallel lookups, default 4. Lower it (e.g. 2) if you get bot-checked. |
| `--output FILE` | Exact output filename instead of the timestamped default. Won't overwrite an existing file. |
| `-h`, `--help` | Full help with examples. |

See all options with:

```bash
python3 playlist_compare.py --help
```

---

## Output: `playlist-comparison_<date>_<time>.csv`

Saved in the same folder as the script. One row per distinct video across both playlists.
A new file is created on every run, so you can keep a history of reports and compare changes
over time.

| Column | Meaning |
|---|---|
| `video_id` | YouTube video ID (unique key) |
| `title` | Video title |
| `channel` | Uploader channel |
| `duration_seconds` / `duration` | Length in seconds / `M:SS` |
| `status` | `AVAILABLE`, `PRIVATE`, `REMOVED`, `AGE_RESTRICTED` |
| `status_detail` | Human-readable status |
| `in_playlist_A` / `pos_A` | Membership and 1-based position in playlist A |
| `in_playlist_B` / `pos_B` | Membership and 1-based position in playlist B |
| `presence` | `BOTH`, `ONLY_A`, or `ONLY_B` |
| `embeddable_oembed` | oEmbed HTTP code: 200 = embeddable, 401 = embed-disabled, 403 = private, 404 = deleted |
| `url` | Direct watch URL |

### Sync summary at the end

Every generated CSV has **one blank line** after the last video row, followed by a
sync verdict block:

```
SYNC STATUS,FULLY SYNCED
PLAYLIST A VIDEOS,270
PLAYLIST B VIDEOS,270
COMMON VIDEOS,270
ONLY IN A,0
ONLY IN B,0
ORDER,DIFFERENT
UNAVAILABLE,1
GENERATED,2026-11-18 11:29:04
```

`SYNC STATUS` is `FULLY SYNCED` when both playlists contain exactly the same video IDs
(no rows with `presence = ONLY_A` or `ONLY_B`), otherwise `NOT SYNCED`.

`ORDER` is `MATCH` only when every shared video sits at the same position in both
playlists — the content can be fully synced while the ordering still differs.

If you parse the file programmatically, stop reading at the blank line — the video rows
end there and the summary block starts.

### Status meanings

- **AVAILABLE** — public and playable.
- **PRIVATE** — owner-only; the playlist entry shows `[Private video]` with no metadata.
- **REMOVED** — the video itself is gone from YouTube (deleted or rights-blocked). The playlist still
  contains a stale entry pointing at the dead ID, which is why the row appears in the CSV.
- **AGE_RESTRICTED** — the video exists but needs an age-confirmation sign-in to play.

Filter to just the playable rows with `status == AVAILABLE`.

---

## How availability is determined

Three independent sources are used in priority order so bot-checks cannot create false
"removed" results:

1. **`yt-dlp`** — official `availability` field, title, duration, channel.
2. **YouTube player API** — used when yt-dlp is bot-checked. Distinguishes
   `Video unavailable` (REMOVED), age-gate (LOGIN_REQUIRED), private, and OK.
3. **oEmbed** — fallback: HTTP 403 = private, 404 = deleted, 401 = embed-disabled
   (still available), 200 = available.

Playlist membership and ordering come from `yt-dlp --flat-playlist --dump-single-json`
with full continuation pagination, so long playlists are not truncated to 100 videos.

---

## Requirements

- Python 3.8+
- `yt-dlp` installed at `~/.local/bin/yt-dlp` or on `PATH`:

  ```bash
  pip install -U yt-dlp
  ```

- Internet access. No API key, OAuth or Google Cloud project required.

---

## Tips

- If you get a lot of bot-check failures, either lower `--jobs` to 1–2, wait a few minutes,
  or supply `--cookies cookies.txt` exported from a signed-in browser.
- Unlike the old fixed filename, timestamped output means you never need to close a CSV open
  in LibreOffice before running again — each run writes a separate file.
- Runtime is roughly 3–4 minutes for ~270 videos, depending on network and rate limits.

---

## Other files in this folder

- `playlist-comparison.csv` — the original report from before timestamped output was added.
- `playlist-comparison_YYYY-MM-DD_HHMMSS.csv` — each newer run's report.
- `README.md` — this document.
