from __future__ import annotations

import re


def hms_to_sec(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unsupported timestamp format: {value}")


def parse_visual_cue_markers(cue_sheet_text: str, total_duration: float | None = None) -> list[dict[str, object]]:
    explicit_marker_pattern = re.compile(
        r"#\s*(?:(?P<label>[A-Z])\.\s*)?(?P<time>\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?)\s*(?P<text>.*)$",
        re.IGNORECASE,
    )
    line_comment_pattern = re.compile(
        r"^\s*(?P<time>\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?)\s+(?:(?P<track>\d+)\s+)?(?P<title>[^#]*?)\s*#\s*(?P<text>.+?)\s*$"
    )
    continuation_pattern = re.compile(r"^\s*#\s*(?P<text>.+?)\s*$")

    markers: list[tuple[str, float, list[str]]] = []
    for line in str(cue_sheet_text).splitlines():
        marker = explicit_marker_pattern.search(line)
        if marker:
            label = (marker.group("label") or f"cue_{len(markers) + 1:02d}").upper()
            start_time = hms_to_sec(marker.group("time"))
            text = marker.group("text").strip()
            markers.append((label, start_time, [text] if text else []))
            continue

        line_comment = line_comment_pattern.match(line)
        if line_comment:
            label = f"cue_{len(markers) + 1:02d}"
            start_time = hms_to_sec(line_comment.group("time"))
            text = line_comment.group("text").strip()
            markers.append((label, start_time, [text] if text else []))
            continue

        continuation = continuation_pattern.match(line)
        if continuation and markers:
            text = continuation.group("text").strip()
            if text:
                markers[-1][2].append(text)

    fallback_end = total_duration
    if fallback_end is None and markers:
        fallback_end = markers[-1][1] + 45.0

    cues: list[dict[str, object]] = []
    for idx, (label, start_time, parts) in enumerate(markers):
        if fallback_end is not None and start_time >= fallback_end:
            continue
        next_start = fallback_end or (start_time + 45.0)
        for _next_label, candidate_start, _next_parts in markers[idx + 1 :]:
            if candidate_start > start_time:
                next_start = candidate_start
                break
        if fallback_end is not None:
            next_start = min(next_start, fallback_end)
        summary = " ".join(" ".join(parts).split())
        if summary and next_start > start_time:
            cues.append({"id": label, "start": max(0.0, start_time), "end": max(start_time, next_start), "summary": summary})
    return cues
