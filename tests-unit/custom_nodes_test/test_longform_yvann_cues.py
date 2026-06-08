from pathlib import Path

from script_examples.longform_yvann_cue_parser import parse_visual_cue_markers
from script_examples.longform_yvann_runner import JobConfig, LongformYvannRunner


USER_STYLE_CUE_SHEET = """00:00:00  1 Deep Hertz - Melting Sun # Close-up of a colossal heavy-lift rocket on a night launchpad.
00:00:01  2 Pre-Ignition Pressure # The rocket remains locked in place but pressure builds violently.
00:00:02  3 Ignition Bloom # Ignition begins, orange-white engine glow blooms under the rocket.
"""


def test_dashboard_preview_extracts_line_timestamp_hash_comments():
    cues = parse_visual_cue_markers(USER_STYLE_CUE_SHEET, total_duration=30.0)

    assert [cue["start"] for cue in cues] == [0.0, 1.0, 2.0]
    assert [cue["end"] for cue in cues] == [1.0, 2.0, 30.0]
    assert cues[0]["summary"] == "Close-up of a colossal heavy-lift rocket on a night launchpad."
    assert cues[2]["summary"] == "Ignition begins, orange-white engine glow blooms under the rocket."


def test_backend_runner_extracts_line_timestamp_hash_comments():
    runner = LongformYvannRunner.__new__(LongformYvannRunner)

    cues = runner._extract_visual_cues(USER_STYLE_CUE_SHEET, total_duration=30.0)

    assert [cue.start_time for cue in cues] == [0.0, 1.0, 2.0]
    assert [cue.end_time for cue in cues] == [1.0, 2.0, 30.0]
    assert cues[0].summary == "Close-up of a colossal heavy-lift rocket on a night launchpad."
    assert cues[2].summary == "Ignition begins, orange-white engine glow blooms under the rocket."


def test_existing_explicit_comment_timestamp_format_still_works():
    cue_sheet = """00:00:00  1 Track name # A. 00:00:00 Rocket preparing for launch.
00:00:45  2 Next section # B. 00:00:45 Rocket taking off.
"""

    preview_cues = parse_visual_cue_markers(cue_sheet, total_duration=90.0)
    runner = LongformYvannRunner.__new__(LongformYvannRunner)
    backend_cues = runner._extract_visual_cues(cue_sheet, total_duration=90.0)

    assert [cue["id"] for cue in preview_cues] == ["A", "B"]
    assert [cue.cue_id for cue in backend_cues] == ["A", "B"]
    assert preview_cues[0]["summary"] == "Rocket preparing for launch."
    assert backend_cues[1].summary == "Rocket taking off."


def test_markdown_clip_plan_extracts_explicit_ranges():
    cue_sheet = """# Example Plan

## Section 1 - 0:00-1:01

### Clip 1A - 0:00-0:20 - 20s
```text
The road movie begins inside the TV screen. Keep the CRT centered.
```

### Clip 1B - 0:20-0:40 - 20s
```text
The driver continues forward. Preserve sunglasses and desert direction.
```
"""

    cues = parse_visual_cue_markers(cue_sheet, total_duration=445.0)

    assert [cue["start"] for cue in cues] == [0.0, 20.0]
    assert [cue["end"] for cue in cues] == [20.0, 40.0]
    assert cues[0]["id"] == "Clip 1A"
    assert cues[0]["summary"] == "The road movie begins inside the TV screen. Keep the CRT centered."
    assert "Preserve sunglasses" in cues[1]["summary"]


def test_dense_visual_cues_do_not_force_one_second_render_chunks():
    cue_sheet = "\n".join(
        f"00:00:{idx:02d}  {idx + 1} Section {idx + 1} # Visual scene {idx + 1}"
        for idx in range(31)
    )
    config = JobConfig(
        script_path="script.txt",
        audio_path="audio.mp3",
        global_style_prompt="cinematic",
        output_root="output",
        render_profile="custom",
        chunk_duration_seconds=45.0,
        image_interval_seconds=6.0,
    )
    runner = LongformYvannRunner.__new__(LongformYvannRunner)
    runner.config = config
    runner._audio_duration = 30.5
    runner.load_script_text = lambda: cue_sheet
    runner.audio_chunks_dir = Path("audio_chunks")
    runner.videos_dir = Path("videos")

    chunks = runner.build_manifest()

    assert len(chunks) == 1
    assert chunks[0].chunk_duration == 30.5
    assert chunks[0].visual_cues is not None
    assert len(chunks[0].visual_cues) == 31


def test_one_second_cues_still_plan_one_image_each():
    config = JobConfig(
        script_path="script.txt",
        audio_path="audio.mp3",
        global_style_prompt="cinematic",
        output_root="output",
        render_profile="custom",
        image_interval_seconds=5.0,
    )
    runner = LongformYvannRunner.__new__(LongformYvannRunner)
    runner.config = config

    assert runner._image_count_for_cue_duration(1.0) == 1


def test_five_second_cues_plan_start_and_end_keyframes():
    config = JobConfig(
        script_path="script.txt",
        audio_path="audio.mp3",
        global_style_prompt="cinematic",
        output_root="output",
        render_profile="custom",
        image_interval_seconds=5.0,
    )
    runner = LongformYvannRunner.__new__(LongformYvannRunner)
    runner.config = config

    assert runner._image_count_for_cue_duration(5.0) == 2
