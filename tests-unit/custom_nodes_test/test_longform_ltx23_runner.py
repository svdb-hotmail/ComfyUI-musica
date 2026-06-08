import json
from pathlib import Path

from PIL import Image

from script_examples.longform_ltx23_runner import LTXJobConfig, LongformLTX23Runner


ROOT = Path(__file__).resolve().parents[2]
MOVIE_BUILDER = ROOT / "script_examples" / "workflows" / "Movie_Builder_LTX2.3_workflow.json"
IA2V = ROOT / "script_examples" / "workflows" / "video_ltx2_3_ia2v.json"


def _write_inputs(tmp_path):
    audio = tmp_path / "audio.wav"
    plan = tmp_path / "plan.txt"
    image = tmp_path / "frame.png"
    audio.write_bytes(b"placeholder")
    plan.write_text(
        "00:00:00  1 Opening # Static first image begins to breathe with the kick.\n"
        "00:00:05  2 Build # Lights pulse harder and smoke expands.\n",
        encoding="utf-8",
    )
    Image.new("RGB", (64, 64), (32, 64, 96)).save(image)
    return audio, plan, image


def _runner(tmp_path, renderer="movie_builder"):
    audio, plan, image = _write_inputs(tmp_path)
    template = MOVIE_BUILDER if renderer == "movie_builder" else IA2V
    config = LTXJobConfig(
        audio_path=str(audio),
        image_paths=[str(image)],
        prompt_plan_path=str(plan),
        output_root=str(tmp_path / "out"),
        comfy_root=str(ROOT),
        workflow_template_path=str(template),
        renderer=renderer,
        shot_duration_seconds=5.0,
        width=1280,
        height=720,
        fps=24,
    )
    runner = LongformLTX23Runner(config)
    runner.audio_duration = 10.0
    return runner


def test_manifest_uses_timestamp_cues_for_shots(tmp_path):
    runner = _runner(tmp_path)

    shots = runner.build_manifest()

    assert len(shots) == 2
    assert [shot.start_time for shot in shots] == [0.0, 5.0]
    assert [shot.duration for shot in shots] == [5.0, 5.0]
    assert "Static first image" in shots[0].prompt
    assert shots[0].seed != shots[1].seed


def test_manifest_uses_markdown_clip_plan_ranges(tmp_path):
    runner = _runner(tmp_path)
    runner.prompt_plan_path.write_text(
        "### Clip 1A - 0:00-0:20 - 20s\n"
        "```text\nInside the TV, the driver starts the desert road movie.\n```\n\n"
        "### Clip 1B - 0:20-0:40 - 20s\n"
        "```text\nKeep the same driver and continue the forward road motion.\n```\n",
        encoding="utf-8",
    )
    runner.audio_duration = 45.0

    shots = runner.build_manifest()

    assert len(shots) == 2
    assert [shot.start_time for shot in shots] == [0.0, 20.0]
    assert [shot.duration for shot in shots] == [20.0, 20.0]
    assert "driver starts the desert road movie" in shots[0].prompt
    assert "Keep continuity from the previous shot" in shots[1].prompt


def test_previous_final_frame_becomes_next_reference(tmp_path):
    runner = _runner(tmp_path)
    shots = runner.build_manifest()
    final_frame = runner.final_frame_path_for_shot(shots[0])
    final_frame.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (96, 64, 32)).save(final_frame)
    shots[0].final_frame_path = str(final_frame)

    runner._apply_continuity_reference(shots[1], shots[0])

    assert shots[1].reference_image_path == str(final_frame)
    assert shots[1].image_path == str(final_frame)


def test_output_video_from_history_accepts_animated_image_mp4(tmp_path):
    runner = _runner(tmp_path, renderer="ia2v")
    runner.comfy_root = tmp_path
    video_dir = tmp_path / "output" / "longform_ltx23" / "job_test"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "shot_0001_00001_.mp4"
    video_path.write_bytes(b"fake mp4")

    history = {
        "outputs": {
            "341": {
                "images": [
                    {
                        "filename": video_path.name,
                        "subfolder": "longform_ltx23/job_test",
                        "type": "output",
                    }
                ],
                "animated": [True],
            }
        }
    }

    assert runner._output_video_from_history(history) == video_path


def test_movie_builder_patch_sets_prompt_audio_duration_and_frame_count(tmp_path):
    runner = _runner(tmp_path, renderer="movie_builder")
    runner.config.enable_upscale = True
    runner.config.enable_voice_reference = True
    shot = runner.build_manifest()[0]

    workflow = runner.patched_workflow_for_shot(shot)
    nodes = LongformLTX23Runner._workflow_nodes(workflow)
    text_nodes = [node for node in nodes if node.get("type") == "PrimitiveStringMultiline" and node.get("title") == "Text Prompt"]
    trim_nodes = [node for node in nodes if node.get("type") == "TrimAudioDuration"]
    length_nodes = [node for node in nodes if node.get("type") == "PrimitiveInt" and node.get("title") == "Length"]
    boolean_nodes = [node for node in nodes if node.get("type") == "PrimitiveBoolean"]

    assert any(shot.prompt in (node.get("widgets_values") or [""])[0] for node in text_nodes)
    assert any((node.get("widgets_values") or [])[:2] == [0, 5.0] for node in trim_nodes)
    assert any((node.get("widgets_values") or [None])[0] == 121 for node in length_nodes)
    assert any(node.get("title") == "Enable Upscale" and node.get("widgets_values", [None])[0] is True for node in boolean_nodes)
    assert any(node.get("title") == "Enable Voice Reference" and node.get("widgets_values", [None])[0] is True for node in boolean_nodes)


def test_ia2v_patch_sets_subgraph_proxy_widget_sources(tmp_path):
    runner = _runner(tmp_path, renderer="ia2v")
    shot = runner.build_manifest()[0]

    workflow = runner.patched_workflow_for_shot(shot)
    by_id = {str(node.get("id")): node for node in LongformLTX23Runner._workflow_nodes(workflow)}

    assert by_id["319"]["widgets_values"][0] == shot.prompt
    assert by_id["331"]["widgets_values"][0] == 5.0
    assert by_id["323"]["widgets_values"][0] == 24
    assert by_id["286"]["widgets_values"][0] == shot.seed


def test_movie_builder_workflow_is_present_and_has_video_subgraph():
    workflow = json.loads(MOVIE_BUILDER.read_text(encoding="utf-8"))
    subgraph_names = {subgraph.get("name") for subgraph in workflow.get("definitions", {}).get("subgraphs", [])}

    assert "VIDEO GEN" in subgraph_names
