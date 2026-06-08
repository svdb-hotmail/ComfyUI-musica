import json
from pathlib import Path

from script_examples.ltx_longform_job_package import materialize_package, package_to_runner_config


def test_package_to_runner_config_accepts_plan_text(tmp_path):
    package = {
        "schema": "storydirector.ltx-longform-job.v1",
        "job_id": "sdp_job_001",
        "audio": {"path": "input/song.wav"},
        "images": [{"path": "input/keyframe.png"}],
        "prompt_plan_text": "### Clip 1 - 0:00-0:06 - 6s\n```text\nForward road shot.\n```\n",
        "settings": {
            "output_root": str(tmp_path / "out"),
            "renderer": "ia2v",
            "workflow_template_path": "script_examples/workflows/video_ltx2_3_ia2v.json",
            "max_shots": 1,
            "use_previous_final_frame": True,
        },
    }

    config, plan = package_to_runner_config(package, comfy_root=tmp_path)

    assert config["job_id"] == "sdp_job_001"
    assert config["audio_path"] == "input/song.wav"
    assert config["image_paths"] == ["input/keyframe.png"]
    assert config["renderer"] == "ia2v"
    assert config["max_shots"] == 1
    assert "Forward road shot" in plan


def test_materialize_package_writes_config_and_plan_from_shots(tmp_path):
    audio = tmp_path / "input" / "song.wav"
    image = tmp_path / "input" / "keyframe.png"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")
    image.write_bytes(b"image")
    package_path = tmp_path / "job.json"
    package_path.write_text(
        json.dumps(
            {
                "id": "sdp_scene_42",
                "audio_path": str(audio),
                "keyframes": [{"image_path": str(image)}],
                "shots": [
                    {"id": "Clip 1A", "start": 0, "end": 6, "prompt": "Drive forward through impossible salt flats."},
                    {"id": "Clip 1B", "start_time": 6, "end_time": 12, "summary": "Continue from the last frame into stranger terrain."},
                ],
                "settings": {
                    "output_root": str(tmp_path / "out"),
                    "width": 1024,
                    "height": 576,
                    "fps": 24,
                    "final_concat": False,
                },
            }
        ),
        encoding="utf-8",
    )

    config_path, config = materialize_package(package_path, comfy_root=tmp_path)

    assert config_path.exists()
    assert Path(config["prompt_plan_path"]).exists()
    generated_config = json.loads(config_path.read_text(encoding="utf-8"))
    generated_plan = Path(config["prompt_plan_path"]).read_text(encoding="utf-8")
    assert generated_config["job_id"] == "sdp_scene_42"
    assert generated_config["width"] == 1024
    assert generated_config["final_concat"] is False
    assert "### Clip 1A - 0:00-0:06 - 6s" in generated_plan
    assert "Drive forward through impossible salt flats." in generated_plan
    assert "### Clip 1B - 0:06-0:12 - 6s" in generated_plan
