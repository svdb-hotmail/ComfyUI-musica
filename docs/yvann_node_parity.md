# Yvann Node Parity (Remote, Local, Fork)

This repo now includes a repeatable custom-node sync flow so the same Yvann stack can be installed on:

- remote container ComfyUI
- local clone
- any machine that clones your GitHub fork

## Required nodes

Defined in `scripts/required_custom_nodes.json`:

- `comfyui_yvann-nodes` -> https://github.com/yvann-ba/ComfyUI_Yvann-Nodes
- `comfyui-videohelpersuite` -> https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
- `comfyui-animatediff-evolved` -> https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
- `comfyui_ipadapter_plus` -> https://github.com/cubiq/ComfyUI_IPAdapter_plus
- `comfyui-advanced-controlnet` -> https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet
- `comfyui_controlnet_aux` -> https://github.com/Fannovel16/comfyui_controlnet_aux
- `comfyui-kjnodes` -> https://github.com/kijai/ComfyUI-KJNodes

## Sync command (local)

Run from repo root:

```bash
python scripts/sync_required_custom_nodes.py --comfy-root . --mode sync
```

Optional dependency install:

```bash
python scripts/sync_required_custom_nodes.py --comfy-root . --mode sync --install-requirements
```

## Verify only

```bash
python scripts/sync_required_custom_nodes.py --comfy-root . --mode verify
```

## Output report

The script writes:

- `custom_nodes/required_nodes_sync_report.json`

Use this report as proof of parity when validating remote/local/fork setup.
