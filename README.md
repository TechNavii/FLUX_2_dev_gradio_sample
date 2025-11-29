# FLUX.2 Gradio Demo

Small Gradio UI for the `black-forest-labs/FLUX.2-dev` diffusers pipeline. Defaults to the remote text encoder to save memory; can run fully offline if all weights are cached.

## Requirements
- Python 3.10+
- Hugging Face login with access to `black-forest-labs/FLUX.2-dev` (`huggingface-cli login` or set `HF_TOKEN`/`HUGGINGFACEHUB_API_TOKEN`)

## Setup
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```sh
# Uses env vars (default: remote encoder)
python app.py

# Force remote encoder (internet required)
python app_online.py

# Force local encoder (needs cached weights)
python app_offline.py
```
Open the Gradio URL printed in the console (defaults to http://127.0.0.1:7860) and enter a prompt. Use a negative seed (e.g., `-1`) to randomize.

## Configuration
- `FLUX_USE_REMOTE_ENCODER` = `0/1` to choose offline/online mode.
- `FLUX_REPO_ID` to switch checkpoints.
- `FLUX_DEVICE` (`cuda`/`mps`/`cpu`) and `FLUX_DTYPE` (`float32`/`float16`/`bfloat16`) to override defaults.
- `GRADIO_SERVER_NAME` or `GRADIO_SHARE` to change hosting behavior.

## Troubleshooting
- `ImportError: cannot import name 'Flux2Pipeline'`: install diffusers from GitHub main:
  ```sh
  pip install -U --force-reinstall 'git+https://github.com/huggingface/diffusers.git'
  pip install -U transformers accelerate huggingface_hub
  ```
- Gradio schema errors: reinstall the pinned versions in `requirements.txt` (gradio==4.39.0, gradio_client==1.3.0).
