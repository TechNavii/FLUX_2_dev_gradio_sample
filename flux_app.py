import io
import os
import random
import time
from typing import Optional, Tuple

import gradio as gr
import gradio_client.utils as grc_utils
import requests
import torch
try:
    from diffusers import Flux2Pipeline
except ImportError:
    # Defer hard failure to runtime with a clearer hint.
    Flux2Pipeline = None
from huggingface_hub import get_token

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

DEFAULT_REPO_ID = "black-forest-labs/FLUX.2-dev"
_pipe: Optional[Flux2Pipeline] = None
_pipe_key: Optional[Tuple[str, str, str, bool]] = None


_orig_json_schema_to_python_type = getattr(grc_utils, "_json_schema_to_python_type", None)


def _json_schema_to_python_type_safe(schema, defs=None):
    if isinstance(schema, bool):
        return "bool" if schema else "null"
    if _orig_json_schema_to_python_type is None:
        return "unknown"
    return _orig_json_schema_to_python_type(schema, defs)


if _orig_json_schema_to_python_type:
    grc_utils._json_schema_to_python_type = _json_schema_to_python_type_safe




def _select_device():
    preferred = os.environ.get("FLUX_DEVICE", "").lower()
    auto_device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if preferred in {"cuda", "mps", "cpu"}:
        if preferred == "cuda" and not torch.cuda.is_available():
            device = auto_device
        elif preferred == "mps" and not torch.backends.mps.is_available():
            device = auto_device
        else:
            device = preferred
    else:
        device = auto_device

    allow_mps = os.environ.get("FLUX_ALLOW_MPS", "1").lower() in {"1", "true", "yes"}
    if device == "mps" and not allow_mps:
        device = "cpu"

    if device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    dtype_env = os.environ.get("FLUX_DTYPE", "").lower()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    default_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch_dtype = dtype_map.get(dtype_env, default_dtype)
    return device, torch_dtype


def _ensure_token():
    token = get_token()
    if token is None:
        raise RuntimeError(
            "Hugging Face token is required. Run `huggingface-cli login` or set HF_TOKEN/HUGGINGFACEHUB_API_TOKEN."
        )
    return token


def _ensure_pipe(repo_id, device, torch_dtype, use_remote_encoder, token):
    global _pipe, _pipe_key
    key = (repo_id, device, str(torch_dtype), use_remote_encoder)
    if _pipe is not None and _pipe_key == key:
        return _pipe

    if Flux2Pipeline is None:
        raise ImportError(
            "`Flux2Pipeline` not found in your diffusers install. "
            "Install/upgrade from GitHub main:\n"
            "pip install -U --force-reinstall 'git+https://github.com/huggingface/diffusers.git'\n"
            "pip install -U transformers accelerate huggingface_hub"
        )

    pipe_kwargs = {"torch_dtype": torch_dtype, "token": token}
    if use_remote_encoder:
        pipe_kwargs["text_encoder"] = None  # use remote endpoint for embeddings

    pipe = Flux2Pipeline.from_pretrained(repo_id, **pipe_kwargs)

    if device == "cuda":
        pipe.enable_model_cpu_offload()
    elif device == "mps":
        pipe.to(device, torch_dtype)
    else:
        pipe.to("cpu", torch_dtype)

    _pipe = pipe
    _pipe_key = key
    return pipe


def _remote_text_encoder(prompt: str, token: str, device: str, torch_dtype, url: str):
    try:
        response = requests.post(
            url,
            json={"prompt": prompt},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=90,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise gr.Error(f"Remote text encoder failed: {exc}") from exc

    prompt_embeds = torch.load(io.BytesIO(response.content), map_location=device)
    return prompt_embeds.to(device, dtype=torch_dtype)


def launch_app(use_remote_encoder: Optional[bool] = None, repo_id: Optional[str] = None):
    use_remote_encoder = (
        use_remote_encoder
        if use_remote_encoder is not None
        else os.environ.get("FLUX_USE_REMOTE_ENCODER", "1").lower()
        not in {"0", "false", "no"}
    )
    repo_id = repo_id or os.environ.get("FLUX_REPO_ID", DEFAULT_REPO_ID)

    device, torch_dtype = _select_device()
    token = _ensure_token()
    remote_encoder_url = os.environ.get(
        "FLUX_REMOTE_ENCODER_URL", "https://remote-text-encoder-flux-2.huggingface.co/predict"
    )

    def generate_image(
        prompt: str,
        steps: int,
        guidance: float,
        seed: int,
        progress: gr.Progress = gr.Progress(track_tqdm=True),
    ):
        if not prompt or not prompt.strip():
            raise gr.Error("Please provide a prompt.")

        pipe = _ensure_pipe(repo_id, device, torch_dtype, use_remote_encoder, token)
        prompt_embeds = (
            _remote_text_encoder(prompt, token, device, torch_dtype, remote_encoder_url)
            if use_remote_encoder
            else None
        )

        gen_device = device if device in {"cuda", "mps"} else "cpu"
        generator = torch.Generator(device=gen_device)

        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**31 - 1)
        generator.manual_seed(int(seed))

        total_steps = int(steps)
        progress(0, desc="Starting inference...")

        start = time.time()
        image = pipe(
            prompt_embeds=prompt_embeds,
            prompt=None if prompt_embeds is not None else prompt,
            generator=generator,
            num_inference_steps=total_steps,
            guidance_scale=float(guidance),
        ).images[0]
        duration = time.time() - start
        stats = (
            f"Seed={seed} | Steps={total_steps} | Guidance={guidance} | "
            f"Device={device} | Dtype={torch_dtype} | Mode={mode} | Time={duration:.1f}s"
        )
        progress(1.0, desc="Done")
        return image.convert("RGB"), stats

    mode = "remote text encoder (internet)" if use_remote_encoder else "offline text encoder"
    with gr.Blocks(title="FLUX.2-dev with Gradio", analytics_enabled=False) as demo:
        gr.Markdown(
            "### FLUX.2-dev (diffusers)\n"
            f"- Model repo: `{repo_id}`\n"
            "- Requires access to the gated repository and a Hugging Face login.\n"
            f"- Running on `{device}` with dtype `{torch_dtype}`. Mode: **{mode}**.\n"
            "- Generation will be slow on CPU and MPS."
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A cinematic photo of a futuristic city at dusk...",
                    lines=4,
                )
                steps = gr.Slider(10, 60, value=28, step=1, label="Steps")
                guidance = gr.Slider(
                    0.0, 10.0, value=4.0, step=0.1, label="Guidance scale"
                )
                seed = gr.Number(value=42, precision=0, label="Seed (use -1 for random)")
                run = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                output = gr.Image(type="pil", format="png", image_mode="RGB", label="Generated image")
                stats_box = gr.Textbox(label="Run stats", lines=2, interactive=False)

        run.click(fn=generate_image, inputs=[prompt, steps, guidance, seed], outputs=[output, stats_box])

    # Avoid Gradio API schema generation and the queue; bind address is configurable.
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    share_env = os.environ.get("GRADIO_SHARE")
    if share_env is not None:
        share = share_env.lower() in {"1", "true", "yes"}
    else:
        share = server_name not in {"127.0.0.1", "localhost"}

    try:
        demo.launch(
            show_api=False,
            server_name=server_name,
            share=share,
            inline=False,
        )
    except ValueError as exc:
        if "localhost is not accessible" in str(exc) and not share:
            demo.launch(
                show_api=False,
                server_name=server_name,
                share=True,
                inline=False,
            )
        else:
            raise
