"""Audio clips + the optional "deep listen" path.

Text RAG finds *where* something was said; the audio-native model can then be
pointed at that exact clip to answer questions transcripts can't — tone, music,
sound design, who is laughing, what the archival tape sounds like.

Backends for the audio model (RAGIO_AUDIO_BACKEND):
  * openai-compatible — e.g. `vllm serve Qwen/Qwen2-Audio-7B-Instruct` (Linux/NVIDIA)
  * transformers      — runs Qwen2-Audio in-process (works on Apple Silicon via MPS, slowly)
  * none              — /api/listen returns 501
"""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from . import config, db


def cut_clip(episode_id: str, start: float, end: float, fmt: str = "mp3", pad: float = 0.5) -> Path:
    """Cut [start, end] (seconds) from the episode audio with ffmpeg; cached on disk."""
    with db.tx() as conn:
        ep = db.get_episode(conn, episode_id)
    if not ep or not ep["audio_path"]:
        raise FileNotFoundError(f"no audio for episode {episode_id}")
    start, end = max(0.0, start - pad), end + pad
    out = config.CLIP_DIR / f"{episode_id}_{start:.1f}_{end:.1f}.{fmt}"
    if out.exists():
        return out
    codec = ["-c:a", "libmp3lame", "-b:a", "96k"] if fmt == "mp3" else ["-ar", "16000", "-ac", "1"]
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
         "-i", ep["audio_path"], *codec, str(out)],
        check=True,
    )
    return out


def listen(clip: Path, prompt: str) -> str:
    backend = config.AUDIO_LLM_BACKEND
    if backend == "openai-compatible":
        return _listen_openai_compatible(clip, prompt)
    if backend == "transformers":
        return _listen_transformers(clip, prompt)
    raise NotImplementedError("Set RAGIO_AUDIO_BACKEND=openai-compatible or transformers to enable /api/listen")


def _listen_openai_compatible(clip: Path, prompt: str) -> str:
    import httpx

    wav = clip if clip.suffix == ".wav" else _to_wav(clip)
    payload = {
        "model": config.AUDIO_LLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav.read_bytes()).decode(), "format": "wav"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": 400,
    }
    r = httpx.post(f"{config.AUDIO_LLM_URL}/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


_tf = {}


def _listen_transformers(clip: Path, prompt: str) -> str:
    import librosa
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    if not _tf:
        _tf["proc"] = AutoProcessor.from_pretrained(config.AUDIO_LLM_MODEL)
        _tf["model"] = Qwen2AudioForConditionalGeneration.from_pretrained(config.AUDIO_LLM_MODEL, device_map="auto")
    proc, model = _tf["proc"], _tf["model"]
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio_url": str(clip)}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audio, _ = librosa.load(str(clip), sr=proc.feature_extractor.sampling_rate)
    inputs = proc(text=text, audios=[audio], return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**inputs, max_new_tokens=400)
    out = out[:, inputs.input_ids.size(1):]
    return proc.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _to_wav(clip: Path) -> Path:
    wav = clip.with_suffix(".wav")
    if not wav.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(clip), "-ar", "16000", "-ac", "1", str(wav)],
                       check=True)
    return wav
