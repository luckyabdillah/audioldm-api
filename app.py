import logging
import os
import re
import ssl
import sys
import threading
import time
import uuid
from pathlib import Path

import certifi
import numpy as np
import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_from_directory, url_for
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

os.environ["HF_HOME"] = os.getenv("HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface"))

# AudioLDM v1 downloads checkpoints with urllib. This avoids the
# ASN1/Windows certificate-store issue seen with older Python builds.
if sys.platform == "win32":
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )

from audioldm import build_model, text_to_audio  # noqa: E402


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("audioldm-api")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs"))).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("AUDIOLDM_MODEL", "audioldm-m-full")
TRANSLATION_MODEL_NAME = os.getenv(
    "TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M"
)
ENABLE_TRANSLATION = os.getenv("ENABLE_TRANSLATION", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PROMPT_SUFFIX = os.getenv(
    "PROMPT_SUFFIX",
    "realistic high-quality field recording, clear isolated foreground sound",
).strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

# AudioLDM v1 is not safe to run concurrently on one CPU model instance.
generation_lock = threading.Lock()


def load_models():
    """Load every model exactly once when this process starts."""
    logger.info("Loading AudioLDM model: %s", MODEL_NAME)
    audio_model = build_model(model_name=MODEL_NAME)
    logger.info("AudioLDM model ready")

    translation_tokenizer = None
    translation_model = None

    if ENABLE_TRANSLATION:
        logger.info("Loading translation model: %s", TRANSLATION_MODEL_NAME)
        translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
        translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL_NAME
        ).to("cpu")
        translation_model.eval()
        logger.info("Translation model ready")

    return audio_model, translation_tokenizer, translation_model


AUDIO_MODEL, TRANSLATION_TOKENIZER, TRANSLATION_MODEL = load_models()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


def translate_to_english(text, source_language="ind_Latn"):
    if not ENABLE_TRANSLATION:
        return text

    TRANSLATION_TOKENIZER.src_lang = source_language
    encoded = TRANSLATION_TOKENIZER(text, return_tensors="pt")
    english_token_id = TRANSLATION_TOKENIZER.convert_tokens_to_ids("eng_Latn")

    with torch.inference_mode():
        generated = TRANSLATION_MODEL.generate(
            **encoded,
            forced_bos_token_id=english_token_id,
            max_length=128,
        )

    return TRANSLATION_TOKENIZER.batch_decode(
        generated, skip_special_tokens=True
    )[0]


def enhance_prompt(text):
    if not PROMPT_SUFFIX:
        return text
    return f"{text}, {PROMPT_SUFFIX}"


def parse_boolean(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError("nilai boolean must be true or false")


def validate_payload(payload):
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("'prompt' is required")
    if len(prompt) > 500:
        raise ValueError("'prompt' must be at most 500 characters")

    duration = float(payload.get("duration", 5.0))
    if duration < 2.5 or duration > 20 or abs((duration / 2.5) % 1) > 1e-9:
        raise ValueError(
            "'duration' must be a multiple of 2.5 between 2.5 and 20 seconds"
        )

    steps = int(payload.get("steps", 50))
    if not 10 <= steps <= 200:
        raise ValueError("'steps' must be between 10 and 200")

    guidance_scale = float(payload.get("guidance_scale", 2.5))
    if not 1.0 <= guidance_scale <= 10.0:
        raise ValueError("'guidance_scale' must be between 1 and 10")

    n_candidate_gen_per_text = int(payload.get("n_candidate_gen_per_text", 1))
    if not 1 <= n_candidate_gen_per_text <= 5:
        raise ValueError("'n_candidate_gen_per_text' must be between 1 and 5")

    seed = int(payload.get("seed", 42))
    translate = parse_boolean(payload.get("translate"), ENABLE_TRANSLATION)
    enhance = parse_boolean(payload.get("enhance"), True)
    source_language = str(payload.get("source_language", "ind_Latn")).strip()

    if translate and not ENABLE_TRANSLATION:
        raise ValueError("Translation is disabled on this server")

    return {
        "prompt": prompt,
        "duration": duration,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "translate": translate,
        "enhance": enhance,
        "source_language": source_language,
        "n_candidate_gen_per_text": n_candidate_gen_per_text,
    }


def build_public_url(relative_path):
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}{relative_path}"
    return url_for("serve_audio", filename=Path(relative_path).name, _external=True)


def remove_expired_outputs(max_age_hours=24):
    cutoff = time.time() - (max_age_hours * 3600)
    for path in OUTPUT_DIR.glob("*.wav"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            logger.warning("Could not remove expired output: %s", path)


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        model=MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
        translation_enabled=ENABLE_TRANSLATION,
    )


@app.post("/api/v1/generate")
def generate():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object"), 400

    try:
        params = validate_payload(payload)
    except (TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400

    started_at = time.perf_counter()

    try:
        with generation_lock:
            translated_prompt = (
                translate_to_english(
                    params["prompt"], source_language=params["source_language"]
                )
                if params["translate"]
                else params["prompt"]
            )
            final_prompt = (
                enhance_prompt(translated_prompt)
                if params["enhance"]
                else translated_prompt
            )

            waveforms = text_to_audio(
                latent_diffusion=AUDIO_MODEL,
                text=final_prompt,
                seed=params["seed"],
                ddim_steps=params["steps"],
                duration=params["duration"],
                batchsize=1,
                guidance_scale=params["guidance_scale"],
                n_candidate_gen_per_text=params["n_candidate_gen_per_text"],
            )

        audio = np.asarray(waveforms[0, 0], dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0)

        file_id = uuid.uuid4().hex
        filename = f"{file_id}.wav"
        final_path = OUTPUT_DIR / filename
        temporary_path = OUTPUT_DIR / f"{file_id}.tmp"

        sf.write(
            str(temporary_path),
            audio,
            samplerate=16000,
            format="WAV",
            subtype="PCM_16",
        )
        os.replace(str(temporary_path), str(final_path))

        remove_expired_outputs(
            max_age_hours=int(os.getenv("OUTPUT_RETENTION_HOURS", "24"))
        )

        audio_path = url_for("serve_audio", filename=filename)
        elapsed = round(time.perf_counter() - started_at, 2)

        return jsonify(
            status="completed",
            id=file_id,
            prompt=params["prompt"],
            translated_prompt=translated_prompt,
            final_prompt=final_prompt,
            duration=params["duration"],
            steps=params["steps"],
            guidance_scale=params["guidance_scale"],
            seed=params["seed"],
            generation_time_seconds=elapsed,
            audio_path=audio_path,
            audio_url=build_public_url(audio_path),
            n_candidate_gen_per_text=params["n_candidate_gen_per_text"],
        )

    except Exception as exc:
        logger.exception("Audio generation failed")
        return jsonify(error="audio generation failed", detail=str(exc)), 500


@app.get("/media/<filename>")
def serve_audio(filename):
    if not re.fullmatch(r"[0-9a-f]{32}\.wav", filename):
        return jsonify(error="invalid filename"), 404

    return send_from_directory(
        str(OUTPUT_DIR),
        filename,
        mimetype="audio/wav",
        as_attachment=False,
        conditional=True,
    )


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
        use_reloader=False,
        threaded=True,
    )
