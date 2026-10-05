import functools
import hmac
import logging
import os
import queue
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
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, url_for
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv(Path(__file__).resolve().parent / ".env")

os.environ["HF_HOME"] = os.getenv("HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface"))
os.environ["AUDIOLDM_CACHE_DIR"] = os.getenv("AUDIOLDM_CACHE_DIR", str(Path.home() / ".cache" / "audioldm"))

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

ENABLE_LANGUAGE_DETECTION = os.getenv(
    "ENABLE_LANGUAGE_DETECTION", "true"
).lower() in {"1", "true", "yes", "on"}
LANGUAGE_DETECTION_BACKEND = os.getenv("LANGUAGE_DETECTION_BACKEND", "lingua").lower()
DEFAULT_SOURCE_LANGUAGE = os.getenv("DEFAULT_SOURCE_LANGUAGE", "ind_Latn")
LANGUAGE_DETECTION_MIN_CONFIDENCE = float(
    os.getenv("LANGUAGE_DETECTION_MIN_CONFIDENCE", "0.55")
)

# Restricting the candidate set keeps memory low and accuracy high. Widen it
# only for languages you actually expect in prompts.
DETECTION_LANGUAGES = [
    code.strip().lower()
    for code in os.getenv("DETECTION_LANGUAGES", "id,en,jv,su,ms").split(",")
    if code.strip()
]

# ISO 639-1 -> FLORES-200 codes understood by NLLB.
ISO_TO_FLORES = {
    "id": "ind_Latn",
    "en": "eng_Latn",
    "jv": "jav_Latn",
    "su": "sun_Latn",
    "ms": "zsm_Latn",
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "nl": "nld_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
}

PROMPT_SUFFIX = os.getenv(
    "PROMPT_SUFFIX",
    "realistic high-quality field recording, clear isolated foreground sound",
).strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

# Simple API key auth. Set API_KEYS to one or more comma-separated keys,
# e.g. API_KEYS="key-for-team-a,key-for-team-b". Leave unset/empty to
# disable auth entirely (useful for local dev).
API_KEYS = {
    key.strip() for key in os.getenv("API_KEYS", "").split(",") if key.strip()
}

# AudioLDM v1 is not safe to run concurrently on one model instance, so
# generation happens on a single background worker thread that consumes
# jobs from this queue one at a time. HTTP requests never block on
# generation itself; they just enqueue a job and return immediately.
JOB_QUEUE = queue.Queue()

# In-memory job store: job_id -> job dict. Guarded by JOBS_LOCK because the
# worker thread and Flask request threads both touch it. Jobs are lost on
# process restart, which is fine here since a client can simply resubmit.
JOBS = {}
JOBS_LOCK = threading.Lock()

JOB_RETENTION_SECONDS = int(os.getenv("JOB_RETENTION_MINUTES", "60")) * 60


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


def load_language_detector():
    """Build a small language detector. Returns None when unavailable.

    Both backends are CPU-only and tiny compared to NLLB, so this adds a
    negligible amount of startup time and memory.
    """
    if not ENABLE_LANGUAGE_DETECTION:
        return None

    if LANGUAGE_DETECTION_BACKEND == "lingua":
        try:
            from lingua import LanguageDetectorBuilder, IsoCode639_1
        except ImportError:
            logger.warning(
                "lingua-language-detector is not installed; "
                "language detection disabled"
            )
            return None

        iso_codes = []
        for code in DETECTION_LANGUAGES:
            iso_code = getattr(IsoCode639_1, code.upper(), None)
            if iso_code is None:
                logger.warning("Unknown detection language ignored: %s", code)
                continue
            iso_codes.append(iso_code)

        if len(iso_codes) < 2:
            logger.warning("Need at least two detection languages; disabling")
            return None
            
        logger.info("Loading lingua detector for: %s", DETECTION_LANGUAGES)
        detector = (
            LanguageDetectorBuilder.from_iso_codes_639_1(*iso_codes)
            .with_low_accuracy_mode()
            .build()
        )
        logger.info("Language detector ready")
        return detector

    if LANGUAGE_DETECTION_BACKEND == "langid":
        try:
            import py3langid
        except ImportError:
            logger.warning("py3langid is not installed; language detection disabled")
            return None

        identifier = py3langid.langid.LanguageIdentifier.from_pickled_model(
            py3langid.langid.MODEL_FILE, norm_probs=True
        )
        identifier.set_languages(DETECTION_LANGUAGES)
        logger.info("Language detector ready (py3langid)")
        return identifier

    logger.warning("Unknown detection backend: %s", LANGUAGE_DETECTION_BACKEND)
    return None


AUDIO_MODEL, TRANSLATION_TOKENIZER, TRANSLATION_MODEL = load_models()
LANGUAGE_DETECTOR = load_language_detector()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
CORS(
    app,
    resources={
        r"/api/*": {"origins": CORS_ORIGINS},
        r"/media/*": {"origins": CORS_ORIGINS},
    },
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    methods=["GET", "POST", "OPTIONS"],
)


def _extract_api_key():
    header_key = request.headers.get("X-API-Key")
    if header_key:
        return header_key.strip()

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[len("Bearer "):].strip()

    return None


def require_api_key(view_func):
    """Protects a route with API_KEYS. No-op if API_KEYS is empty (dev mode)."""

    @functools.wraps(view_func)
    def wrapper(*args, **kwargs):
        if not API_KEYS:
            return view_func(*args, **kwargs)

        provided_key = _extract_api_key()
        if provided_key is None:
            return jsonify(error="API key required"), 401

        is_valid = any(
            hmac.compare_digest(provided_key, valid_key) for valid_key in API_KEYS
        )
        if not is_valid:
            return jsonify(error="invalid API key"), 401

        return view_func(*args, **kwargs)

    return wrapper


def detect_language(text):
    """Detect the FLORES-200 code of `text`, e.g. 'ind_Latn'.

    Returns (flores_code, confidence). Falls back to DEFAULT_SOURCE_LANGUAGE
    when detection is off, fails, or is not confident enough. Audio prompts are
    short, so a wrong-but-confident guess is worse than the default.
    """
    if LANGUAGE_DETECTOR is None:
        return DEFAULT_SOURCE_LANGUAGE, None

    cleaned = " ".join(text.split())
    if len(cleaned) < 3:
        return DEFAULT_SOURCE_LANGUAGE, None

    try:
        if LANGUAGE_DETECTION_BACKEND == "lingua":
            best = LANGUAGE_DETECTOR.compute_language_confidence_values(cleaned)[0]
            iso_code = best.language.iso_code_639_1.name.lower()
            confidence = float(best.value)
        else:
            iso_code, confidence = LANGUAGE_DETECTOR.classify(cleaned)
            confidence = float(confidence)
    except Exception:
        logger.exception("Language detection failed; using default")
        return DEFAULT_SOURCE_LANGUAGE, None

    flores_code = ISO_TO_FLORES.get(iso_code)

    if flores_code is None:
        logger.info("Detected unmapped language %s; using default", iso_code)
        return DEFAULT_SOURCE_LANGUAGE, confidence

    if confidence < LANGUAGE_DETECTION_MIN_CONFIDENCE:
        logger.info(
            "Low confidence (%s, %.2f); using default %s",
            iso_code,
            confidence,
            DEFAULT_SOURCE_LANGUAGE,
        )
        return DEFAULT_SOURCE_LANGUAGE, confidence

    return flores_code, confidence


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
    # Optional: omit it (or send "auto") to detect the language from the prompt.
    # An explicit value always wins over detection.
    raw_source_language = payload.get("source_language")
    source_language = (
        None
        if raw_source_language is None
        or str(raw_source_language).strip().lower() in {"", "auto"}
        else str(raw_source_language).strip()
    )

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
    # Generation runs outside an HTTP request, so url_for cannot infer a host.
    return relative_path


def update_job(job_id, **fields):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def remove_expired_jobs():
    """Drop finished jobs from memory after JOB_RETENTION_SECONDS.

    Keeps the JOBS dict from growing forever; the .wav files themselves
    are cleaned up separately by remove_expired_outputs().
    """
    cutoff = time.time() - JOB_RETENTION_SECONDS
    with JOBS_LOCK:
        expired = [
            job_id
            for job_id, job in JOBS.items()
            if job["status"] in {"completed", "failed"}
            and job.get("finished_at", 0) < cutoff
        ]
        for job_id in expired:
            del JOBS[job_id]


def run_generation(job_id, params):
    """Do the actual model inference for one job. Runs on the worker thread."""
    update_job(job_id, status="processing", started_at=time.perf_counter())
    started_at = time.perf_counter()

    try:
        if params["source_language"] is None:
            source_language, detection_confidence = detect_language(params["prompt"])
            language_origin = "detected"
        else:
            source_language = params["source_language"]
            detection_confidence = None
            language_origin = "client"

        logger.info(
            "Job %s: source language %s (%s, confidence=%s)",
            job_id,
            source_language,
            language_origin,
            detection_confidence,
        )

        # Skip a pointless round trip when the prompt is already English.
        if not params["translate"] or source_language == "eng_Latn":
            translated_prompt = params["prompt"]
        else:
            translated_prompt = translate_to_english(
                params["prompt"], source_language=source_language
            )
        final_prompt = (
            enhance_prompt(translated_prompt) if params["enhance"] else translated_prompt
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

        filename = f"{job_id}.wav"
        final_path = OUTPUT_DIR / filename
        temporary_path = OUTPUT_DIR / f"{job_id}.tmp"

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

        elapsed = round(time.perf_counter() - started_at, 2)

        audio_path = f"/media/{filename}"
        audio_url = build_public_url(audio_path)

        update_job(
            job_id,
            status="completed",
            finished_at=time.time(),
            result={
                "id": job_id,
                "prompt": params["prompt"],
                "translated_prompt": translated_prompt,
                "final_prompt": final_prompt,
                "duration": params["duration"],
                "steps": params["steps"],
                "guidance_scale": params["guidance_scale"],
                "seed": params["seed"],
                "source_language": source_language,
                "source_language_origin": language_origin,
                "source_language_confidence": detection_confidence,
                "generation_time_seconds": elapsed,
                "audio_path": audio_path,
                "audio_url": audio_url,
                "n_candidate_gen_per_text": params["n_candidate_gen_per_text"],
            },
        )

    except Exception as exc:
        logger.exception("Job %s: audio generation failed", job_id)
        update_job(
            job_id,
            status="failed",
            finished_at=time.time(),
            error="audio generation failed",
            detail=str(exc),
        )


def worker_loop():
    """Runs forever on a background thread, processing one job at a time."""
    logger.info("Generation worker thread started")
    while True:
        job_id, params = JOB_QUEUE.get()
        try:
            run_generation(job_id, params)
        finally:
            JOB_QUEUE.task_done()
            remove_expired_jobs()


# Single background worker: the model can only do one generation at a time
# anyway, so one thread is all we need. daemon=True so it doesn't block
# process shutdown.
threading.Thread(target=worker_loop, daemon=True, name="audioldm-worker").start()


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
        language_detection_enabled=LANGUAGE_DETECTOR is not None,
        language_detection_backend=(
            LANGUAGE_DETECTION_BACKEND if LANGUAGE_DETECTOR is not None else None
        ),
        queue_size=JOB_QUEUE.qsize(),
        auth_enabled=bool(API_KEYS),
    )


@app.post("/api/v1/generate")
@require_api_key
def generate():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object"), 400

    try:
        params = validate_payload(payload)
    except (TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400

    job_id = uuid.uuid4().hex
    queue_position = JOB_QUEUE.qsize()

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "params": params,
        }

    JOB_QUEUE.put((job_id, params))

    logger.info("Job %s queued (position %s)", job_id, queue_position)

    return jsonify(
        status="queued",
        id=job_id,
        queue_position=queue_position,
        status_url=url_for("job_status", job_id=job_id),
    ), 202


@app.get("/api/v1/jobs/<job_id>")
@require_api_key
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if job is None:
        return jsonify(error="job not found"), 404

    response = {"id": job["id"], "status": job["status"]}

    if job["status"] == "completed":
        response.update(job["result"])
    elif job["status"] == "failed":
        response["error"] = job["error"]
        response["detail"] = job["detail"]
    elif job["status"] == "queued":
        # Position among jobs still waiting (0 = next up).
        response["queue_position"] = sum(
            1
            for j in JOBS.values()
            if j["status"] == "queued" and j["created_at"] < job["created_at"]
        )

    return jsonify(**response)


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