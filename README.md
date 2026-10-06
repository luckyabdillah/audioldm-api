# AudioLDM v1 Flask API
​
A ready-to-use Flask API for the first version of AudioLDM. AudioLDM and NLLB are loaded once when the Flask/Gunicorn process starts. The generation endpoint saves a WAV file and returns JSON containing the file URL, not base64 or a raw audio body.
​
## API behavior

```text
POST /api/v1/generate -> run inference -> save WAV -> JSON audio_url
GET  /media/<id>.wav  -> serve the WAV file
GET  /health          -> health check
GET  /api/v1/queue    -> current queue status
GET  /api/v1/history  -> completed generations history
```
​
Generation runs in a single background worker. Requests are accepted until the global active-job limit is reached; additional requests receive `429 Too Many Requests`. On CPU, a job can take several minutes. Gunicorn is configured with a single worker so that the large checkpoints are not loaded multiple times, and with a 30-minute timeout.
​
## 1. Directory structure
​
The source code does not need to live inside the Conda environment.
​
Windows:
​
```text
Example
R:\luckyabdillah\projects\audioldm-api\
    app.py
    requirements.txt
    gunicorn.conf.py
    outputs\
​
C:\Users\Administrator\.conda\envs\audioldm\
    python.exe
    Lib\site-packages\audioldm\
```
​
Linux VPS:
​
```text
/srv/audioldm-api/                 # source
/opt/miniconda3/envs/audioldm/     # environment
/var/lib/audioldm/cache/           # checkpoints
/var/lib/audioldm/outputs/         # generated WAV files
```
​
## 2. Installation on Windows

### 2.1 Prerequisites: AudioLDM v1 and PyTorch

`requirements.txt` only contains the *additional* packages this API needs. It deliberately does **not** install AudioLDM or PyTorch, because those depend on your hardware (CPU vs. a specific CUDA version) and pinning them here would break other setups.

Install them first, following the upstream [AudioLDM repository](https://github.com/haoheliu/AudioLDM) instructions:

```powershell
# Optional
conda create -n audioldm python=3.8; conda activate audioldm
# Install AudioLDM
pip3 install git+https://github.com/haoheliu/AudioLDM.git
```

Verify the environment before continuing:

```powershell
python -c "import torch, audioldm; print(torch.__version__, torch.cuda.is_available())"
```

The AudioLDM and NLLB checkpoints are **not** installed manually. They are
downloaded automatically on first run and cached under `HF_HOME` and `AUDIOLDM_CACHE_DIR` (configure the path in `.env`). Expect several GB and a slow first startup.

### 2.2 Install the API packages and run

With the AudioLDM v1 environment active:

```powershell
conda activate audioldm
cd D:\your-directory\audioldm-api
python -m pip install -r requirements.txt
python app.py
```

This also installs `lingua-language-detector`, used to auto-detect the prompt language. It is small and CPU-only, and only the languages listed in
`DETECTION_LANGUAGES` are loaded into memory. If it is missing, the API still starts and simply falls back to `DEFAULT_SOURCE_LANGUAGE`.

The development server runs at `http://127.0.0.1:5000`. Confirm it is up with `GET /health`, which also reports whether language detection is active.

Do not run it with the debug reloader: `app.py` can be imported twice, which
would load both models twice.
​
## 3. Generation request
​
PowerShell:
​
```powershell
$body = @{
    prompt = "hujan deras di atap seng"
    translate = $true
    enhance = $true
    duration = 5
    steps = 50
    guidance_scale = 2.5
    seed = 42
} | ConvertTo-Json
​
Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/api/v1/generate" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```
​
Example response:
​
```json
{
  "status": "completed",
  "id": "d6fef84b8f994cd79883bf5c820f99ac",
  "prompt": "hujan deras di atap seng",
  "translated_prompt": "heavy rain on a metal roof",
  "final_prompt": "heavy rain on a metal roof, realistic high-quality field recording, clear isolated foreground sound",
  "duration": 5.0,
  "steps": 50,
  "guidance_scale": 2.5,
  "seed": 42,
  "source_language": "ind_Latn",
  "source_language_origin": "detected",
  "source_language_confidence": 0.98,
  "generation_time_seconds": 242.73,
  "audio_path": "/media/d6fef84b8f994cd79883bf5c820f99ac.wav",
  "audio_url": "http://127.0.0.1:5000/media/d6fef84b8f994cd79883bf5c820f99ac.wav"
}
```
​
Download the result:
​
```powershell
Invoke-WebRequest `
    -Uri "http://127.0.0.1:5000/media/d6fef84b8f994cd79883bf5c820f99ac.wav" `
    -OutFile "result.wav"
```
​
### Parameters
​
| Parameter | Default | Description |
| --- | ---: | --- |
| `prompt` | required | Maximum 500 characters |
| `translate` | `true` | Translate from `source_language` into English |
| `enhance` | `true` | Append a recording-quality description |
| `source_language` | auto-detected | FLORES-200 code such as `ind_Latn`. Omit it, or send `"auto"`, to detect it from the prompt |
| `duration` | `5` | Multiple of 2.5, maximum 20 seconds |
| `steps` | `50` | 10 to 200 |
| `guidance_scale` | `2.5` | 1 to 10 |
| `seed` | `42` | Integer seed |


`n_candidate_gen_per_text` is intentionally locked to `1`. AudioLDM v1 calls
`waveform.cuda()` when scoring multiple candidates, so it crashes on a CPU-only
PyTorch build.

### Automatic language detection
​
When `source_language` is omitted, the prompt language is detected and mapped to
the matching FLORES-200 code before translation. An explicit `source_language`
always wins, so existing clients keep working unchanged.

The response reports what happened:
​
- `source_language` — the code actually used
- `source_language_origin` — `detected` or `client`
- `source_language_confidence` — `null` when the client supplied the code

​
Detection falls back to `DEFAULT_SOURCE_LANGUAGE` when the prompt is shorter than 3 characters, the detected language is not in the mapping table, confidence is below `LANGUAGE_DETECTION_MIN_CONFIDENCE`, or the detector is unavailable. Audio prompts are short, so a confident wrong guess is worse than the default. When the prompt is detected as English, the NLLB round trip is skipped entirely.
​
Two backends are available:
​
| Backend | Install | Notes |
| --- | --- | --- |
| `lingua` (default) | `lingua-language-detector` | Built for short text; loads only the languages in `DETECTION_LANGUAGES` |
| `langid` | `py3langid` | ~2 MB, pure NumPy, weaker on very short prompts |


Keep `DETECTION_LANGUAGES` as narrow as realistic. A short prompt like
`suara hujan` is much easier to classify against 5 candidates than against 75.
​
## 4. Environment configuration
​
| Variable | Default |
| --- | --- |
| `AUDIOLDM_MODEL` | `audioldm-m-full` |
| `ENABLE_TRANSLATION` | `true` |
| `TRANSLATION_MODEL` | `facebook/nllb-200-distilled-600M` |
| `ENABLE_LANGUAGE_DETECTION` | `true` |
| `LANGUAGE_DETECTION_BACKEND` | `lingua` (or `langid`) |
| `DETECTION_LANGUAGES` | `id,en,jv,su,ms` (ISO 639-1, comma separated) |
| `DEFAULT_SOURCE_LANGUAGE` | `ind_Latn` |
| `LANGUAGE_DETECTION_MIN_CONFIDENCE` | `0.55` |
| `PROMPT_SUFFIX` | high-quality recording description |
| `OUTPUT_DIR` | `outputs` folder next to `app.py` |
| `OUTPUT_RETENTION_HOURS` | `24` |
| `HISTORY_FILE` | `outputs/history.json` |
| `MAX_QUEUE_SIZE` | `10` (queued + processing jobs) |
| `PUBLIC_BASE_URL` | host taken from the request |
| `CORS_ORIGINS` | `*` (comma-separated; set to your frontend origin(s) in production) |
| `HOST` | `127.0.0.1` |
| `PORT` | `5000` |

If you run behind a reverse proxy or custom domain, set the public URL:
​
```bash
export PUBLIC_BASE_URL="https://audio.example.com"
```

### Queue limit and status

`MAX_QUEUE_SIZE` limits the total number of active jobs, including the job
currently being processed. When the limit is reached, `POST /api/v1/generate`
returns `429`:

```json
{
  "error": "job queue is full",
  "message": "Try again when an existing job completes",
  "queued": 10,
  "processing": 1,
  "total": 11,
  "max_queue_size": 11
}
```

Use `GET /api/v1/queue` to inspect the current state:

```json
{
  "status": "ok",
  "queued": 10,
  "processing": 1,
  "total": 11,
  "max_queue_size": 11,
  "available_slots": 0
}
```

The queue endpoint follows the same API-key authentication as the generation
and job-status endpoints.

### Generation history

`GET /api/v1/history` returns completed generations from the last
`OUTPUT_RETENTION_HOURS` hours. The history is persisted in the JSON file at
`HISTORY_FILE`, which defaults to `outputs/history.json`. Entries are written
when generation completes and expired entries are removed when a new entry is
saved.

Example response:

```json
{
  "status": "ok",
  "count": 1,
  "retention_hours": 24,
  "history": [
    {
      "id": "d6fef84b8f994cd79883bf5c820f99ac",
      "prompt": "hujan deras di atap seng",
      "audio_url": "http://127.0.0.1:5000/media/d6fef84b8f994cd79883bf5c820f99ac.wav",
      "finished_at": 1760000000.0
    }
  ]
}
```
​
## 5. Gunicorn and systemd deployment
​
A Windows environment cannot be copied to Linux. Recreate the AudioLDM
environment on the VPS, place the source in `/srv/audioldm-api`, then install the
API packages:
​
```bash
/opt/miniconda3/envs/audioldm/bin/python -m pip install -r requirements.txt
```
​
Test Gunicorn:
​
```bash
cd /srv/audioldm-api
/opt/miniconda3/envs/audioldm/bin/gunicorn -c gunicorn.conf.py app:app
```
​
Install `audioldm-api.service`:
​
```bash
sudo cp audioldm-api.service /etc/systemd/system/
sudo mkdir -p /var/lib/audioldm/cache /var/lib/audioldm/outputs
sudo chown -R audioldm:audioldm /srv/audioldm-api /var/lib/audioldm
sudo systemctl daemon-reload
sudo systemctl enable --now audioldm-api
sudo journalctl -u audioldm-api -f
```
​
Adjust the `/opt/miniconda3` path and the `audioldm` user if yours differ.
​
## 6. Production notes
​
- Serving WAV files is more efficient than base64 in JSON. Base64 inflates the
  payload and makes caching and range requests harder.
- Files are served by Flask for prototyping. In production, Nginx or object
  storage is more efficient for `/media/`.
- The synchronous endpoint can exceed reverse-proxy timeouts. Match the Nginx
  timeout to Gunicorn's, or move generation to a job queue.
- Add authentication, rate limiting, quota limits, and HTTPS before exposing the
  API to the internet.
- NLLB 600M and AudioLDM medium use a lot of RAM. If translation is not needed,
  set `ENABLE_TRANSLATION=false`.
- Language detection is cheap compared to the other models, but if every prompt
  is already in one known language, set `ENABLE_LANGUAGE_DETECTION=false` and
  rely on `DEFAULT_SOURCE_LANGUAGE` instead.
- Log `source_language_confidence` in production for a while. If low-confidence
  fallbacks are frequent, narrow `DETECTION_LANGUAGES` before raising the
  threshold.
- `PUBLIC_BASE_URL` must use a public HTTPS domain so that `audio_url` is correct
  when the API sits behind Nginx.