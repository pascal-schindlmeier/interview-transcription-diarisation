import os
import json
import torch
import logging
from pathlib import Path
from datetime import timedelta
from pydub import AudioSegment
from tqdm import tqdm
from pyannote.audio import Pipeline
import whisperx
from dotenv import load_dotenv

load_dotenv()

# Config
AUDIO_FILE = Path("input") / "interview_.m4a" # Rename .m4a according to your interview file name
LANGUAGE = "de"  # "" for auto-detect
MODEL_NAME = "large-v3"
DEVICE = "cpu"
USE_WORD_ALIGN = True
USE_DIARISATION = True
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")  # Secure hugging face token
if not HF_AUTH_TOKEN:
    raise ValueError("HF_AUTH_TOKEN not found in environment.")

# Output paths
AUDIO_FILE.parent.mkdir(parents=True, exist_ok=True)
WAV_FILE = AUDIO_FILE.with_suffix(".wav")
OUTPUT_JSON = AUDIO_FILE.with_suffix(".segments.json")
OUTPUT_TXT = AUDIO_FILE.parent / f"{AUDIO_FILE.stem}_final.txt"
LOG_FILE = AUDIO_FILE.parent / f"{AUDIO_FILE.stem}_log.log"

# Logger setup
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger()

def safe_log(msg):
    print(msg)
    logger.info(msg)

# Audio conversion
try:
    safe_log("Converting input audio to WAV...")
    audio = AudioSegment.from_file(AUDIO_FILE)
    audio.export(WAV_FILE, format="wav")
except Exception as e:
    logger.exception("Failed to convert audio to WAV.")
    raise

# Load whisperx
try:
    safe_log(f"Loading model {MODEL_NAME} on {DEVICE}...")
    model = whisperx.load_model(MODEL_NAME, device=DEVICE, compute_type="int8")
except Exception as e:
    logger.exception("Failed to load WhisperX model.")
    raise

# Transcription
try:
    safe_log("Transcribing audio...")
    transcription = model.transcribe(str(WAV_FILE), batch_size=8, language=LANGUAGE)
    logger.debug(f"Transcription result: {json.dumps(transcription, indent=2)}")
except Exception as e:
    logger.exception("Transcription failed.")
    raise

# Word alignment
if USE_WORD_ALIGN:
    try:
        safe_log("Aligning words...")
        align_model, metadata = whisperx.load_align_model(language_code=LANGUAGE or "de", device=DEVICE)
        transcription = whisperx.align(
            transcription["segments"], align_model, metadata, str(WAV_FILE), device=DEVICE
        )
        logger.debug(f"Aligned transcription: {json.dumps(transcription, indent=2)}")
    except Exception as e:
        logger.exception("Word alignment failed.")
        raise

# Diarisation
if USE_DIARISATION:
    try:
        safe_log("Diarising speakers...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_AUTH_TOKEN)
        diarization = pipeline(str(WAV_FILE))
        logger.debug("Diarization output:")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            logger.debug(f"{turn.start:.2f}s --> {turn.end:.2f}s : {speaker}")

        # Merge diarization with segments
        for segment in transcription["segments"]:
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            speaker_label = "Unknown"
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start < seg_end and turn.end > seg_start:
                    speaker_label = speaker
                    break
            segment["speaker"] = speaker_label
    except Exception as e:
        logger.exception("Speaker diarisation failed.")
        raise

# Save JSON
try:
    safe_log(f"Saving JSON output to {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f_json:
        json.dump(transcription["segments"], f_json, ensure_ascii=False, indent=2)
except Exception as e:
    logger.exception("Failed to save JSON file.")
    raise

# Save txt
try:
    safe_log("Writing readable transcript to TXT...")
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f_txt:
        segments = transcription.get("segments", [])
        safe_log(f"Total segments: {len(segments)}")
        for seg in segments:
            start = str(timedelta(seconds=int(seg.get("start", 0))))
            end = str(timedelta(seconds=int(seg.get("end", 0))))
            speaker = seg.get("speaker", "Speaker ?")
            text = seg.get("text", "").strip().replace("\n", " ")
            f_txt.write(f"[{start} – {end}] {speaker}: {text}\n")
    safe_log(f"Transcript saved to {OUTPUT_TXT}")
except Exception as e:
    logger.exception("Failed to write TXT transcript.")
    raise