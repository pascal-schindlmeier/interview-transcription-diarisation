# Interview Transcription & Diarisation Pipeline

A Python-based CLI tool that transcribes German-language `.m4a` interview recordings into readable, speaker-annotated text using **WhisperX** and **pyannote.audio**. It includes optional word alignment, speaker diarisation, and outputs in both `.txt` and `.json` formats.

## Features

- Audio conversion: `.m4a` → `.wav`
- Transcription with WhisperX (word-aligned)
- Speaker diarisation with pretrained pyannote pipeline
- Outputs:
  - Human-readable `.txt` transcript
  - JSON transcript with segment data
  - Speaker logs

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/interview-transcription-diarisation.git
cd interview-transcription-diarisation-pipeline
```

### 2. Create and Activate a Virtual Environme

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
```
Open .env and paste your Hugging Face API token:
HF_AUTH_TOKEN=your_token_here

You can get a token from: https://huggingface.co/settings/tokens
