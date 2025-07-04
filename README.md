# Interview Transcription & Diarisation Pipeline

A Python-based CLI tool that transcribes German-language `.m4a` interview recordings into readable, speaker-annotated text using **WhisperX** and **pyannote.audio**. It includes word alignment, speaker diarisation.

## Features

- Audio conversion: `.m4a` → `.wav`
- Transcription with WhisperX (word-aligned)
- Speaker diarisation with pretrained pyannote pipeline
- Outputs:
  - Human-readable `.txt` transcript
  - Log file
  - .wav audio file

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/pascal-schindlmeier/interview-transcription-diarisation.git
```

### 2. Create and Activate a Virtual Environme

```bash
python -m venv nameOfYourVenv
.venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

Create a .env file in the same directory and paste your Hugging Face API token:
HF_AUTH_TOKEN=your_token_here

You can get a token from: https://huggingface.co/settings/tokens
