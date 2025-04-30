# Kinyarwanda Voice Assistant

This project implements a Kinyarwanda Voice Assistant as part of the Intelligent Robotics course assignment (April 2025). The assistant uses Automatic Speech Recognition (ASR), Natural Language Processing (NLP), and Text-to-Speech (TTS) to process Kinyarwanda audio, transcribe it, match questions to predefined answers, and generate spoken responses.

## Features
- **Speech Recognition (ASR)**: Transcribes Kinyarwanda audio files using the `benax-rw/KinyaWhisper` model.
- **NLP Matching**: Matches transcribed questions to a dictionary of at least 5 predefined question-answer pairs.
- **Text-to-Speech (TTS)**: Generates spoken responses in Kinyarwanda using the `facebook/mms-tts-kin` model.
- **Output Storage**: Saves transcriptions as text files and spoken answers as WAV files.
- **Logging**: Logs all operations for debugging and monitoring.

## Requirements
- Python 3.8 or higher
- Dependencies (listed in `requirements.txt`):
  - `transformers`
  - `torchaudio`
  - `python-dotenv`
  - `huggingface_hub`
- A Hugging Face API token (stored in a `.env` file)
- At least 5 Kinyarwanda audio files (`.wav` or `.mp3`)

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory.
   - Add your Hugging Face API token:
     ```
     HUG_TOKEN=<your-huggingface-token>
     ```

4. **Prepare Audio Files**:
   - Place at least 5 Kinyarwanda audio files (`.wav` or `.mp3`) in the `audio/` folder.
   - Example questions to include: "amakuru", "witwa nde", "ubuzima bumeze gute", "ikinyarwanda ni iki", "amakuru yawe", "urashobora gukora iki".

## Usage
Run the script to process all audio files in the `audio/` folder:
```bash
python kinyarwanda_voice_assistant.py
```

### What Happens
- **Input**: Audio files in `audio/` are processed.
- **Transcription**: Each audio file is transcribed using KinyaWhisper, and the transcription is saved to `transcriptions/<filename>_transcription.txt`.
- **Question Matching**: The transcription is matched to a predefined answer using NLP.
- **Spoken Response**: If a match is found, a spoken response is generated and saved to `outputs/<filename>_answer.wav`.
- **Output**: Console displays transcriptions and answers. Logs are saved to `voice_assistant.log`.

### Example Output
```
🗣️ Transcription for audio1.wav: amakuru
💬 Answer: Ni meza, urakoze!
```

## Project Structure
- `kinyarwanda_voice_assistant.py`: Main Python script implementing the voice assistant.
- `audio/`: Folder containing at least 5 Kinyarwanda audio files (`.wav` or `.mp3`).
- `transcriptions/`: Folder with transcription text files (`<filename>_transcription.txt`).
- `outputs/`: Folder with spoken answer WAV files (`<filename>_answer.wav`).
- `voice_assistant.log`: Log file for debugging and execution tracking.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This documentation file.
- `.env`: Environment file for storing the Hugging Face API token (not tracked in Git).


## Troubleshooting
- **No Transcription Output**: Ensure audio files are clear and in Kinyarwanda. Check `voice_assistant.log` for errors.
- **Model Loading Issues**: Verify the Hugging Face API token in `.env` and internet connectivity.
- **Dependency Errors**: Run `pip install -r requirements.txt` to install all required libraries.
- **TensorFlow Warnings**: The script suppresses TensorFlow warnings by setting environment variables (`TF_ENABLE_ONEDNN_OPTS=0`, `TF_CPP_MIN_LOG_LEVEL=3`).

## Acknowledgments
- **Models**:
  - KinyaWhisper: `benax-rw/KinyaWhisper`
  - MMS-TTS: `facebook/mms-tts-kin`
- **Course**: Intelligent Robotics, Rwanda Coding Academy
- **Instructor**: Gabriel Baziramwabo
- **Assignment**: Building a Mini Kinyarwanda Voice Assistant (April 2025)

For issues or questions, please open an issue on this repository.