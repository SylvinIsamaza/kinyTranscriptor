import os
import logging
import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration, VitsModel, VitsTokenizer
from huggingface_hub import login
from dotenv import load_dotenv

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KinyarwandaVoiceAssistant:
    """A class to implement a Kinyarwanda voice assistant with ASR, NLP, and TTS."""

    def __init__(self):
        """Initialize the voice assistant with models and configurations."""
        try:
            # Load environment variables
            load_dotenv()
            api_token = os.getenv('HUG_TOKEN')
            if not api_token:
                raise ValueError("Hugging Face API token not found in environment variables.")
            login(api_token)
            logger.info("Successfully logged into Hugging Face Hub.")

            # Initialize ASR model and processor
            self.asr_model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
            self.asr_processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")
            logger.info("KinyaWhisper ASR model and processor initialized.")

            # Initialize TTS model and tokenizer
            self.tts_model = VitsModel.from_pretrained("facebook/mms-tts-kin")
            self.tts_tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kin")
            logger.info("TTS model and tokenizer initialized.")

            # Define QA pairs (minimum 5 as per assignment)
            self.qa_pairs = {
                "amakuru": "Ni meza, urakoze!",
                "witwa nde": "Nitwa ISAMAZA",
                "ubuzima bumeze gute": "Bumeze neza!",
                "ikinyarwanda ni iki": "Ni ururimi kavukire ruvugwa nAbanyarwanda.",
                "amakuru yawe": "Ni meza, ndashima Imana!",
                "urashobora gukora iki": "Nshobora gufasha kuvuga no kumva mu Kinyarwanda!"
            }
            logger.info("QA pairs initialized with %d entries.", len(self.qa_pairs))

            # Set up folders
            self.transcription_folder = 'transcriptions/'
            self.output_folder = 'outputs/'
            os.makedirs(self.transcription_folder, exist_ok=True)
            os.makedirs(self.output_folder, exist_ok=True)
            logger.info("Folders ensured: %s, %s", self.transcription_folder, self.output_folder)

        except Exception as e:
            logger.error("Initialization failed: %s", str(e))
            raise

    def transcribe_audio(self, file_path: str) -> str:
        """
        Transcribe an audio file using KinyaWhisper and save the transcription.

        Args:
            file_path (str): Path to the input audio file.

        Returns:
            str: The transcribed text.
        """
        try:
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(file_path)
            logger.info("Loaded audio file: %s with sample rate %d Hz", file_path, sample_rate)

            # Convert stereo to mono if necessary
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
                logger.info("Converted stereo audio to mono.")

            # Resample to 16000 Hz if necessary
            if sample_rate != 16000:
                resampler = Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                logger.info("Resampled audio from %d Hz to 16000 Hz.", sample_rate)

            # Prepare input for the model
            inputs = self.asr_processor(waveform, sampling_rate=16000, return_tensors="pt")
            logger.info("Prepared audio inputs for transcription.")

            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.asr_model.generate(
                    inputs["input_features"],
                    max_new_tokens=100,
                    no_repeat_ngram_size=1,
                    suppress_tokens=[]
                )
            transcription = self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            logger.info("Transcribed text: %s", transcription)

            # Save transcription to a text file
            transcription_file = os.path.join(
                self.transcription_folder,
                f"{os.path.splitext(os.path.basename(file_path))[0]}_transcription.txt"
            )
            with open(transcription_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            logger.info("Transcription saved to: %s", transcription_file)

            return transcription

        except Exception as e:
            logger.error("Error transcribing audio file %s: %s", file_path, str(e))
            return ""

    def match_question(self, transcription: str) -> str:
        """
        Match the transcribed text to a predefined answer using simple NLP.

        Args:
            transcription (str): The transcribed text from the audio.

        Returns:
            str: The matched answer or an empty string if no match is found.
        """
        try:
            recognized_text = transcription.lower().strip()
            matched_answer = next(
                (answer for question, answer in self.qa_pairs.items() if question in recognized_text),
                ""
            )
            if matched_answer:
                logger.info("Matched answer: %s", matched_answer)
            else:
                logger.warning("No matching answer found for transcription: %s", recognized_text)
            return matched_answer

        except Exception as e:
            logger.error("Error matching question: %s", str(e))
            return ""

    def generate_speech(self, text: str, output_file: str) -> None:
        """
        Generate and save speech from text using the TTS model.

        Args:
            text (str): Text to convert to speech.
            output_file (str): Path to save the generated audio file.
        """
        try:
            inputs = self.tts_tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                speech = self.tts_model(**inputs).waveform

            torchaudio.save(output_file, speech, sample_rate=self.tts_model.config.sampling_rate)
            logger.info("Generated speech saved to: %s", output_file)

        except Exception as e:
            logger.error("Error generating speech for text '%s': %s", text, str(e))

    def process_audio_file(self, file_path: str) -> None:
        """
        Process an audio file through ASR, NLP, and TTS stages.

        Args:
            file_path (str): Path to the input audio file.
        """
        try:
            # Step 1: Transcribe audio
            transcription = self.transcribe_audio(file_path)
            if not transcription:
                logger.error("Transcription failed for %s, skipping further processing.", file_path)
                return

            print(f"🗣️ Transcription for {os.path.basename(file_path)}: {transcription}")

            # Step 2: Match question to answer
            matched_answer = self.match_question(transcription)
            if not matched_answer:
                print("No matching answer found.")
                return

            print(f"💬 Answer: {matched_answer}")

            # Step 3: Generate spoken response
            output_file = os.path.join(
                self.output_folder,
                f"{os.path.splitext(os.path.basename(file_path))[0]}_answer.wav"
            )
            self.generate_speech(matched_answer, output_file)

        except Exception as e:
            logger.error("Error processing audio file %s: %s", file_path, str(e))

def main():
    """Main function to process audio files in the audio folder."""
    try:
        assistant = KinyarwandaVoiceAssistant()
        audio_folder = 'audio/'

        if not os.path.exists(audio_folder):
            logger.error("Audio folder not found: %s", audio_folder)
            return

        audio_files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3'))]
        if len(audio_files) < 5:
            logger.warning("Fewer than 5 audio files found. Assignment requires at least 5.")

        for file_name in audio_files:
            file_path = os.path.join(audio_folder, file_name)
            logger.info("Processing file: %s", file_path)
            assistant.process_audio_file(file_path)

    except Exception as e:
        logger.error("Main execution failed: %s", str(e))

if __name__ == "__main__":
    main()