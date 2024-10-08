## Objective

This project demonstrates the use of OpenAI’s Whisper model for transcribing and translating audio files. It includes functionality for speaker diarization and text-to-speech (TTS) conversion using Google Text-to-Speech (gTTS). Additionally, the project provides a user-friendly interface using Gradio, making it easy to interact with these functionalities.


## 1: Install Dependencies

This section handles the installation of the required libraries necessary for transcription, translation, speaker diarization, and the user interface.

- **OpenAI's Whisper Library**: Whisper is the primary model used for transcription and translation.
- **Pyannote.audio**: This library is used for speaker diarization to separate different speakers in an audio file.
- **Gradio**: Gradio provides an interactive interface for users to upload audio files and view transcriptions.
- **gTTS**: Google Text-to-Speech (gTTS) is used for converting text transcriptions into spoken audio.


## 2: Import Required Libraries

This section imports all necessary Python libraries for performing transcription, translation, audio processing, and generating a UI interface.

- **torch** and **torchaudio**: Used for tensor computations and handling audio data.
- **Whisper**: The primary model used for speech recognition and transcription.
- **Google Text-to-Speech (gTTS)**: Converts text to speech.
- **Pyannote**: Enables speaker diarization.
- **Gradio**: Used to create an interactive interface for audio input and output.


## 3: Whisper Model Setup

### 3.1: Loading Whisper Model

In this step, the Whisper model is loaded. The model is pre-trained for transcription and translation tasks.

- **Whisper Model**: Loaded to handle the transcription process.
- **Device Configuration**: The model automatically identifies whether it runs on a GPU or CPU.


### 3.2: Test the Whisper Model with Audio

This subsection demonstrates how to test the model with a sample audio file. The steps include:

- **Downloading a Sample Audio File**: The sample audio file is used to test the Whisper model's transcription capabilities.
- **Spectrogram Generation**: The audio file is transformed into a log-Mel spectrogram for transcription.
- **Language Detection**: Whisper can automatically detect the spoken language in the audio.
- **Transcription**: The model transcribes the spoken content in the audio.


## 4: Transcription and Translation Functions

### 4.1: Transcription Function

This function handles the transcription and translation process. It takes an audio file and translates it into the desired target language. 

- **Language Mapping**: A dictionary that maps different language names to language codes is used for translations.
- **Transcription**: The audio file is transcribed using Whisper.
- **Translation**: The transcribed text can be translated into different languages using Whisper and gTTS.

### 4.2: Testing the Function with a French Audio File

In this subsection, we demonstrate how the function transcribes a French audio file and translates it into English text and speech.


## 5: Using Whisper through OpenAI API

### 5.1: API Authentication

Here, the code authenticates with the OpenAI API using a stored API key. This key allows interaction with OpenAI services such as Whisper.


### 5.2: Create OpenAI Client

An OpenAI client is created to communicate with the Whisper API for transcription and translation tasks.


### 5.3: Using Whisper for Transcription and Translation

This part showcases how to use Whisper's transcription and translation endpoints via the OpenAI API to process audio files. The steps include:

- **Transcription**: Convert audio to text in the original language.
- **Translation**: Convert non-English audio into English text.


## 6: Text-to-Speech Example

This section demonstrates how to convert transcribed text into speech using the gTTS library. The generated speech is saved as an audio file.


## 7: Any-Audio to English-Audio Translation

This section provides a comprehensive function to translate non-English audio into English speech.

- **Steps**:
  1. Translate audio into English text using Whisper.
  2. Convert the translated text into English speech using gTTS.


## 9: Capturing User Audio

### 9.1: Record Audio from the User

This section demonstrates how to record audio from the user within the Colab environment. The audio is saved as an MP3 file for further processing.


### 9.2: Convert Recorded Audio to English Speech

Once the user records audio, it can be processed to convert it into English speech using the functions defined earlier.


## 10: Audio Capture and Speaker Diarization

### 10.1: Pre-requisites for Using Speaker Diarization

Speaker diarization is performed using the Pyannote library. This section ensures that all necessary setup is in place.

### 10.2: Download a Sample Conversation Audio

The code downloads a conversation audio file with multiple speakers to demonstrate speaker diarization.



### 10.3: Transcribing the Audio Using Whisper API

The downloaded audio is transcribed using the Whisper API.



### 10.6: Run the Speaker Diarization Pipeline

The Pyannote library separates the speakers in the conversation, providing time segments for each speaker.



## 11: Gradio Interface for Transcription and Translation

A simple **Gradio** interface allows users to upload their audio files and select a language for translation. The interface outputs the transcribed text and translated speech.


## Conclusion

This project demonstrates the capabilities of OpenAI’s Whisper model for audio transcription and translation, Google Text-to-Speech for text-to-speech conversion, and Pyannote for speaker diarization. Additionally, it provides an interactive Gradio interface to simplify user interaction.
