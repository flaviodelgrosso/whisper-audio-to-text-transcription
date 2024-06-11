# :ear: Whisper Transcriber

This is a Python project that uses the OpenAI open source Whisper [library](https://github.com/openai/whisper.git) to transcribe audio files and detect the language of the transcribed text.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Run in Google Colab

<a href="https://colab.research.google.com/github/flaviodelgrosso/whisper-audio-to-text-transcription/blob/master/notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Running with DevContainers in Visual Studio Code

If you have the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension installed in Visual Studio Code, you can open this project in a development container. This will automatically install all the necessary dependencies and set up the environment for you in an isolated Docker container.

Follow these steps:

1. Clone the repository.
2. Open the project in Visual Studio Code.
3. When prompted to "Reopen in Container", select "Reopen in Container". If you're not prompted, you can press `F1` to open the command palette, then select "Remote-Containers: Reopen Folder in Container".

The first time you open the container, it may take a few minutes to build. Once the container is built, the terminal will connect to the running container.

To run the script, use the terminal in Visual Studio Code:

```sh
python app.py
```

### Usage

Run the app.py script to transcribe the audio files in the audio directory:

```sh
python app.py
```

The transcribed text will be saved in the transcriptions directory.

#### Speakers Diarization

If you have an audio file with multiple speakers, you can perform speaker diarization by adding your Hugging Face authentication token in your `.env` file

```sh
HF_AUTH_TOKEN=
```
