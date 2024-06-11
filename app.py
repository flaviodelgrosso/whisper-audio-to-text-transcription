import torch
import os
import platform

from dotenv import load_dotenv
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from transformers import pipeline

from utils.result import build_result
from utils.diarization_pipeline import diarize
from utils.convert_output import convert


load_dotenv()

is_cuda_available = torch.cuda.is_available()

# Check if the system is macOS (Darwin)
if platform.system() == "Darwin":
    device = "mps"  # Use Metal Performance Shaders on macOS
else:
    # Check if CUDA is available, otherwise use CPU
    device = "cuda:0" if is_cuda_available else "cpu"

torch_dtype = torch.float16 if is_cuda_available else torch.float32
model = "openai/whisper-large-v3"

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    torch_dtype=torch.float16,
    device=device,
    model_kwargs={"attn_implementation": "sdpa"}
)

if device == "mps":
    torch.mps.empty_cache()


audio_dir = "audio"
hf_auth_token = os.getenv("HF_AUTH_TOKEN")

for file_name in os.listdir(audio_dir):
    if (file_name.startswith(".")):
        continue

    audio_path = os.path.join(audio_dir, file_name)

    with Progress(
        TextColumn("🖋️ [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task(
            f"[yellow]Transcribing {file_name} [Device: {device}]...", total=None)

        outputs = pipe(
            audio_path,
            chunk_length_s=30,
            batch_size=24,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=True,
        )

    # Determine the speakers_transcript based on the hf_auth_token
    if hf_auth_token is None:
        speakers_transcript = []
    else:
        speakers_transcript = diarize(
            outputs, audio_path, device, hf_auth_token)

    # Build the result and format
    result = build_result(speakers_transcript, outputs)
    output_path = f"transcriptions/{os.path.splitext(file_name)[0]}.txt"
    convert(result, output_path)

print(
    f"✨ Your file has been successfully transcribed!"
)
