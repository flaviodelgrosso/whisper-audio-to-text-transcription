import torch
from pyannote.audio import Pipeline
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn

from .diarize import post_process_segments_and_transcripts, diarize_audio, \
    preprocess_inputs


def diarize(outputs, audio_path, device, hf_auth_token):
    diarization_pipeline = Pipeline.from_pretrained(
        checkpoint_path="pyannote/speaker-diarization-3.1",
        use_auth_token=hf_auth_token,
    )
    diarization_pipeline.to(
        torch.device(device)
    )

    with Progress(
            TextColumn("🔊 [progress.description]{task.description}"),
            BarColumn(style="yellow1", pulse_style="white"),
            TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Segmenting...", total=None)

        inputs, diarizer_inputs = preprocess_inputs(inputs=audio_path)

        segments = diarize_audio(diarizer_inputs, diarization_pipeline,
                                 None, None, None)

        return post_process_segments_and_transcripts(
            segments, outputs["chunks"], group_by_speaker=False
        )
