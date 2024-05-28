import torch
from transformers import pipeline
import librosa
import io

def convert_audio(audio):
    audio = io.BytesIO(audio)
    audio,sample_rate = librosa.load(audio)
    print(sample_rate)
    return audio

def decode_audio(audio):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task ="automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=device,
        )

    audio = convert_audio(audio)


    prediction = pipe(audio, batch_size=1)["text"]
    return prediction