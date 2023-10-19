# tts.py

import io
import torch
import os
from torchaudio.transforms import Resample

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = os.path.join(os.getcwd(), '/home/silero-user/silero/model.pt')
sample_rate = 16000

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt', local_file)

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

resampler = Resample(orig_freq=sample_rate, new_freq=sample_rate)


def speak(text: str, speaker: str):
    wav = model(text=text, speaker=speaker, sample_rate=sample_rate)
    wav = resampler(wav)
    audio_buffer = io.BytesIO()
    torchaudio.save(audio_buffer, src=wav, sample_rate=sample_rate, format='wav')
    audio = audio_buffer.getvalue()
    return audio
