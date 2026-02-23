#!/usr/bin/env python3
import re
import timeit
import torch
import sys
import wave
import os
import io
import numpy as np
from pydantic import BaseModel
from datetime import datetime, timedelta
from num2t4ru import num2text
from omegaconf import OmegaConf
from fastapi.responses import StreamingResponse, FileResponse
from starlette.background import BackgroundTask

output: str = "/home/silero-user/silero/output/"
cache_dir: str = "/home/silero-user/.cache/torch/hub"

# Configurable parameters:
model_id: str = 'v4_ru'
language: str = 'ru'
put_accent: bool = True
put_yo: bool = True
speaker: str = 'xenia'
sample_rate: int = 48000  # Hz - 48000, 24000 or 8000
torch_device: str = 'auto'  # cpu, cuda or auto
torch_num_threads: int = 6
line_length_limits: dict = {
    'aidar': 870,
    'baya': 860,
    'eugene': 1000,
    'kseniya': 870,
    'xenia': 957,
    'random': 355,
}

# Global constants:
wave_channels: int = 1  # Mono
wave_sample_width: int = int(16 / 8)  # 16 bits == 2 bytes


class TTSRequest(BaseModel):
    text: list
    speaker: str
    file: str


def audio_to_wav_bytes(audio_tensor: torch.Tensor) -> bytes:
    """Конвертирует тензор аудио в WAV байты"""
    # Создаем буфер в памяти
    buffer = io.BytesIO()

    # Создаем WAV файл в памяти
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(wave_channels)
        wf.setsampwidth(wave_sample_width)
        wf.setframerate(sample_rate)

        # Конвертируем тензор в int16 и записываем
        audio_numpy = (audio_tensor * 32767).numpy().astype('int16')
        wf.writeframes(audio_numpy.tobytes())

    # Возвращаем байты
    buffer.seek(0)
    return buffer.getvalue()


def silero_tts(request: TTSRequest):
    print("sileroTTS - streaming mode")

    # Устанавливаем параметры
    global speaker
    speaker = request.speaker
    origin_lines = [request.text]
    line_length_limit: int = line_length_limits[speaker]

    # Препроцессинг текста
    preprocessed_lines, preprocessed_text_len = preprocess_text(origin_lines, line_length_limit)

    print(f'Available speakers: {tts_model.speakers}')
    print(f"Generating audio for {len(preprocessed_lines)} lines...")

    # Генерируем аудио для всех линий и объединяем
    all_audio = []
    for line in preprocessed_lines:
        if line.strip():
            audio = tts_model.apply_tts(
                text=line.strip(),
                speaker=speaker,
                sample_rate=sample_rate,
                put_accent=put_accent,
                put_yo=put_yo
            )
            all_audio.append(audio)

    # Объединяем все аудио в одно
    if len(all_audio) > 1:
        combined_audio = torch.cat(all_audio, dim=0)
    else:
        combined_audio = all_audio[0]

    # Конвертируем в WAV байты
    wav_bytes = audio_to_wav_bytes(combined_audio)

    print(f"Generated {len(wav_bytes)} bytes of audio data")

    # Возвращаем аудио как streaming response
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename={speaker}_audio.wav",
            "Content-Length": str(len(wav_bytes))
        }
    )


def preprocess_text(lines: list, length_limit: int) -> (list, int):
    print(f"Preprocessing text with line length limit={length_limit}")

    if length_limit > 3:
        length_limit = length_limit - 2
    else:
        print(F"ERROR: line length limit must be >= 3, got {length_limit}")
        return [], 0

    preprocessed_text_len: int = 0
    preprocessed_lines: list = []
    for line in lines:
        line = line.strip()
        if line == '\n' or line == '':
            continue

        # Replace chars not supported by model
        line = line.replace("…", "...")
        line = line.replace("*", " звёздочка ")
        line = re.sub(r'(\d+)[\.|,](\d+)', r'\1 и \2', line)
        line = line.replace("%", " процентов ")
        line = line.replace(" г.", " году")
        line = line.replace(" гг.", " годах")
        line = re.sub("д.\s*н.\s*э.", " до нашей эры", line)
        line = re.sub("н.\s*э.", " нашей эры", line)
        line = spell_digits(line)

        while len(line) > 0:
            if len(line) < length_limit:
                line = line + "\n"
                preprocessed_lines.append(line)
                preprocessed_text_len += len(line)
                break

            split_position: int = 0
            split_position = find_split_position(line, split_position, ".", length_limit)
            split_position = find_split_position(line, split_position, "!", length_limit)
            split_position = find_split_position(line, split_position, "?", length_limit)

            if split_position == 0:
                split_position = find_split_position(line, split_position, " ", length_limit)

            if split_position == 0:
                split_position = length_limit

            part: str = line[0:split_position + 1] + "\n"
            preprocessed_lines.append(part)
            preprocessed_text_len += len(part)
            line = line[split_position + 1:]

    return preprocessed_lines, preprocessed_text_len


def spell_digits(line) -> str:
    digits: list = re.findall(r'\d+', line)
    digits = sorted(digits, key=len, reverse=True)
    for digit in digits:
        line = line.replace(digit, num2text(int(digit[:12])))
    return line


def find_char_positions(string: str, char: str) -> list:
    pos: list = []
    for n in range(len(string)):
        if string[n] == char:
            pos.append(n)
    return pos


def find_max_char_position(positions: list, limit: int) -> int:
    max_position: int = 0
    for pos in positions:
        if pos < limit:
            max_position = pos
        else:
            break
    return max_position


def find_split_position(line: str, old_position: int, char: str, limit: int) -> int:
    positions: list = find_char_positions(line, char)
    new_position: int = find_max_char_position(positions, limit)
    position: int = max(new_position, old_position)
    return position


def download_models_config():
    print("Downloading models config")
    config_path = os.path.join(cache_dir, 'silero_models.yml')

    if os.path.exists(config_path):
        print(f"Config already exists at {config_path}")
        return config_path

    torch.hub.download_url_to_file(
        'https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
        config_path,
        progress=True
    )
    return config_path


def print_models_information():
    config_path = os.path.join(cache_dir, 'silero_models.yml')
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
    else:
        config_path = download_models_config()
        config = OmegaConf.load(config_path)

    available_languages = list(config.tts_models.keys())
    print(f'Available languages {available_languages}')
    for lang in available_languages:
        models: list = list(config.tts_models.get(lang).keys())
        print(f'Available models for {lang}: {models}')


def init_model(device: str, threads_count: int) -> torch.nn.Module:
    print("Initialising model")
    t0 = timeit.default_timer()

    torch._C._jit_set_profiling_mode(False)

    if not torch.cuda.is_available() and device == "auto":
        device = 'cpu'
    if torch.cuda.is_available() and device == "auto" or device == "cuda":
        torch_dev: torch.device = torch.device("cuda", 0)
        gpus_count = torch.cuda.device_count()
        print("Using {} GPU(s)...".format(gpus_count))
    else:
        torch_dev: torch.device = torch.device(device)

    torch.set_num_threads(threads_count)

    torch.hub.set_dir(cache_dir)

    print(f"Loading model {language}/{model_id} from cache: {cache_dir}")
    tts_model, tts_sample_text = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language=language,
        speaker=model_id,
        force_reload=False
    )

    print("Setup takes {:.2f}".format(timeit.default_timer() - t0))

    print("Loading model to device")
    t1 = timeit.default_timer()
    tts_model.to(torch_dev)
    print("Model to device takes {:.2f}".format(timeit.default_timer() - t1))

    if torch.cuda.is_available() and device == "auto" or device == "cuda":
        print("Synchronizing CUDA")
        t2 = timeit.default_timer()
        torch.cuda.synchronize()
        print("Cuda Synch takes {:.2f}".format(timeit.default_timer() - t2))

    print("Model is loaded")
    return tts_model


# Инициализация модели при старте
print("Initializing Silero TTS...")
config_path = download_models_config()
print_models_information()
tts_model: torch.nn.Module = init_model(torch_device, torch_num_threads)
print("✓ Silero TTS initialized successfully!")