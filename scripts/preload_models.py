#!/usr/bin/env python3
import torch
import os
import time


def preload_silero_models():
    print("Preloading Silero models...")

    # Настройки для предзагрузки
    languages = ['ru', 'en']
    models = ['v4_ru', 'v3_en']

    # Создаем директорию для кэша если её нет
    cache_dir = os.path.expanduser('~/.cache/torch/hub')
    os.makedirs(cache_dir, exist_ok=True)

    # Отключаем profiling mode
    torch._C._jit_set_profiling_mode(False)

    # Предзагружаем модели для разных языков
    for language in languages:
        for model_id in models:
            try:
                print(f"Loading model: {language}/{model_id}")
                start_time = time.time()

                model, example_text = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language=language,
                    speaker=model_id
                )

                # Загружаем на CPU для кэширования
                model.to('cpu')

                # Делаем тестовый прогон для загрузки всех весов
                test_text = "Тестовое сообщение для предварительной загрузки модели."
                _ = model.apply_tts(
                    text=test_text,
                    speaker='xenia' if language == 'ru' else 'en_0',
                    sample_rate=48000
                )

                elapsed_time = time.time() - start_time
                print(f"✓ Model {language}/{model_id} loaded in {elapsed_time:.2f}s")

            except Exception as e:
                print(f"✗ Failed to load {language}/{model_id}: {e}")

    # Также предзагружаем конфиг
    print("\nDownloading models config...")
    config_path = torch.hub.download_url_to_file(
        'https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
        os.path.join(cache_dir, 'silero_models.yml'),
        progress=True
    )

    print("\n✓ All models preloaded successfully!")


if __name__ == "__main__":
    preload_silero_models()