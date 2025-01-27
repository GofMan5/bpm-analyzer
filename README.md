# 🎵 BPM Analyzer

Инструмент для автоматического определения темпа (BPM) в аудиофайлах с удобным консольным интерфейсом.

## 📋 Описание

BPM Analyzer - это Python-приложение, которое анализирует аудиофайлы и определяет их темп (BPM - beats per minute). Программа поддерживает работу с различными форматами аудио и предоставляет детальную статистику обработки файлов.

### ✨ Основные возможности

- Определение BPM в аудиофайлах
- Поддержка форматов: MP3, WAV, OGG, M4A
- Многопоточная обработка файлов
- Красивый консольный интерфейс с прогресс-баром
- Сохранение результатов анализа в JSON
- Подробная статистика обработки

## 🚀 Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/GofMan5/bpm-analyzer.git
cd bpm-analyzer
```

2. Установите зависимости:

```bash
pip install -r requirements.txt
```


## 📦 Зависимости

- librosa >= 0.10.0
- numpy >= 1.20.0
- tqdm >= 4.65.0
- rich >= 13.0.0
- colorama >= 0.4.6

## 🎮 Использование

1. Поместите аудиофайлы в папку `audios/`

2. Запустите анализатор:

```bash
python bpm_detector.py
```


3. Дождитесь завершения анализа. Результаты будут:
   - Отображены в консоли
   - Сохранены в файл `results.json`

## 📊 Вывод результатов

Программа предоставляет подробную информацию о каждом обработанном файле:
- Статус обработки (успешно/ошибка)
- Имя файла
- Определенный BPM
- Длительность файла
- Время обработки

## 🛠️ Технические детали

- Многопоточная обработка для ускорения анализа
- Использование библиотеки librosa для анализа аудио
- Красивый вывод с помощью библиотеки rich
- Сохранение результатов в JSON для дальнейшего использования

## 📝 Формат результатов

Результаты сохраняются в JSON-файл со следующей структурой:

```json
{
  "results": [
    {
      "filename": "example.mp3",
      "bpm": 120,
      "duration": 180,
      "processing_time": 2.5
    }
  ]
}
```


## 🤝 Вклад в проект

Если вы хотите внести свой вклад в проект:
1. Создайте форк репозитория
2. Создайте ветку для новой функциональности
3. Отправьте пулл-реквест

## 📄 Лицензия

[MIT License](LICENSE)