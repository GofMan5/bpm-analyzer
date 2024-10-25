from pathlib import Path
from typing import Dict, Optional, Set, Union, List, Tuple
import logging
from dataclasses import dataclass
import json
from datetime import datetime
import librosa
import numpy as np
from abc import ABC, abstractmethod
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import colorama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.logging import RichHandler
import warnings
import os
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.style import Style
from rich.text import Text
from rich import box
from rich.live import Live
from datetime import timedelta

# Инициализация rich консоли
console = Console()
colorama.init()

# Игнорируем предупреждения librosa
warnings.filterwarnings("ignore")

# Настройка красивого логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True, show_time=False),
        logging.FileHandler('bpm_detector.log')
    ]
)
logger = logging.getLogger("bpm_detector")

@dataclass
class AudioConfig:
    """Конфигурация для обработки аудио"""
    supported_extensions: Set[str]
    input_folder: Path
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'AudioConfig':
        """Загрузка конфигурации из JSON файла"""
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(
            supported_extensions=set(config['supported_extensions']),
            input_folder=Path(config['input_folder'])
        )

@dataclass
class ProcessingStatus:
    """Статус обработки файла"""
    total_duration: float
    processing_time: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class AnalysisResult:
    """Результат анализа аудиофайла"""
    filename: str
    bpm: float
    timestamp: datetime
    status: ProcessingStatus

class AudioAnalyzer(ABC):
    """Абстрактный класс для анализа аудио"""
    
    @abstractmethod
    def analyze(self, audio_path: Path) -> tuple[Optional[float], ProcessingStatus]:
        """Анализирует аудиофайл"""
        pass

class BPMAnalyzer(AudioAnalyzer):
    """Оптимизированный анализатор BPM"""
    
    def __init__(self):
        self.sr = 22050  # Оптимальная частота дискретизации для BPM
        self.hop_length = 512  # Оптимальный размер хопа
    
    def analyze(self, audio_path: Path) -> Tuple[Optional[float], ProcessingStatus]:
        start_time = time.time()
        try:
            # Оптимизированная загрузка аудио
            y, sr = librosa.load(
                str(audio_path),
                sr=self.sr,
                mono=True,
                duration=120  # Ограничиваем длину до 2 минут для ускорения
            )
            
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Оптимизированное определение темпа
            onset_env = librosa.onset.onset_strength(
                y=y, 
                sr=sr,
                hop_length=self.hop_length
            )
            tempo = librosa.beat.tempo(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=self.hop_length
            )
            
            if hasattr(tempo, '__len__'):
                tempo = tempo[0]
            
            processing_time = time.time() - start_time
            status = ProcessingStatus(
                total_duration=duration,
                processing_time=processing_time,
                success=True
            )
            
            return round(float(tempo), 2), status
            
        except Exception as e:
            processing_time = time.time() - start_time
            status = ProcessingStatus(
                total_duration=0,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
            return None, status

class AudioProcessor:
    """Улучшенный процессор для обработки аудиофайлов"""
    
    def __init__(self, config: AudioConfig, analyzer: AudioAnalyzer):
        self.config = config
        self.analyzer = analyzer
        self.results: Dict[str, AnalysisResult] = {}
        self.max_workers = min(32, (os.cpu_count() or 1) + 4)
        self.results_file = Path("results.json")
        # Загружаем предыдущие результаты при инициализации
        self.load_results()
    
    def is_supported_file(self, file_path: Path) -> bool:
        """Проверяет, поддерживается ли формат файла"""
        return file_path.suffix.lower() in self.config.supported_extensions
    
    def get_audio_files(self) -> List[Path]:
        """Получает список поддерживаемых аудиофайлов"""
        return [
            f for f in self.config.input_folder.iterdir()
            if self.is_supported_file(f)
        ]
    
    def process_file(self, file_path: Path) -> Tuple[Path, Optional[float], ProcessingStatus]:
        """Обработка одного файла"""
        bpm, status = self.analyzer.analyze(file_path)
        return file_path, bpm, status
    
    def save_results(self) -> None:
        """Сохраняет результаты в JSON файл"""
        results_dict = {
            filename: {
                "filename": result.filename,
                "bpm": result.bpm,
                "timestamp": result.timestamp.isoformat(),
                "status": {
                    "total_duration": result.status.total_duration,
                    "processing_time": result.status.processing_time,
                    "success": result.status.success,
                    "error_message": result.status.error_message
                }
            }
            for filename, result in self.results.items()
        }
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Результаты сохранены в {self.results_file}")
    
    def load_results(self) -> None:
        """Загружает результаты из JSON файла"""
        if not self.results_file.exists():
            return
            
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                results_dict = json.load(f)
            
            self.results = {
                filename: AnalysisResult(
                    filename=data["filename"],
                    bpm=data["bpm"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    status=ProcessingStatus(
                        total_duration=data["status"]["total_duration"],
                        processing_time=data["status"]["processing_time"],
                        success=data["status"]["success"],
                        error_message=data["status"]["error_message"]
                    )
                )
                for filename, data in results_dict.items()
            }
            
            logger.info(f"Загружены предыдущие результаты из {self.results_file}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке результатов: {e}")
    
    def process_folder(self) -> None:
        """Параллельная обработка файлов"""
        audio_files = self.get_audio_files()
        # Фильтруем только новые файлы
        new_files = [f for f in audio_files if f.name not in self.results]
        total_files = len(new_files)
        
        if total_files == 0:
            console.print(Panel(
                "[yellow]Новых аудиофайлов не найдено![/yellow]",
                border_style="yellow",
                box=box.ROUNDED
            ))
            return
        
        console.print(Panel(
            f"[blue]📁 Найдено новых файлов: {total_files}[/blue]",
            border_style="blue",
            box=box.ROUNDED
        ))
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Обработка файлов...", total=total_files)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_file, file_path): file_path 
                    for file_path in new_files
                }
                
                for future in as_completed(future_to_file):
                    file_path, bpm, status = future.result()
                    if status.success:
                        self.results[file_path.name] = AnalysisResult(
                            filename=file_path.name,
                            bpm=bpm,
                            timestamp=datetime.now(),
                            status=status
                        )
                        # Сохраняем результаты после каждого успешного анализа
                        self.save_results()
                    progress.advance(task)
    
    def create_summary_panel(self, successful: int, total: int) -> Panel:
        """Создает панель с общей статистикой"""
        grid = Table.grid(expand=True)
        grid.add_column(justify="center")
        
        # Заголовок
        grid.add_row(Text("📊 Статистика обработки", style="bold cyan"))
        grid.add_row("")
        
        # Статистика
        stats = Table.grid(expand=True)
        stats.add_column(style="bright_white", justify="right")
        stats.add_column(style="cyan", justify="left")
        
        stats.add_row("Всего файлов:", f" {total}")
        stats.add_row(
            "Успешно обработано:", 
            f" [green]{successful}[/green]"
        )
        stats.add_row(
            "Ошибок обработки:", 
            f" [red]{total - successful}[/red]"
        )
        
        grid.add_row(stats)
        
        return Panel(
            grid,
            border_style="cyan",
            box=box.ROUNDED,
            title="[bold cyan]🎵 BPM Analyzer[/bold cyan]",
            subtitle="[dim]Результаты анализа[/dim]"
        )

    def create_results_table(self) -> Table:
        """Создает таблицу с результатами анализа"""
        table = Table(
            box=box.ROUNDED,
            expand=True,
            show_header=True,
            header_style="bold cyan",
            border_style="bright_black"
        )
        
        table.add_column("Статус", justify="center", width=4)
        table.add_column("Файл", style="bright_white")
        table.add_column("BPM", justify="right", width=8)
        table.add_column("Длительность", justify="right", width=12)
        table.add_column("Время обработки", justify="right", width=12)
        
        for result in self.results.values():
            status_symbol = "✓" if result.status.success else "✗"
            status_style = "green" if result.status.success else "red"
            
            duration = str(timedelta(seconds=int(result.status.total_duration)))
            proc_time = f"{result.status.processing_time:.1f}s"
            
            table.add_row(
                f"[{status_style}]{status_symbol}[/{status_style}]",
                result.filename,
                f"[cyan]{result.bpm}[/cyan]" if result.status.success else "[red]ERROR[/red]",
                duration,
                proc_time
            )
        
        return table

    def print_results(self) -> None:
        """Красивый вывод результатов"""
        successful = len([r for r in self.results.values() if r.status.success])
        total = len(self.results)
        
        # Создаем layout
        layout = Layout()
        layout.split_column(
            Layout(name="summary"),
            Layout(name="results", ratio=3)
        )
        
        # Добавляем панели
        layout["summary"].update(self.create_summary_panel(successful, total))
        layout["results"].update(self.create_results_table())
        
        # Выводим результаты
        console.print("\n")
        console.print(layout)
        console.print("\n")

def show_welcome_message():
    """Показывает приветственное сообщение"""
    welcome = Table.grid(expand=True)
    welcome.add_column(justify="center")
    
    welcome.add_row(Text("🎵 BPM ANALYZER 🎵", style="bold cyan"))
    welcome.add_row(Text("Анализ темпа аудиофайлов", style="dim"))
    welcome.add_row("")
    
    console.print(Panel(
        welcome,
        border_style="cyan",
        box=box.ROUNDED
    ))

def main():
    try:
        show_welcome_message()
        
        config = AudioConfig(
            supported_extensions={'.mp3', '.wav', '.ogg', '.m4a'},
            input_folder=Path('audios')
        )
        
        processor = AudioProcessor(config=config, analyzer=BPMAnalyzer())
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
            expand=True
        ) as progress:
            # Инициализация
            init_task = progress.add_task(
                "⚡ [bold green]Инициализация...", 
                total=1
            )
            progress.advance(init_task)
            
            # Получаем список файлов
            audio_files = processor.get_audio_files()
            total_files = len(audio_files)
            
            if total_files == 0:
                console.print(Panel(
                    "[yellow]Аудиофайлы не найдены![/yellow]",
                    border_style="yellow",
                    box=box.ROUNDED
                ))
                return
            
            progress.console.print(Panel(
                f"[blue]📁 Найдено файлов: {total_files}[/blue]",
                border_style="blue",
                box=box.ROUNDED
            ))
            
            # Обработка файлов
            process_task = progress.add_task(
                "🎵 [cyan]Анализ аудиофайлов...", 
                total=total_files
            )
            
            with ThreadPoolExecutor(max_workers=processor.max_workers) as executor:
                future_to_file = {
                    executor.submit(processor.process_file, file_path): file_path 
                    for file_path in audio_files
                }
                
                for future in as_completed(future_to_file):
                    file_path, bpm, status = future.result()
                    if status.success:
                        processor.results[file_path.name] = AnalysisResult(
                            filename=file_path.name,
                            bpm=bpm,
                            timestamp=datetime.now(),
                            status=status
                        )
                    progress.advance(process_task)
            
            # Вывод результатов
            report_task = progress.add_task(
                "📊 [bold green]Формирование отчета...", 
                total=1
            )
            processor.print_results()
            progress.advance(report_task)
        
        # Сохраняем результаты перед выходом
        processor.save_results()
        
    except Exception as e:
        console.print(Panel(
            f"[bold red]❌ Критическая ошибка:[/bold red] {str(e)}",
            border_style="red",
            box=box.ROUNDED
        ))
        logger.exception("Критическая ошибка")
    finally:
        console.input("\n[dim]Нажмите Enter для выхода...[/dim]")

if __name__ == "__main__":
    main()
