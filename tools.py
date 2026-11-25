import io
import time
import tempfile
import PyPDF2
import os
from typing import List, Union, Dict, Any


def get_time_counter():
    """Retorna o contador de performance atual para calcular o tempo de execução."""
    return time.perf_counter()


def calc_time_counter(start_time: float):
    """Calcula e formata o tempo decorrido desde o início da execução."""
    processing_time = {}
    processing_time["start_time"] = start_time
    processing_time["end_time"] = get_time_counter()
    total_time = processing_time["end_time"] - processing_time["start_time"]
    processing_time["elapsed_time"] = f"{total_time:.2f} segundos"
    return processing_time


def create_temp_files(files: Union[List, object]):
    """Cria arquivos temporários no sistema para processamento."""
    if not isinstance(files, list):
        files = [files]

    temp_files = []
    for file in files:
        file.seek(0)
        file_content = file.read()
        file_name = getattr(file, "name", f"temp_file_{os.getpid()}.pdf")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_files.append(
                {
                    "file": temp_file,
                    "metadata": {
                        "original_name": file_name,
                        "type": "pdf",
                        "temp_name": temp_file.name,
                    },
                }
            )
    return temp_files
