import re
from pathlib import Path
import ollama
from typing import List, Optional


def read_from_file(path: Path) -> str:
    with open(path, "r") as file:
        return file.read()


def generate_text(model: str, options: ollama.Options, prompt: str, k: int) -> List[ollama.ChatResponse]:
    responses = []
    for i in range(k):
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            options=options
        )
        responses.append(response)

    return responses


def get_code_from_response(response: ollama.ChatResponse, regex: List[str]) -> str:
    for reg in regex:
        code = re.search(reg, response['message']['content'], re.DOTALL)
        if code:
            return code.group(1).strip()

    return ''


def append_and_save_code(path: Path, txt: str):
    with open(path, 'a') as file:
        file.write(txt)


def indent_code(code: str, signature: Optional[str] = None) -> str:
    indented_code = ''
    if signature:
        indented_code += f'    {signature}\n'
    indented_code += '\n'.join(['    ' + line for line in code.split('\n')])
    return indented_code
