from pydantic import BaseModel, Field
from uuid import uuid4
from enum import StrEnum
from abc import ABC, abstractmethod
from typing import Optional, Any, Callable
import requests

class Topic(BaseModel):
    topic:str

class BaseLLM(ABC):
    def __init__(self, model_id,):
        self.model_id = model_id
        self.endpoint_url: str = "http://localhost:11434/api/chat"

    @abstractmethod
    def run(self, system_prompt:str, content:str)->Any:
        """Abstract method to be implemented by child classes"""
        pass

class Ollama(BaseLLM):
    def __init__(self, model_id='llama3.2:1b', Output:Optional[Callable]=None):
        super().__init__(model_id)
        self.Output = Output if Output else lambda x: x

    def run(self, system_prompt:str, content:str):
        response = requests.post(
            self.endpoint_url,
            json={
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                "stream": False
            },
        )
        return self.Output(response.json())