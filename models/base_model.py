from abc import ABC, abstractmethod
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

class BaseModel(ABC):
    @abstractmethod
    def load(self, model_id: str) -> tuple[AutoProcessor, AutoModelForImageTextToText]:
        pass

    @abstractmethod
    def generate(self, input_batch: list[tuple[str, str]]) -> list[str]:
        pass

    def unload(self) -> None:
        self.processor = None
        self.model = None
        torch.cuda.empty_cache()