from abc import ABC, abstractmethod
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
)


class BaseModel(ABC):

    @abstractmethod
    def load(self, model_id: str) -> None:
        self.model: AutoModelForImageTextToText | AutoModelForCausalLM
        self.processor: AutoProcessor

    @abstractmethod
    def generate(self, input_batch: tuple[str, str]) -> str:
        pass

    def unload(self) -> None:
        self.processor = None
        self.model = None
        torch.cuda.empty_cache()
