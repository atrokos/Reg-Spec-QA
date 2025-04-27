from abc import ABC, abstractmethod
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
)
import requests
from PIL import Image
from io import BytesIO
from typing import Literal, Union


class BaseModel(ABC):

    @abstractmethod
    def load(self, model_id: str) -> None:
        self.model: AutoModelForImageTextToText | AutoModelForCausalLM
        self.processor: AutoProcessor

    @abstractmethod
    def generate(self, input_tuple: tuple[str, str], input_type: Literal["url", "url_agent", "file"] = "url") -> str:
        pass

    @abstractmethod
    def generate_batch(self, input_batch: list[tuple[str, str]]) -> list[str]:
        pass

    def unload(self) -> None:
        self.processor = None
        self.model = None
        torch.cuda.empty_cache()

    def load_image(self, source: str) -> Image.Image:
        if source.startswith("http://") or source.startswith("https://"):
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/117.0.0.0 Safari/537.36"
            }
            response = requests.get(source, headers=headers)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(source)

        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
