from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from typing import Literal
from models.base_model import BaseModel
from models.prompt import DEFAULT_AYA_SYSTEM_PROMPT
import requests
from PIL import Image
from io import BytesIO

class AyaModel(BaseModel):
    def __init__(self, lang_flag: str, system_prompt: str | None = None, size: Literal["8b", "32b"] = "8b"):
        super().__init__()
        self.model_id = f"CohereForAI/aya-vision-{size}"
        self.system_prompt = (
            system_prompt if system_prompt else DEFAULT_AYA_SYSTEM_PROMPT
        )
        language = "Czech" if lang_flag == "CS" else "English" if lang_flag == "EN" else "Ukrainian" if lang_flag == "UK" else "Unknown"
        self.system_prompt = self.system_prompt.replace("{{language}}", language)
        self.message_template = lambda image_url, question: {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": f"{self.system_prompt}\n\n{question}"},
            ],
        }

        self.load(self.model_id)

    def load(self, model_id: str) -> None:
        self.unload()
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )

    def load_image(self, source: str) -> Image.Image:
        return super().load_image(source)

    def generate(self, input_batch: tuple[str, str], input_type: Literal["url", "url_agent", "file"] = "url_agent") -> str:
        image_source, question = input_batch

        if input_type == "url":
            image = image_source  # Pass URL directly
        elif input_type == "url_agent":
            image = self.load_image(image_source)  # Use a User-Agent to load the image
        elif input_type == "file":
            image = self.load_image(image_source)  # Load local file
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")

        messages = [self.message_template(image, question)]
        inputs = self.processor.apply_chat_template(
            messages,
            padding=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_tokens = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.001,
        )

        decoded = self.processor.tokenizer.decode(
            gen_tokens[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        print('Model output:', decoded)
        return decoded

    def generate_batch(self, input_batch: list[tuple[str, str]]) -> list[str]:
        raise NotImplementedError("Batch generation is not supported for Aya model")

# Example usage:
if __name__ == "__main__":
    # Example image URL and question.
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    question = "Describe the image in detail."

    aya = AyaModel()
    answer = aya.generate((image_url, question))
    print("Generated Answer:", answer)
