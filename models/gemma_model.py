from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
from typing import Literal
from models.base_model import BaseModel
from models.prompt import DEFAULT_GEMMA_SYSTEM_PROMPT
import requests
from io import BytesIO
from typing import Literal, Union
from PIL import Image

class GemmaModel(BaseModel):
    def __init__(
        self, lang_flag: str = "EN", system_prompt: str | None = None, size: Literal["4b", "12b", "27b"] = "4b"
    ):
        super().__init__()
        self.model_id = f"google/gemma-3-{size}-it"
        self.system_prompt = system_prompt if system_prompt else DEFAULT_GEMMA_SYSTEM_PROMPT
        language = "Czech" if lang_flag == "CS" else "English" if lang_flag == "EN" else "Ukrainian" if lang_flag == "UK" else "Unknown"
        self.system_prompt = self.system_prompt.replace("{{language}}", language)
        self.load(self.model_id)

    def load(self, model_id: str) -> None:
        self.unload()
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()

    def load_image(self, source: str) -> Image.Image:
        return super().load_image(source)

    def prepare_messages(self, image: Union[str, Image.Image], question: str) -> list[dict]:
        system_message = {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
        user_content = []
        if isinstance(image, Image.Image):
            user_content.append({"type": "image", "image": image})
        else:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": question})
        return [system_message, {"role": "user", "content": user_content}]

    def generate(self, input_tuple: tuple[str, str], input_type: Literal["url", "url_agent", "file"] = "url_agent") -> str:
        image_source, question = input_tuple

        if input_type == "url":
            image = image_source  # Pass URL directly
        elif input_type == "url_agent":
            image = self.load_image(image_source)  # Use a User-Agent to load the image
        elif input_type == "file":
            image = self.load_image(image_source)  # Load local file
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")

        messages = self.prepare_messages(image, question)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            max_length=4096,
            truncation=True
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            gen_tokens = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.001,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        gen_tokens = gen_tokens[0][input_len:]
        decoded = self.processor.decode(gen_tokens, skip_special_tokens=True)
        print('decoded:', decoded)
        return decoded
    
    def generate_batch(self, input_batch: list[tuple[str, str]]) -> list[str]:
        pass


# Example usage:
if __name__ == "__main__":
    # Assume we have an image URL and a question.
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    question = "Describe this image in detail."

    gemma = GemmaModel()
    answer = gemma.generate((image_url, question))
    print("Generated Answer:", answer)
