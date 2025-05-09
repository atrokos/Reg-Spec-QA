from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from models.base_model import BaseModel
from PIL import Image
import requests
from functools import lru_cache
from typing import Literal

from models.prompt import DEFAULT_PHI4_SYSTEM_PROMPT

class Phi4Model(BaseModel):
    def __init__(self, lang_flag: str, system_prompt: str | None = None):
        super().__init__()
        self.model_id = "microsoft/Phi-4-multimodal-instruct"
        self.system_prompt = system_prompt if system_prompt else DEFAULT_PHI4_SYSTEM_PROMPT
        language = "Czech" if lang_flag == "CS" else "English" if lang_flag == "EN" else "Ukrainian" if lang_flag == "UK" else "Unknown"
        self.system_prompt = self.system_prompt.replace("{{language}}", language)
        self.message_template = lambda question: (
            "<|system|>"+self.system_prompt+"<|end|>"
            "<|user|><|image_1|>" + question + "<|end|><|assistant|>"
        )
        self.load(self.model_id)

    def load_image(self, source: str) -> Image.Image:
        return super().load_image(source)

    def load(self, model_id: str) -> None:
        self.unload()
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_flash_attention_2=True  # Enable Flash Attention 2 for faster inference
        ).eval()

    def generate(self, input_tuple: tuple[str, str], input_type: Literal["url", "url_agent", "file"] = "url_agent") -> str:
        image_source, question = input_tuple

        if input_type == "url":
            image = self.load_image(image_source)  # Pass URL directly
        elif input_type == "url_agent":
            image = self.load_image(image_source)  # Use a User-Agent to load the image
        elif input_type == "file":
            image = self.load_image(image_source)  # Load local file
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")

        prompt = self.message_template(question)

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            gen_tokens = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                num_return_sequences=1,
                num_logits_to_keep=1
            )

        gen_tokens = gen_tokens[:, input_len:]
        decoded = self.processor.batch_decode(
            gen_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        print('Model output:', decoded)
        return decoded

    def generate_batch(self, input_batch: list[tuple[str, str]]) -> list[str]:
        try:
            # Load all images first
            images = [self.load_image(image_url) for image_url, _ in input_batch]
            prompts = [self.message_template(question) for _, question in input_batch]

            # Process all inputs at once
            inputs = self.processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(self.model.device)
            input_lens = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                gen_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    num_return_sequences=1,
                    num_logits_to_keep=1
                )

            # Process all responses
            responses = []
            for i in len(input_batch):
                gen_tokens_i = gen_tokens[i][input_lens:]
                decoded = self.processor.batch_decode(
                    gen_tokens_i.unsqueeze(0),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                responses.append(decoded)

            return responses
        except Exception as e:
            # Fallback to sequential processing if batch processing fails
            responses = []
            for input_tuple in input_batch:
                response = self.generate(input_tuple)
                responses.append(response)
            return responses

# Example usage:
if __name__ == "__main__":
    # Example image URL and question.
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    question = "Describe the image in detail."

    phi4 = Phi4Model()
    answer = phi4.generate((image_url, question))
    print("Generated Answer:", answer)
