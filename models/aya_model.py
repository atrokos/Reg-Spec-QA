from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from typing import Literal
from models.base_model import BaseModel


class AyaModel(BaseModel):
    def __init__(self, size: Literal["8b", "32b"] = "8b"):
        super().__init__()
        self.model_id = f"CohereForAI/aya-vision-{size}"
        self.message_template = lambda image_url, question: {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": question},
            ],
        }

        self.load(self.model_id)

    def load(self, model_id: str) -> None:
        self.unload()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )

    def generate(self, input_batch: tuple[str, str]) -> str:
        messages = [self.message_template(*input_batch)]
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
            max_new_tokens=300,
            do_sample=True,
            temperature=0.3,
        )

        return self.processor.tokenizer.decode(
            gen_tokens[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )


# Example usage:
if __name__ == "__main__":
    # Example image URL and question.
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    question = "Describe the image in detail."

    aya = AyaModel()
    answer = aya.generate((image_url, question))
    print("Generated Answer:", answer)
