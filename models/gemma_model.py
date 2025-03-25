from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
from typing import Literal
from models.base_model import BaseModel


class GemmaModel(BaseModel):
    def __init__(
        self, system_prompt: str | None = None, size: Literal["4b", "12b", "27b"] = "4b"
    ):
        super().__init__()
        self.model_id = f"google/gemma-3-{size}-it"
        self.system_prompt = (
            system_prompt if system_prompt else "You are a helpful assistant."
        )
        self.message_template = lambda image_url, question: [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": question},
                ],
            },
        ]
        self.load(self.model_id)

    def load(self, model_id: str) -> None:
        self.unload()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval()

    def generate(self, input_batch: tuple[str, str]) -> str:
        messages = self.message_template(*input_batch)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            gen_tokens = self.model.generate(
                **inputs, max_new_tokens=100
            )

        gen_tokens = gen_tokens[0][input_len:]
        decoded = self.processor.decode(gen_tokens, skip_special_tokens=True)
        return decoded


# Example usage:
if __name__ == "__main__":
    # Assume we have an image URL and a question.
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    question = "Describe this image in detail."

    gemma = GemmaModel()
    answer = gemma.generate((image_url, question))
    print("Generated Answer:", answer)
