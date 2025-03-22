from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from models.base_model import BaseModel
from PIL import Image
import requests

class Phi4Model(BaseModel):
    def __init__(self, system_prompt: str | None = None):
        super().__init__()
        self.model_id = "microsoft/Phi-4-multimodal-instruct"
        self.system_prompt = system_prompt if system_prompt else "You are a helpful assistant."
        self.message_template = lambda question: (
            "<|system|>"+self.system_prompt+"<|end|>"
            "<|user|><|image_1|>" + question + "<|end|><|assistant|>"
        )
        self.load(self.model_id)

    def load(self, model_id: str) -> None:
        self.unload()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
        ).eval()

    def generate(self, input_batch: tuple[str, str]) -> str:
        image_url, question = input_batch
        image = Image.open(requests.get(image_url, stream=True).raw)

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
        return decoded

# Example usage:
if __name__ == "__main__":
    # Example image URL and question.
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    question = "Describe the image in detail."

    phi4 = Phi4Model()
    answer = phi4.generate((image_url, question))
    print("Generated Answer:", answer)
