from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
from typing import Literal
from models.base_model import BaseModel
from models.prompt import DEFAULT_GEMMA_SYSTEM_PROMPT

class GemmaModel(BaseModel):
    def __init__(
        self, lang_flag: str, system_prompt: str | None = None, size: Literal["4b", "12b", "27b"] = "4b"
    ):
        super().__init__()
        self.model_id = f"google/gemma-3-{size}-it"
        self.system_prompt = (
            system_prompt if system_prompt else DEFAULT_GEMMA_SYSTEM_PROMPT
        )
        language = "Czech" if lang_flag == "CS" else "English" if lang_flag == "EN" else "Ukrainian" if lang_flag == "UK" else "Unknown"
        self.system_prompt = self.system_prompt.replace("{{language}}", language)
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
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()

    def generate(self, input_tuple: tuple[str, str]) -> str:
        messages = self.message_template(*input_tuple)
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
        # Prepare messages for all inputs
        messages = [self.message_template(*input_tuple) for input_tuple in input_batch]

        # Process all inputs at once
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device, dtype=torch.bfloat16)

        # Get input lengths for each sequence
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()

        # Generate responses for all inputs
        with torch.inference_mode():
            gen_tokens = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        # Decode all generated responses
        responses = []
        for i in range(len(input_batch)):
            # Slice the generated tokens to get only the new tokens
            gen_tokens_i = gen_tokens[i][input_lens[i]:]
            response = self.processor.decode(gen_tokens_i, skip_special_tokens=True)
            # Clean up the response
            response = response.strip()  # Remove leading/trailing whitespace
            response = ' '.join(response.split())  # Normalize whitespace
            responses.append(response)

        return responses


# Example usage:
if __name__ == "__main__":
    # Assume we have an image URL and a question.
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    question = "Describe this image in detail."

    gemma = GemmaModel()
    answer = gemma.generate((image_url, question))
    print("Generated Answer:", answer)
