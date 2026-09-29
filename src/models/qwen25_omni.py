import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from src.models.base import BaseAudioModel

class Qwen25OmniModel(BaseAudioModel):
    """
    Implementation specifically for Qwen2.5-Omni models.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-Omni-7B", device="cuda"):
        super().__init__(model_name, device)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing: {model_name}")
        
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2"
        ).to(self.device)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    def query(self, audio_path, text_prompt):
        """
        Prepare inputs and run inference for Qwen2.5-Omni.
        """
        if audio_path is not None:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": text_prompt},
                    ],
                },
            ]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                    ],
                },
            ]

        text = self.processor.apply_chat_template(
            conversation,
            load_audio_from_video=False,
            add_generation_prompt=True,
            tokenize=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos, 
            return_tensors="pt", padding=True, use_audio_in_video=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            text_ids, _ = self.model.generate(
                **inputs, use_audio_in_video=False, do_sample=False, max_new_tokens=512
            )

        resp = self.processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Clean the output
        if "assistant\n" in resp:
            return resp.split("assistant\n")[1]
        return resp

if __name__ == "__main__":
    pass