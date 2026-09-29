import torch
from transformers import AutoModel
import librosa
from src.models.base import BaseAudioModel

class MiniCPMModel(BaseAudioModel):
    def __init__(self, model_name="openbmb/MiniCPM-o-4_5", device="cuda"):
        super().__init__(model_name, device)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="sdpa", # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=False,
            init_audio=True,
            init_tts=True,
        )
        self.model.eval().cuda()
        self.model.init_tts()

    def query(self, audio_path, text_prompt):
        if audio_path is not None:
            audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
            msgs = [{"role": "user", "content": [text_prompt, audio_input]}]
        else:
            msgs = [{"role": "user", "content": [text_prompt]}]
        
        res = self.model.chat(
            msgs=msgs,
            do_sample=False,
            max_new_tokens=512,
            use_tts_template=False,
            generate_audio=False,
            output_audio_path='',
        )
        return res