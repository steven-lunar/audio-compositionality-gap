from src.models.base import BaseAudioModel

# Replace this Hugging Face ID with a local tokenizer path when needed.
MIMO_TOKENIZER_PATH = "XiaomiMiMo/MiMo-Audio-Tokenizer"


class MiMoAudioModel(BaseAudioModel):
    """Thin adapter around the official MiMo-Audio inference package."""

    def __init__(self, model_name, device="cuda"):
        super().__init__(model_name, device)
        try:
            from mimo_audio.mimo_audio import MimoAudio
        except ImportError as exc:
            raise ImportError(
                "MiMo-Audio inference code is not bundled. Clone the official "
                "XiaomiMiMo/MiMo-Audio repository, install its dependencies, "
                "and add its `src` directory to PYTHONPATH."
            ) from exc
        self.tokenizer_path = MIMO_TOKENIZER_PATH
        self.model = MimoAudio(model_name, self.tokenizer_path)

    def query(self, audio_path, text_prompt):
        if audio_path is not None:
            return self.model.audio_understanding_sft(audio_path, text_prompt)
        return self.model.text_dialogue_sft(text_prompt, thinking=False)
