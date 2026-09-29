from src.models.base import BaseAudioModel


class KimiAudioModel(BaseAudioModel):
    """Thin adapter around the official Kimi-Audio inference package."""

    def __init__(self, model_name, device="cuda"):
        super().__init__(model_name, device)
        try:
            from kimia_infer.api.kimia import KimiAudio
        except ImportError as exc:
            raise ImportError(
                "Kimi-Audio inference code is not bundled. Install the official "
                "package with `pip install "
                "git+https://github.com/MoonshotAI/Kimi-Audio.git`."
            ) from exc
        self.model = KimiAudio(model_path=model_name, load_detokenizer=False)
        self.sampling_params = {
            "audio_temperature": 0.0,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
            "max_new_tokens": 128,
        }

    def query(self, audio_path, text_prompt):
        if audio_path is not None:
            messages = [
                {"role": "user", "message_type": "text", "content": text_prompt},
                {"role": "user", "message_type": "audio", "content": audio_path},
            ]
            self.sampling_params["max_new_tokens"] = 256
        else:
            messages = [
                {"role": "user", "message_type": "text", "content": text_prompt}
            ]
            self.sampling_params["max_new_tokens"] = 32
        _, text_output = self.model.generate(
            messages, **self.sampling_params, output_type="text"
        )
        return text_output
