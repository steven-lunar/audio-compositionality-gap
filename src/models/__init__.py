"""Model factory for the four models evaluated in the paper.

Weights are not bundled with this repository. To use a local checkpoint,
edit the corresponding value in ``MODEL_PATHS`` below.
"""

from __future__ import annotations

# Replace a Hugging Face ID with a local checkpoint path when needed.
MODEL_PATHS = {
    "qwen25_omni": "Qwen/Qwen2.5-Omni-7B",
    "minicpm": "openbmb/MiniCPM-o-4_5",
    "mimo": "XiaomiMiMo/MiMo-Audio-7B-Instruct",
    "kimi": "moonshotai/Kimi-Audio-7B-Instruct",
}


def get_model(model_name: str, device: str = "cuda"):
    """Instantiate a supported model using the path configured above."""
    alias = model_name.lower()
    if alias not in MODEL_PATHS:
        raise ValueError(
            f"Unsupported model {model_name!r}. Supported aliases: "
            + ", ".join(sorted(MODEL_PATHS))
        )
    model_path = MODEL_PATHS[alias]

    if alias == "qwen25_omni":
        from src.models.qwen25_omni import Qwen25OmniModel

        return Qwen25OmniModel(model_name=model_path, device=device)
    if alias == "minicpm":
        from src.models.minicpm import MiniCPMModel

        return MiniCPMModel(model_name=model_path, device=device)
    if alias == "mimo":
        from src.models.mimo_audio import MiMoAudioModel

        return MiMoAudioModel(model_name=model_path, device=device)
    if alias == "kimi":
        from src.models.kimi_audio import KimiAudioModel

        return KimiAudioModel(model_name=model_path, device=device)
