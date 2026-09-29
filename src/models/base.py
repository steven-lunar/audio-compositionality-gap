from abc import ABC, abstractmethod

class BaseAudioModel(ABC):
    """
    Abstract base class for all audio-language models.
    Enforces a strict interface for model implementations.
    """
    def __init__(self, model_name, device="cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def query(self, audio_path, text_prompt):
        """
        Execute inference given an audio file and a text prompt.
        Must be implemented by all subclasses.
        """
        pass

if __name__ == "__main__":
    pass