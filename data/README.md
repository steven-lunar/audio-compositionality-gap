# Data location

No dataset files are committed to this repository.

Download the final Hugging Face dataset into `data/processed/`. This directory
is ignored by Git. Each subset contains `metadata.jsonl` and the audio files
referenced by its `audio_path` field.

Expected primary subset names:

```text
data/processed/
├── env_math_concat_snr_0/
├── env_math_concat_snr_10/
├── env_math_concat_snr_20/
├── env_trivia_qa_concat_snr_0/
├── env_trivia_qa_concat_snr_10/
├── env_trivia_qa_concat_snr_20/
├── emo3way_math_concat/
└── emo3way_trivia_qa_concat/
```
