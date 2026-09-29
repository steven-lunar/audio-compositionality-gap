# Audio Composition Diagnostics

Code for a controlled diagnostic study of capability composition in large
audio-language models (LALMs). Each example contains two sequential speech
segments. A model must recognize an acoustic cue, bind that cue to the correct
segment, and execute a downstream task on that segment.

The released code supports three cue families:

- background environment/noise;
- speaker gender;
- speech emotion.

Downstream tasks include ASR, arithmetic QA, and TriviaQA. The evaluation
separates atomic recognition, explicit segment selection, integrated
composition, output-format compliance, and distractor-answer errors.

## Release scope

This repository contains:

- adapters for the four models evaluated in the primary study;
- atomic and compositional evaluation functions;
- strict and LLM-assisted graders;
- ASR WER, segment-selection, and output-format evaluation utilities.

The final evaluation dataset is distributed separately through Hugging Face.
This repository intentionally does **not** include dataset-construction code,
audio files, model weights, raw predictions, derived result tables, or
vendored third-party model repositories. See [data/README.md](data/README.md)
for the expected dataset layout.

## Repository layout

```text
.
├── baseline_eval.py                 # Main evaluation CLI
├── src/
│   ├── models/                      # Four lightweight model adapters
│   └── evaluators/                  # Atomic/compositional task evaluators
├── LLM_parser.py                    # Optional lenient answer extraction
├── grader.py                        # Strict vs. lenient QA grading
├── dual_atomic_grader.py            # Attribute-task regrading
├── grade_seg_idx.py                 # Segment-selection diagnostics
├── asr_wer_eval.py                  # ASR and distractor WER evaluation
├── format_eval.py                   # Tag-format compliance
├── environment.yml                  # Default Conda environment
├── data/                             # Local data mount (Git-ignored)
└── outputs/                          # Predictions/graded files (Git-ignored)
```

## Installation

Conda is the default environment manager. Create the supplied Python 3.10
environment from the repository root:

```bash
conda env create -f environment.yml
conda activate audio-composition
```

The environment installs the Python dependencies through pip inside Conda.
If your system needs a different CUDA-specific PyTorch build, install the
appropriate PyTorch package in this environment before running evaluation.

The four primary aliases are:

| Alias | Default model |
|---|---|
| `qwen25_omni` | `Qwen/Qwen2.5-Omni-7B` |
| `minicpm` | `openbmb/MiniCPM-o-4_5` |
| `mimo` | `XiaomiMiMo/MiMo-Audio-7B-Instruct` |
| `kimi` | `moonshotai/Kimi-Audio-7B-Instruct` |

Model locations are configured in the model package rather than passed to
the evaluation CLI. To use local checkpoints, edit `MODEL_PATHS` in
`src/models/__init__.py`. For MiMo, edit `MIMO_TOKENIZER_PATH` in
`src/models/mimo_audio.py`. The committed defaults are public Hugging Face
IDs and contain no machine-specific paths.

For gated models, authenticate with the Hugging Face CLI or set `HF_TOKEN`.
Never place a token in source code. Copy `.env.example` only as a reference;
the scripts do not automatically load `.env` files.

### Model setup

Qwen2.5-Omni and MiniCPM-o run through Transformers. With the default entries
in `MODEL_PATHS`, their weights are downloaded from Hugging Face on first
use; no separate source repository is needed.

The Qwen adapter uses FlashAttention 2, matching the study configuration.
Install a build compatible with your PyTorch and CUDA versions before running
Qwen, or change `attn_implementation` in `src/models/qwen25_omni.py` if you
intend to use another supported attention backend.

Kimi-Audio requires its official inference package:

```bash
pip install git+https://github.com/MoonshotAI/Kimi-Audio.git
```

MiMo-Audio requires its official inference code. Clone it anywhere outside
this repository, install the dependencies required by the official project,
and expose its `src` directory to Python:

```bash
git clone https://github.com/XiaomiMiMo/MiMo-Audio.git /path/to/MiMo-Audio
pip install -r /path/to/MiMo-Audio/requirements.txt
pip install flash-attn==2.7.4.post1
export PYTHONPATH="/path/to/MiMo-Audio/src:${PYTHONPATH}"
```

The official MiMo setup currently specifies Python 3.12 and CUDA 12 or newer.
If those requirements conflict with the default environment, create a
separate Conda environment for MiMo and install this repository's
requirements in that environment.

All four adapters accept either the committed Hugging Face ID or a local
checkpoint directory. To download weights explicitly, use:

```bash
hf download Qwen/Qwen2.5-Omni-7B \
  --local-dir /path/to/Qwen2.5-Omni-7B

hf download openbmb/MiniCPM-o-4_5 \
  --local-dir /path/to/MiniCPM-o-4_5

hf download XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --local-dir /path/to/MiMo-Audio-7B-Instruct
hf download XiaomiMiMo/MiMo-Audio-Tokenizer \
  --local-dir /path/to/MiMo-Audio-Tokenizer

hf download moonshotai/Kimi-Audio-7B-Instruct \
  --local-dir /path/to/Kimi-Audio-7B-Instruct
```

After a manual download, replace the relevant value in `MODEL_PATHS` with
the local directory. For MiMo, also replace `MIMO_TOKENIZER_PATH` with its
local tokenizer directory. These settings intentionally live in
`src/models/`, not in `baseline_eval.py` command-line arguments. Refer to
the official [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni),
[MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o-Demo),
[MiMo-Audio](https://github.com/XiaomiMiMo/MiMo-Audio), and
[Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio) repositories if their
upstream dependency requirements change.

## Dataset location

The final evaluation dataset is released separately on Hugging Face:

```text
YOUR_ORG/YOUR_DATASET
```

After replacing the placeholder with the released dataset ID, download the
repository directly into the ignored local data directory:

```bash
hf download YOUR_ORG/YOUR_DATASET \
  --repo-type dataset \
  --local-dir data/processed
```

Alternatively, copy an existing local snapshot into `data/processed/`.
Each subset must have this structure:

```text
<dataset_name>/
├── metadata.jsonl
└── audios/...
```

Each metadata row must contain a unique `concat_id`, a relative `audio_path`,
and a two-element `sequence`. Primary prepared subset names are listed in
[data/README.md](data/README.md).

## Run evaluation

Run commands from the repository root. The evaluator appends one JSON object
per example and resumes by skipping completed `concat_id` values.

```bash
python baseline_eval.py \
  --model qwen25_omni \
  --task env_math_select \
  --dataset_dir data/processed \
  --dataset_name env_math_concat_snr_10 \
  --output_dir outputs/raw \
  --output_file qwen25_omni_env_math_concat_snr_10_env_math_select.jsonl \
  --device cuda
```

Main task families:

- atomic: `env_dual_base`, `gender_dual_base`, `emo_dual_base`,
  `math_dual_base`, `trivia_qa_dual_base`, `asr_dual_base`;
- explicit selection: `env_seg_idx_select`, `gender_seg_idx_select`,
  `emo_seg_idx_select`;
- composition: `<cue>_math_select`, `<cue>_trivia_qa_select`, and
  `<cue>_asr_select`, where `<cue>` is `env`, `gender`, or `emo`;
- CoT controls: the corresponding `*_select_cot` tasks.

Run `python baseline_eval.py --help` for the authoritative choices.

## Parse and grade saved predictions

Raw predictions should be treated as immutable. Always write parser and grader
outputs to new paths. The segment count depends on the evaluation family:

| Evaluation family | Answers processed | Parser option | Follow-up evaluator |
|---|---:|---|---|
| Atomic QA (`math_dual_base`, `trivia_qa_dual_base`) | Both segments | `--num-segments 2` | `grader.py --num_segments 2` |
| Compositional QA (`<cue>_math_select`, `<cue>_trivia_qa_select`) | Selected target only | `--num-segments 1` | `grader.py --num_segments 1` |
| Atomic ASR (`asr_dual_base`) | Both segments | `--num-segments 2` | `asr_wer_eval.py --mode concat_asr` |
| Compositional ASR (`<cue>_asr_select`) | Selected target only | `--num-segments 1` | `asr_wer_eval.py --mode conditional_asr` |

`LLM_parser.py` uses a hyphen in `--num-segments`, while `grader.py` uses an
underscore in `--num_segments`. The QA tag names must also match the task:
atomic QA outputs use `ans_1 ans_2`, whereas compositional QA outputs use
the single tag `ans`.

The acoustic-attribute atomic tasks (`env_dual_base`, `gender_dual_base`,
and `emo_dual_base`) also predict both segments, but they are handled by
`dual_atomic_grader.py` rather than the LLM-parser/QA-grader workflow below.

Atomic math or TriviaQA parsing and grading:

```bash
export OPENAI_API_KEY=...
python LLM_parser.py \
  --task-type math \
  --num-segments 2 \
  --input-file outputs/raw/atomic_example.jsonl \
  --output-file outputs/parsed/atomic_example.jsonl

python grader.py \
  --input_file outputs/parsed/atomic_example.jsonl \
  --output_file outputs/graded/atomic_example.jsonl \
  --tags ans_1 ans_2 \
  --answer_type math \
  --num_segments 2
```

Compositional math or TriviaQA parsing and grading:

```bash
python LLM_parser.py \
  --task-type math \
  --num-segments 1 \
  --input-file outputs/raw/composition_example.jsonl \
  --output-file outputs/parsed/composition_example.jsonl

python grader.py \
  --input_file outputs/parsed/composition_example.jsonl \
  --output_file outputs/graded/composition_example.jsonl \
  --tags ans \
  --answer_type math \
  --num_segments 1
```

For TriviaQA, replace `--task-type math` and `--answer_type math` with
`trivia`. ASR uses the same parser segment-count rule, followed by
`asr_wer_eval.py` instead of `grader.py`.

Other diagnostics:

```bash
python grade_seg_idx.py outputs/raw/*_seg_idx_select.jsonl
python format_eval.py --task env_math_select --results_file outputs/raw/example.jsonl
python asr_wer_eval.py --help
```

## Reproducibility and safety

- The released dataset fixes the item pairing used in the study.
- Model generation is deterministic where the upstream API supports it.
- Raw predictions, parsed predictions, and graded outputs use separate files.
- `data/`, `outputs/`, checkpoints, audio, JSONL, and
  common weight formats are ignored by Git.
- Credentials are read from environment variables only.

Before publishing a fork, run:

```bash
python -m compileall -q .
git check-ignore data/example.wav outputs/example.jsonl weights/model.safetensors
rg -n --hidden '(sk-|hf_|/home/|/tmp/|/tmp2/)' . \
  --glob '!.git/**' --glob '!.env.example'
```

## License

Code in this repository is released under the MIT License. Models, the
separately released dataset, and external inference repositories retain their
own licenses and terms.

## Citation

If you find this work useful, please cite our paper:

**Diagnostic Study of Capability Composition in Large Audio-Language Models**  
Author A, Author B, and Author C  
[arXiv:ARXIV_ID](https://arxiv.org/abs/ARXIV_ID)

```bibtex
@misc{authorYEARaudio_composition,
  title         = {Diagnostic Study of Capability Composition in Large Audio-Language Models},
  author        = {Author A and Author B and Author C},
  year          = {YEAR},
  eprint        = {ARXIV_ID},
  archivePrefix = {arXiv},
  primaryClass  = {PRIMARY_CLASS},
  url           = {https://arxiv.org/abs/ARXIV_ID}
}
```

Replace `Author A and Author B and Author C`, `YEAR`, `ARXIV_ID`, and
`PRIMARY_CLASS` with the final arXiv metadata before release.
