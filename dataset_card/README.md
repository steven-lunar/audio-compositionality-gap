---
pretty_name: Audio Composition Diagnostics
language:
- en
license: other
task_categories:
- audio-text-to-text
- automatic-speech-recognition
- question-answering
- audio-classification
tags:
- audio
- speech
- large-audio-language-models
- compositional-reasoning
- synthetic-speech
- benchmark
- environmental-sound
- emotion-recognition
- gender-recognition
- arxiv:2609.36921
size_categories:
- 1K<n<10K
configs:
- config_name: env_math_concat_snr_0
  data_files:
  - split: test
    path: env_math_concat_snr_0/metadata.jsonl
- config_name: env_math_concat_snr_10
  data_files:
  - split: test
    path: env_math_concat_snr_10/metadata.jsonl
- config_name: env_math_concat_snr_20
  data_files:
  - split: test
    path: env_math_concat_snr_20/metadata.jsonl
- config_name: env_trivia_qa_concat_snr_0
  data_files:
  - split: test
    path: env_trivia_qa_concat_snr_0/metadata.jsonl
- config_name: env_trivia_qa_concat_snr_10
  data_files:
  - split: test
    path: env_trivia_qa_concat_snr_10/metadata.jsonl
- config_name: env_trivia_qa_concat_snr_20
  data_files:
  - split: test
    path: env_trivia_qa_concat_snr_20/metadata.jsonl
- config_name: emo3way_math_concat
  data_files:
  - split: test
    path: emo3way_math_concat/metadata.jsonl
- config_name: emo3way_trivia_qa_concat
  data_files:
  - split: test
    path: emo3way_trivia_qa_concat/metadata.jsonl
---

# Audio Composition Diagnostics

## Dataset Description

Audio Composition Diagnostics is a controlled evaluation benchmark for
studying whether large audio-language models (LALMs) can compose capabilities
that they possess individually. Each example contains two sequential English
speech segments. A textual instruction identifies a target by an acoustic
attribute, and the model must bind that attribute to the correct segment before
transcribing its speech or answering its question.

The benchmark supports three acoustic cue families:

- environmental sound: `rain`, `crickets`, `wind`, `sea_waves`, or
  `chirping_birds`;
- speaker gender: `female` or `male`, following the binary labels used during
  controlled synthesis;
- intended speech emotion: `angry`, `sad`, or `happy`.

The downstream tasks are automatic speech recognition (ASR), arithmetic
question answering, and factual question answering based on TriviaQA. The same
audio can be evaluated at four diagnostic levels: atomic attribute recognition,
cue-conditioned segment selection, atomic downstream execution, and complete
cue-conditioned composition.

This dataset accompanies the paper
[When Capabilities Fail to Compose: Diagnosing the Compositionality Gap in
Large Audio-Language Models](https://arxiv.org/abs/2609.36921). Evaluation code
and task prompts are available in the
[Audio Compositionality Gap repository](https://github.com/steven-lunar/audio-compositionality-gap).

### Dataset Summary

- 4,800 two-segment examples in eight configurations
- 4,800 mono, 24 kHz, 16-bit PCM WAV files
- approximately 17.617 hours of audio and 2.840 GiB on disk
- 600 examples in every configuration
- one 1.5-second silence interval between the two segments
- English speech and text
- evaluation-only release; no train/validation split
- metadata and audio only; model predictions are not included

The 17.617-hour total counts the 0, 10, and 20 dB versions of an environmental
example as separate audio files. Those SNR configurations share pair identities
and should not be treated as independent semantic samples.

### Intended Uses

The dataset is intended for research on:

- compositional capability evaluation in audio-language models;
- acoustic-attribute recognition and attribute-to-segment binding;
- cue-conditioned ASR and spoken question answering;
- robustness to environmental-cue salience;
- positional bias, output-format compliance, and error analysis.

It is designed as a diagnostic benchmark, not as a general-purpose speech
training corpus. If aggregate scores are reported, results should also be
broken down by cue type, downstream task, SNR, and diagnostic stage.

## Dataset Structure

### Configurations

Every configuration contains one `test` split.

| Configuration | Downstream content | Available cue evaluation | SNR | Examples | Duration | Approx. size |
|---|---|---|---:|---:|---:|---:|
| `env_math_concat_snr_0` | Arithmetic QA / ASR | Environmental sound | 0 dB | 600 | 2.053 h | 338.8 MiB |
| `env_math_concat_snr_10` | Arithmetic QA / ASR | Environmental sound | 10 dB | 600 | 2.053 h | 338.8 MiB |
| `env_math_concat_snr_20` | Arithmetic QA / ASR | Environmental sound | 20 dB | 600 | 2.053 h | 338.8 MiB |
| `env_trivia_qa_concat_snr_0` | Factual QA / ASR | Environmental sound | 0 dB | 600 | 2.346 h | 387.5 MiB |
| `env_trivia_qa_concat_snr_10` | Factual QA / ASR | Environmental sound | 10 dB | 600 | 2.346 h | 387.5 MiB |
| `env_trivia_qa_concat_snr_20` | Factual QA / ASR | Environmental sound | 20 dB | 600 | 2.346 h | 387.5 MiB |
| `emo3way_math_concat` | Arithmetic QA / ASR | Gender or emotion | N/A | 600 | 2.042 h | 337.0 MiB |
| `emo3way_trivia_qa_concat` | Factual QA / ASR | Gender or emotion | N/A | 600 | 2.377 h | 392.5 MiB |

In environmental configurations, SNR is the speech-to-background ratio. A
higher SNR makes speech clearer but the environmental cue less salient. The
three SNR versions for a downstream task contain the same 600 pair identities,
mixture assignments, and durations; only the mixing level changes.

The two `emo3way_*` configurations are shared by gender- and
emotion-conditioned evaluations. They are not emotion-only datasets even
though their legacy `cue_type` value is `emotion`.

### Repository Layout

```text
.
├── README.md
├── env_math_concat_snr_0/
│   ├── metadata.jsonl
│   └── audios/
│       ├── bgnoise_math_000000.wav
│       └── ...
├── env_math_concat_snr_10/
├── env_math_concat_snr_20/
├── env_trivia_qa_concat_snr_0/
├── env_trivia_qa_concat_snr_10/
├── env_trivia_qa_concat_snr_20/
├── emo3way_math_concat/
└── emo3way_trivia_qa_concat/
```

Each configuration directory contains 600 audio files and one
`metadata.jsonl`. The `audio_path` in each row is relative to that
configuration directory.

### Data Fields

All metadata rows contain:

| Field | Type | Description |
|---|---|---|
| `concat_id` | string | Pair identifier, unique within a configuration. Environmental SNR variants deliberately reuse the same identifiers. |
| `audio_path` | string | Relative path to the concatenated WAV file, normally `audios/<concat_id>.wav`. This is the field to use for loading audio. |
| `cue_type` | string | Construction-time cue label: `bg_noise` for environmental configurations and the legacy value `emotion` for `emo3way_*`. |
| `order` | string | `target_first` or `target_second`. |
| `target_index` | integer | Zero-based target segment index: `0` or `1`. |
| `sequence` | list of two objects | Ground-truth metadata for the two segments in temporal order. |

Environmental configurations additionally contain:

| Field | Type | Description |
|---|---|---|
| `target_bg_category` | string | Environmental class associated with the target segment. |
| `distractor_bg_category` | string | Environmental class associated with the non-target segment. |

The shared gender/emotion configurations additionally contain:

| Field | Type | Description |
|---|---|---|
| `target_gender` | string | Gender cue associated with the target segment. |
| `distractor_gender` | string | Gender cue associated with the non-target segment. |
| `target_emotion` | string | Intended emotion associated with the target segment. |
| `distractor_emotion` | string | Intended emotion associated with the non-target segment. |

Every object in `sequence` contains:

| Field | Type | Description |
|---|---|---|
| `id` | string | Source item or synthesized-utterance identifier. |
| `gender` | string | Segment-level `female` or `male` synthesis label. |
| `emotion` | string | Segment-level intended emotion label. Environmental configurations may include additional construction-time emotion values beyond the three evaluated in `emo3way_*`. |
| `bg_category` | string | Environmental class in `env_*`; see the schema caveat below for `emo3way_*`. |
| `question` | string | Spoken transcript and downstream question. It is also the ASR reference. |
| `source_audio_path` | string | Relative provenance path used during construction. The pre-concatenation files are not part of this release; do not use this field to load benchmark audio. |
| `expected_answer` | string | Arithmetic answer. Present only in math configurations. |
| `expected_answers` | list of strings | Accepted normalized aliases for factual QA. Present only in TriviaQA configurations. |
| `snr_db` | float | Speech-to-background SNR. Present only in environmental configurations. |

#### Schema Caveats

- In `emo3way_*`, the legacy segment-level `bg_category` field duplicates the
  segment's `gender` value (`female` or `male`). It is not an environmental
  label and should be ignored. Use `gender` and `emotion` directly.
- In `emo3way_*`, `cue_type` is always `emotion`, although the same files are
  intentionally used for both gender- and emotion-conditioned tasks. Select
  the relevant `target_*` and `distractor_*` fields based on the task.
- `source_audio_path` records an internal, relative construction path and is
  not expected to resolve in the published repository. Use `audio_path` for
  all released audio.
- `target_index` is zero-based, while prompts shown to models may describe the
  segments as first and second.

### Example Record

The following abbreviated record illustrates the math schema:

```json
{
  "concat_id": "emotion_math_000000",
  "audio_path": "audios/emotion_math_000000.wav",
  "cue_type": "emotion",
  "order": "target_first",
  "target_index": 0,
  "target_gender": "female",
  "target_emotion": "sad",
  "distractor_gender": "male",
  "distractor_emotion": "happy",
  "sequence": [
    {
      "id": "73_female_sad_v2",
      "gender": "female",
      "emotion": "sad",
      "question": "compute 274 plus 30",
      "expected_answer": "304"
    },
    {
      "id": "27_male_happy_v1",
      "gender": "male",
      "emotion": "happy",
      "question": "what's 355 minus 900",
      "expected_answer": "-545"
    }
  ]
}
```

### Balance and Reuse

Within each configuration:

- the target appears first in 300 examples and second in 300 examples;
- each environmental target and distractor class appears 120 times;
- each intended emotion is the target 200 times and the distractor 200 times
  in `emo3way_*`;
- each gender label is the target 300 times and the distractor 300 times in
  `emo3way_*`.

The release contains 100 unique arithmetic question texts and 910 unique
factual question texts across all configurations. Questions and synthesized
utterances are intentionally reused across cue conditions. The three SNR
configurations for a given downstream task are paired interventions, not three
independent samples. Statistical analysis should preserve this pairing, for
example by comparing results by `concat_id` across SNR values.

## Download and Use

Replace `YOUR_HF_NAMESPACE/audio-composition-diagnostics` below with the final
dataset repository ID.

```bash
pip install -U huggingface_hub

hf download YOUR_HF_NAMESPACE/audio-composition-diagnostics \
  --repo-type dataset \
  --local-dir data/processed
```

The benchmark's canonical representation is WAV plus JSONL. Downloading a
snapshot preserves the relative paths expected by the evaluation code:

```python
import json
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(
    snapshot_download(
        repo_id="YOUR_HF_NAMESPACE/audio-composition-diagnostics",
        repo_type="dataset",
    )
)

subset = root / "env_math_concat_snr_10"
with (subset / "metadata.jsonl").open(encoding="utf-8") as f:
    example = json.loads(next(f))

audio_file = subset / example["audio_path"]
target_segment = example["sequence"][example["target_index"]]

print(audio_file)
print(target_segment["question"])
print(target_segment["expected_answer"])
```

For the paper's evaluation workflow, download the data into
`data/processed/` in the code repository and pass the configuration directory
name through `--dataset_name`. See the code repository README for model setup,
prompts, parsing, and grading commands.

## Dataset Creation

### Source Data

- Speech was synthesized with
  [CosyVoice 3](https://github.com/QwenAudio/CosyVoice).
- Reference speech came from six gender-balanced speakers in
  [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), with angry,
  sad, and happy reference conditions.
- Arithmetic content came from
  [Spoken-MQA](https://huggingface.co/datasets/amao0o0/spoken-mqa).
- Factual questions and answer aliases came from
  [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa); the
  1,000-question audio subset used during construction was obtained from
  [fixie-ai/trivia_qa-audio](https://huggingface.co/datasets/fixie-ai/trivia_qa-audio).
- Environmental recordings came from
  [ESC-50](https://github.com/karolpiczak/ESC-50). The five selected classes
  are wind, sea waves, rain, crickets, and chirping birds.

### Construction Process

Speech was synthesized from the math or factual question text. For
environmental conditions, a selected ESC-50 recording was looped or cropped to
the speech duration and mixed using RMS energy at 0, 10, or 20 dB
speech-to-background SNR. Two segments with distinct values of the evaluated
attribute were then concatenated with 1.5 seconds of silence, ensuring a unique
target for the corresponding cue.

Target attributes, distractor attributes, and target positions were balanced.
The same synthesized speech serves as the input for QA and, using `question` as
the reference transcript, ASR evaluation.

### Quality Control

The construction pipeline used automatic checks with emotion2vec for emotion
consistency, UTMOSv2 for speech quality, and Whisper for transcription accuracy
against the source text. The synthesized samples were also manually inspected
for intelligibility, attribute consistency, and label correctness. Invalid
samples were regenerated or removed.

The published files were additionally checked for:

- exactly 600 JSONL rows and 600 referenced WAV files per configuration;
- unique `concat_id` values within every configuration;
- exactly two sequence entries per example;
- no missing `audio_path` files;
- consistent mono, 24 kHz, 16-bit PCM WAV encoding;
- balanced target position and target/distractor cue counts.

## Evaluation

The accompanying code evaluates:

- attribute recognition and segment selection with accuracy;
- arithmetic QA with exact match;
- factual QA with normalized exact match over `expected_answers` aliases;
- ASR with normalized word error rate (WER).

Atomic tasks ask for outputs for both segments in a single response.
Compositional tasks ask for only the output associated with a queried acoustic
cue. This distinction is important when configuring the parser and grader:
atomic outputs contain two segment predictions, while compositional outputs
contain one selected-target prediction.

No model predictions or baseline result files are included in this dataset.
Refer to the paper for the reported results on Qwen2.5-Omni, MiniCPM-o,
MiMo-Audio, and Kimi-Audio.

## Considerations for Using the Data

### Known Limitations

- All speech is synthesized. Performance may reflect artifacts of the TTS
  system or its prompting rather than behavior on natural conversational audio.
- Only English and a small, controlled set of acoustic attributes are covered.
- Gender is represented by two source labels used in the experiment. These
  labels do not represent the full range of gender identities and should not be
  interpreted as a person's identity.
- Emotion labels represent intended synthesis conditions, not verified inner
  emotional states. The data is not suitable for inferring emotion from real
  people.
- The benchmark uses six reference speakers, so results may not generalize to
  broader speaker, accent, age, or demographic variation.
- Arithmetic content has only 100 unique question texts and is deliberately
  repeated across synthesized voices and cue conditions.
- Factual questions and alias lists inherit ambiguity, dated facts, and noise
  from TriviaQA and its derived audio subset.
- The SNR configurations and many cross-cue items reuse content. Treating all
  4,800 rows as independent observations will overstate effective sample size.
- The legacy metadata caveats documented above should be handled explicitly by
  custom loaders and analyses.

### Personal and Sensitive Information

The metadata does not include speaker names or direct identifiers. Audio was
synthesized using reference recordings from CREMA-D and may retain
speaker-like characteristics. Users must comply with CREMA-D's terms and
should not use this dataset for speaker identification, biometric inference,
voice impersonation, surveillance, or decisions about real individuals.

Questions inherited from source QA datasets may mention public figures,
places, organizations, or sensitive topics. No additional sensitive-attribute
annotation was created for this release.

### Out-of-Scope Uses

The dataset should not be used to make claims about a real person's gender,
emotion, identity, mental state, or competence. It is not intended for model
training, production decision-making, speaker verification, or biometric
applications.

## Licensing Information

This dataset is a composite release, so its Hugging Face metadata uses
`license: other`. The MIT License in the accompanying code repository applies
to the code only and does not apply to the dataset or upstream source material.

Users are responsible for reviewing and complying with all applicable source
terms. In particular:

- ESC-50 is distributed under CC BY-NC 3.0, with per-clip attribution in its
  upstream license file. The environmental mixtures therefore carry a
  non-commercial restriction from their source material.
- CREMA-D is distributed under ODbL 1.0, with individual database contents
  under the Database Contents License.
- The CosyVoice code is Apache-2.0; that software license does not by itself
  determine the rights attached to generated audio or reference voices.
- The Hugging Face cards for Spoken-MQA and `fixie-ai/trivia_qa-audio` did not
  declare a dataset license when this card was prepared. The original TriviaQA
  Hugging Face card labels its license as unknown.

The maintainers do not grant additional rights to third-party source material.
The release should be treated as research-only unless a user's intended use is
permitted by every applicable upstream term. This section is informational and
is not legal advice.

## Dataset Curators

The dataset was created by the authors of the accompanying paper:

- Chien-Feng Liu
- Chih-Kai Yang
- Bo-Han Feng
- Yu-Hsuan Li Liang
- Hung-yi Lee
- Cheng-Fu Chou

For questions or corrections, open an issue in the
[code repository](https://github.com/steven-lunar/audio-compositionality-gap/issues).

## Citation

If you use this dataset, please cite:

```bibtex
@misc{liu2026capabilitiesfailcomposediagnosing,
      title={When Capabilities Fail to Compose: Diagnosing the Compositionality Gap in Large Audio-Language Models},
      author={Chien-Feng Liu and Chih-Kai Yang and Bo-Han Feng and Yu-Hsuan Li Liang and Hung-yi Lee and Cheng-Fu Chou},
      year={2026},
      eprint={2609.36921},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2609.36921}
}
```

Please also cite the applicable source datasets and generation system when
using the corresponding portions of the release.
