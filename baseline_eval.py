import argparse
import os
import random
import torch
from src.models import get_model
from src.evaluators import *

TASK_LIST = {
    # "math_single_base": evaluate_single_math,
    # "env_single_base": evaluate_single_env_sound,
    # "env_math_single_base": evaluate_single_env_math,

    "math_dual_base": evaluate_dual_math,
    "env_dual_base": evaluate_dual_bgnoise_snr,
    "asr_dual_base": evaluate_dual_asr,
    "trivia_qa_dual_base": evaluate_dual_trivia,
    "gender_dual_base": evaluate_dual_gender,
    "emo_dual_base": evaluate_dual_emotion,

    "env_asr_select": evaluate_conditional_env_asr,
    "env_math_select": evaluate_conditional_env_math,
    "env_trivia_qa_select": evaluate_conditional_env_trivia,

    "env_math_select_cot": evaluate_conditional_env_math_cot,
    "env_trivia_qa_select_cot": evaluate_conditional_env_trivia_cot,

    "env_seg_idx_select": evaluate_segment_index_env,
    "gender_seg_idx_select": evaluate_segment_index_gender,
    "emo_seg_idx_select": evaluate_segment_index_emotion,

    "gender_asr_select": evaluate_conditional_asr_gender,
    "gender_math_select": evaluate_conditional_gender_math,
    "gender_trivia_qa_select": evaluate_conditional_gender_trivia,
    "gender_math_select_cot": evaluate_conditional_gender_math_cot,
    "gender_trivia_qa_select_cot": evaluate_conditional_gender_trivia_cot,

    "emo_asr_select": evaluate_conditional_asr_emotion,
    "emo_math_select": evaluate_conditional_emotion_math,
    "emo_trivia_qa_select": evaluate_conditional_emotion_trivia,
    "emo_math_select_cot": evaluate_conditional_emotion_math_cot,
    "emo_trivia_qa_select_cot": evaluate_conditional_emotion_trivia_cot
}

DATASET_DIR = {
    "env_concat": "env_concat",
    "math_concat": "math_concat",

    "trivia_qa_concat": "trivia_qa_concat",
    "env_trivia_qa_concat_snr_0": "env_trivia_qa_concat_snr_0",
    "env_trivia_qa_concat_snr_10": "env_trivia_qa_concat_snr_10",
    "env_trivia_qa_concat_snr_20": "env_trivia_qa_concat_snr_20",

    "env_math_base": "env_math_base",
    "env_math_concat": "env_math_concat",
    "env_math_concat_snr_0": "env_math_concat_snr_0",
    "env_math_concat_snr_10": "env_math_concat_snr_10",
    "env_math_concat_snr_20": "env_math_concat_snr_20",

    "gender_math_concat": "gender_math_concat",
    "gender_trivia_qa_concat": "gender_trivia_qa_concat",

    "emo_math_concat": "emo_math_concat",
    "emo_trivia_qa_concat": "emo_trivia_qa_concat",

    "emo3way_math_concat": "emo3way_math_concat",
    "emo3way_trivia_qa_concat": "emo3way_trivia_qa_concat",
}

def parse_args():
    parser = argparse.ArgumentParser(description="LALM Evaluation Pipeline")
    
    parser.add_argument("--task", required=True, choices=list(TASK_LIST.keys()), help="Specify one task to run.") 
    parser.add_argument("--model", type=str, default="qwen25_omni", help="Model alias configured in src/models/__init__.py.")
    parser.add_argument("--dataset_dir", type=str, default="./data/processed", help="Directory containing dataset subdirectories.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to evaluate (e.g., env_base, math_concat).")
    parser.add_argument("--output_dir", type=str, default="./outputs/raw", help="Directory to save evaluation JSONL files.")
    parser.add_argument("--output_file", type=str, required=True, help="Output filename, normally ending in .jsonl.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device for inference.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, f"{args.output_file}")
    dataset_path = os.path.join(args.dataset_dir, DATASET_DIR[args.dataset_name])
    metadata_path = os.path.join(dataset_path, "metadata.jsonl")
    try:
        model = get_model(args.model, device=args.device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model {args.model!r}: {e}") from e


    print(f"Starting Task: {args.task.upper()}")
    
    eval_function = TASK_LIST[args.task]
    eval_function(model, dataset_path, metadata_path, output_path)

if __name__ == "__main__":
    main()
