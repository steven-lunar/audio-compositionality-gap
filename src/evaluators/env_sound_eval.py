import json
import os
import random
from collections import defaultdict
from tqdm import tqdm
from src.evaluators.extractor import *

NUM_ESC50_TESTS_PER_CAT = 10 
TARGET_CATEGORIES = [
    "rain", "sea_waves", "crickets", "chirping_birds", "wind" 
]

def evaluate_single_env_sound(model, dataset_dir, metadata_path, output_path):
    print(f"\n--- Starting ESC-50 (Audio Event) Baseline Evaluation ---")
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get("category") in TARGET_CATEGORIES:
                records.append(r)
                
    # Group by category to ensure balanced testing
    grouped_records = defaultdict(list)
    for r in records:
        grouped_records[r["category"]].append(r)
    
    processed_ids = set()
    correct_count = 0
    total_tested = 0
    
    # Load existing progress if the file exists
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data["id"])
                    total_tested += 1
                    if data.get("is_correct"):
                        correct_count += 1
                except json.JSONDecodeError:
                    continue
    
    # Prompt design: Multi-choice formulation heavily reduces model hallucination
    options_str = ", ".join(TARGET_CATEGORIES)
    prompt = f"Identify the primary environmental sound in this audio. Choose exactly ONE category from the following list: [{options_str}]. Output only the category name."
    
    for category, category_records in grouped_records.items():
        # Sample N records per category (Ensure random.seed() is set in main() for reproducibility)
        test_samples = random.sample(category_records, min(NUM_ESC50_TESTS_PER_CAT, len(category_records)))
        
        for record in tqdm(test_samples, desc=f"Evaluating {category}"):
            record_id = record.get("id")
            
            # Skip if this specific audio has already been evaluated in a previous run
            if record_id in processed_ids:
                continue
                
            audio_path = os.path.join(dataset_dir, record["audio_path"])
            
            # Call the fixed Qwen2-Audio inference function
            prediction = model.query(audio_path, prompt).strip().lower()
            
            # Check if the correct category name is in the model's output
            is_correct = (category.lower() == prediction)
            
            if is_correct:
                correct_count += 1
            total_tested += 1
                
            result_dict = {
                "id": record_id,
                "ground_truth": category,
                "prediction": prediction,
                "is_correct": is_correct
            }
            
            # Save immediately (Append mode)
            with open(output_path, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
                
            # Print real-time results for monitoring
            print(f"\n[ID: {record_id}] GT: {category} | Pred: {prediction} | Correct: {is_correct}")

def evaluate_dual_env_sound(model, dataset_dir, metadata_path, output_path):
    print("Starting Sequential Audio Evaluation")
    
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f]
        
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data["concat_id"])
                except json.JSONDecodeError:
                    continue
                    
    for record in tqdm(records, desc="Evaluating Sequence"):
        concat_id = record["concat_id"]
        
        if concat_id in processed_ids:
            continue
            
        audio_path = os.path.join(dataset_dir, record["audio_path"])
        
        # Ground truth is now expected to be a list, e.g., ["rain", "sea_waves"]
        expected_list = [ans.lower() for ans in record["expected_output"]]
        
        categories_str = ", ".join(TARGET_CATEGORIES)
        
        prompt = (
            f"Listen to the audio carefully. It contains exactly TWO distinct environmental sounds "
            f"played sequentially.\n"
            f"Identify both sounds in the exact order they appear.\n"
            f"CRITICAL INSTRUCTION: You MUST choose your answers ONLY from the following list of categories:\n"
            f"[{categories_str}]\n\n"
            f"You MUST wrap your answer for the first sound inside <event_1> and </event_1> tags.\n"
            f"You MUST wrap your answer for the second sound inside <event_2> and </event_2> tags.\n"
            f"Do not output anything else."
        )
        
        prediction_raw = model.query(audio_path, prompt).strip()
        prediction_list = extract_dual_environment_events(prediction_raw)
        
        # Calculate sequence accuracy metrics
        correct_count = 0
        for expected, predicted in zip(expected_list, prediction_list):
            if expected == predicted:
                correct_count += 1
                
        is_exact_match = (correct_count == 2)
        is_partial_match = (correct_count == 1)
        
        result_dict = {
            "concat_id": concat_id,
            "expected_list": expected_list,
            "prediction_raw": prediction_raw,
            "prediction_list": prediction_list,
            "is_exact_match": is_exact_match,
            "is_partial_match": is_partial_match,
            "correct_count": correct_count
        }
        
        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

def evaluate_dual_env_snr(model, dataset_dir, metadata_path, output_path):
    """
    Atomic environmental-sound classification on the SAME noise-mixed,
    two-segment compositional audio used by the env_{math,trivia_qa,asr}_select
    tasks (i.e. dataset_dir should point at an env_math_concat_snr_X /
    env_trivia_qa_concat_snr_X directory, NOT the clean two-segment env audio
    used by evaluate_duel_env_sound).
 
    Unlike evaluate_duel_env_sound (clean audio, ground truth from a plain
    `expected_output` list), ground truth here comes from each segment's
    `bg_category` field in `record["sequence"]`, ordered by `sequence_index`,
    matching the noise-mixing metadata schema used across the SNR sweep.
 
    This gives the atomic "can it classify background sound correctly, on the
    same audio/SNR conditions as the compositional select tasks" data point
    needed for a "both correct" (isolated classification AND isolated
    QA/ASR) vs. compositional-task-correct comparison.
    """
    print("Starting Dual Environmental Sound Classification (SNR-conditioned audio)")
 
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f]
 
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data["concat_id"])
                except json.JSONDecodeError:
                    continue
 
    categories_str = ", ".join(TARGET_CATEGORIES)
 
    for record in tqdm(records, desc="Evaluating Dual Env (SNR)"):
        concat_id = record["concat_id"]
 
        if concat_id in processed_ids:
            continue
 
        audio_path = os.path.join(dataset_dir, record["audio_path"])
 
        # Ground truth from each segment's bg_category, ordered by sequence_index
        # (matches the noise-mixing metadata schema, e.g.
        # {"sequence_index": 1, "bg_category": "chirping_birds", "snr_db": 10.0, ...})
        ordered_segments = sorted(record["sequence"], key=lambda s: s["sequence_index"])
        ground_truth = [seg["bg_category"].lower() for seg in ordered_segments]
 
        prompt = (
            f"Listen to the audio carefully. It contains exactly TWO distinct trivia or math questions "
            f"played sequentially, each mixed with a different background sound.\n"
            f"Identify the background sound present in each segment, in the order they appear.\n"
            f"CRITICAL INSTRUCTION: You MUST choose your answers ONLY from the following list of categories:\n"
            f"[{categories_str}]\n\n"
            f"You MUST wrap your answer for the first segment's background sound inside <event_1> and </event_1> tags.\n"
            f"You MUST wrap your answer for the second segment's background sound inside <event_2> and </event_2> tags.\n"
            f"Do not output anything else."
        )
 
        prediction_raw = model.query(audio_path, prompt).strip()
        prediction_list = extract_dual_environment_events(prediction_raw)
 
        is_correct_1 = (prediction_list[0] == ground_truth[0])
        is_correct_2 = (prediction_list[1] == ground_truth[1])
        correct_count = int(is_correct_1) + int(is_correct_2)
        is_exact_match = (correct_count == 2)
        is_partial_match = (correct_count == 1)
 
        result_dict = {
            "concat_id": concat_id,
            "ground_truth": ground_truth,
            "prediction_raw": prediction_raw,
            "extracted_answers": prediction_list,
            "is_correct_1": is_correct_1,
            "is_correct_2": is_correct_2,
            "correct_count": correct_count,
            "is_exact_match": is_exact_match,
            "is_partial_match": is_partial_match,
        }
 
        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    pass