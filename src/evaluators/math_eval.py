import os
import json
import random
from tqdm import tqdm
from src.evaluators.extractor import *

NUM_MQA_TESTS = 50

def evaluate_single_math(model, dataset_dir, metadata_path, output_path):
    print(f"\n--- Starting MQA (Math) Baseline Evaluation ---")
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
            
    test_samples = random.sample(records, min(NUM_MQA_TESTS, len(records)))
    processed_ids = set()
    correct_count = 0
    total_tested = 0
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

    # Prompt design: Ask for a direct answer to simplify parsing
    prompt = "Listen to the audio and solve the math problem. Output ONLY the final numerical answer, without any explanations."
    
    for record in tqdm(test_samples, desc="Evaluating MQA"):
        record_id = record.get("id")
        if record_id in processed_ids:
            continue
        audio_rel_path = record.get("audio_path")
        audio_path = os.path.join(dataset_dir, audio_rel_path)
        ground_truth = str(record['answer']['text'][0].strip())
        
        prediction = model.query(audio_path, prompt).strip()
            
        result_dict = {
            "id": record["id"],
            "question": record["context_transcript"],
            "ground_truth": ground_truth,
            "prediction": prediction,
        }
        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
        
    print(f"\n[ID: {record.get('id', 'N/A')}] GT: {ground_truth} | Pred: {prediction}")

def get_expected_answer(segment):
    """
    Dynamically extracts the ground truth answer string from various schema formats.
    Supports:
    1. segment["expected_answer"] (string or list)
    2. segment["answer"]["text"] (list)
    """
    # 1. Try to fetch direct expected_answer fields
    ans = segment.get("expected_answer") or segment.get("expected_answers")
    
    # 2. Fallback to nested answer["text"] structure if the direct field is missing
    if ans is None and "answer" in segment and isinstance(segment["answer"], dict):
        ans = segment["answer"].get("text")
        
    # 3. Handle list structures by extracting the first available element
    if isinstance(ans, list):
        return str(ans[0]).strip() if ans else ""
        
    return str(ans).strip() if ans is not None else ""

def evaluate_dual_math(model, dataset_dir, metadata_path, output_path):
    print("Starting Sequential MQA Evaluation")
    
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f]
        
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data.get("concat_id"))
                except json.JSONDecodeError:
                    continue
                    
    for record in tqdm(records, desc="Evaluating Dual Math Capability"):
        concat_id = record["concat_id"]
        
        if concat_id in processed_ids:
            continue
            
        audio_path = os.path.join(dataset_dir, record["audio_path"])
        
        # Extract ground truth answers from metadata sequence
        gt_ans_1 = get_expected_answer(record["sequence"][0])
        gt_ans_2 = get_expected_answer(record["sequence"][1])
        
        prompt = (
            "Listen to the audio carefully. It contains exactly TWO distinct math problems played sequentially.\n"
            "Solve both problems in the exact order they appear.\n"
            "Output ONLY the final number for the first problem inside <ans_1> and </ans_1> tags.\n"
            "Output ONLY the final number for the second problem inside <ans_2> and </ans_2> tags.\n"
            "Do not output any reasoning, equations, or other text."
        )
        
        prediction_raw = model.query(audio_path, prompt)
        extracted_ans_1, extracted_ans_2 = extract_dual_math_answers(prediction_raw)
        
        is_correct_1 = (extracted_ans_1 == gt_ans_1)
        is_correct_2 = (extracted_ans_2 == gt_ans_2)
        correct_count = sum([is_correct_1, is_correct_2])
        is_exact_match = (correct_count == 2)
        
        result_dict = {
            "concat_id": concat_id,
            "ground_truth": [gt_ans_1, gt_ans_2],
            "prediction_raw": prediction_raw,
            "extracted_answers": [extracted_ans_1, extracted_ans_2],
            "is_correct_1": is_correct_1,
            "is_correct_2": is_correct_2,
            "correct_count": correct_count,
            "is_exact_match": is_exact_match
        }
        
        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    pass