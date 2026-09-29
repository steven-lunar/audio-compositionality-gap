import json
import os
import random
from collections import defaultdict
from tqdm import tqdm
from src.evaluators.extractor import *

EMOTION_ADJ = {
    "angry": "angry",
    "disgust": "disgusted",
    "fear": "fearful",
    "happy": "happy",
    "sad": "sad",
}

def evaluate_conditional_env_asr(model, dataset_dir, metadata_path, output_path):
    print("\n--- Starting Conditional ASR Evaluation ---")
    
    # Load metadata
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f]
        
    # Checkpoint and resume logic
    processed_ids = set()
    total_tested = 0
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data["concat_id"])
                    total_tested += 1
                except json.JSONDecodeError:
                    continue
                    
    if processed_ids:
        print(f"Detected historical progress. {len(processed_ids)} tests completed.")
        print("Automatically skipping completed items...\n")

    # Evaluation loop
    for record in tqdm(records, desc="Evaluating Conditional ASR"):
        concat_id = record["concat_id"]
        
        # Skip if already evaluated
        if concat_id in processed_ids:
            continue
            
        audio_path = os.path.join(dataset_dir, record["audio_path"])
        
        # Extract ordered ground truths directly from the sequence
        ground_truth_1 = record["sequence"][0]["question"]
        ground_truth_2 = record["sequence"][1]["question"]
        
        # Randomly select ONE segment as the target for ASR
        rng = random.Random(concat_id)
        target_segment = rng.choice(record["sequence"])
        target_bg = target_segment["bg_category"]
        ground_truth_text = target_segment["question"]
        
        # Identify distractors
        distractors = [seg for seg in record["sequence"] if seg != target_segment]
        
        # Construct the ASR Prompt
        prompt = (
            f"Listen to the audio carefully. It contains exactly TWO distinct problems played sequentially, "
            f"each mixed with a different background noise.\n"
            f"Find the segment that is accompanied by the sound of '{target_bg}'. "
            f"Please transcribe EXACTLY what is spoken in that specific segment. "
            f"You MUST wrap your final transcription inside <asr> and </asr> tags. "
            f"Do not solve the problem and do not output any other explanations."
        )
        
        prediction_raw = model.query(audio_path, prompt).strip()
        prediction_clean = extract_asr_result(prediction_raw)
        
        total_tested += 1
        
        # Pack results
        result_dict = {
            "concat_id": concat_id,
            "target_bg": target_bg,
            "ground_truth_text": ground_truth_text,
            "ground_truth_1": ground_truth_1,
            "ground_truth_2": ground_truth_2,
            "prediction_raw": prediction_raw,
            "prediction_clean": prediction_clean,
            "all_distractors_info": [
                {
                    "bg_category": d["bg_category"], 
                    "text": d["question"]
                } for d in distractors
            ]
        }
        
        # Save immediately in append mode
        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

    print(f"\n=== ASR Generation Complete. Total Tested: {total_tested} ===")

def evaluate_dual_asr(model, dataset_dir, metadata_path, output_path):
    print("Starting Dual ASR Inference Pipeline...")
    
    
    prompt = (
        "Listen to the audio carefully. It contains exactly TWO distinct problems played sequentially. "
        "Each problem is mixed with a different background noise.\n"
        "Transcribe the problem for both segments exactly as you hear them.\n"
        "Output ONLY the transcription of the first problem inside <asr_1> and </asr_1> tags.\n"
        "Output ONLY the transcription of the second problem inside <asr_2> and </asr_2> tags.\n"
        "Do not output any reasoning, mathematical answers, or other text."
    )
    
    records = []
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            records = [json.loads(line.strip()) for line in f]
    except FileNotFoundError:
        print(f"Error: {metadata_path} not found.")
        return

    # Support for resuming interrupted inference processes
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data.get("concat_id"))
                except json.JSONDecodeError:
                    continue

    for record in tqdm(records, desc="Evaluating Dual ASR"):
        concat_id = record.get("concat_id")
        
        if concat_id in processed_ids:
            continue
            
        audio_path = os.path.join(dataset_dir, record["audio_path"])
        
        # Extract ground truths to save alongside the prediction for easier evaluation later
        gt_text_1 = record["sequence"][0]["question"]
        gt_text_2 = record["sequence"][1]["question"]

        prediction_raw = model.query(audio_path, prompt)
        
        pred_asr_1, pred_asr_2 = extract_dual_transcriptions(prediction_raw)
        
        # Build the result dictionary with separated fields
        result_dict = {
            "concat_id": concat_id,
            "audio_path": audio_path,
            "ground_truth_1": gt_text_1,
            "ground_truth_2": gt_text_2,
            "prediction_raw": prediction_raw,
            "prediction_asr_1": pred_asr_1,
            "prediction_asr_2": pred_asr_2
        }
        
        # Append the raw result immediately to prevent data loss if the script crashes
        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

    print(f"Inference complete. Results saved to {output_path}")

def evaluate_conditional_asr_gender(model, dataset_dir, metadata_path, output_path):
    print("\n--- Starting Conditional ASR Evaluation (gender cue) ---")
 
    # Load metadata
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f]
 
    # Checkpoint and resume logic
    processed_ids = set()
    total_tested = 0
 
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data["concat_id"])
                    total_tested += 1
                except json.JSONDecodeError:
                    continue
 
    if processed_ids:
        print(f"Detected historical progress. {len(processed_ids)} tests completed.")
        print("Automatically skipping completed items...\n")
 
    # Evaluation loop
    for record in tqdm(records, desc="Evaluating Conditional ASR (gender)"):
        concat_id = record["concat_id"]
 
        # Skip if already evaluated
        if concat_id in processed_ids:
            continue
 
        audio_path = os.path.join(dataset_dir, record["audio_path"])
 
        # Extract ordered ground truths directly from the sequence
        ground_truth_1 = record["sequence"][0]["question"]
        ground_truth_2 = record["sequence"][1]["question"]
 
        # Target is FIXED at dataset-construction time -- read it directly,
        # no random selection needed (unlike the noise-cue version).
        target_index = record["target_index"]
        target_gender = record["target_gender"]
        target_segment = record["sequence"][target_index]
        ground_truth_text = target_segment["question"]
 
        # Identify distractors: every segment other than target_index.
        distractors = [
            seg for i, seg in enumerate(record["sequence"]) if i != target_index
        ]
 
        # "math" if this segment has a bare-string expected_answer, "trivia"
        # if it has an expected_answers list -- purely informational output,
        # not used for grading here.
        if "expected_answer" in target_segment:
            task = "math"
        elif "expected_answers" in target_segment:
            task = "trivia"
        else:
            task = None
 
        cue_type = record.get("cue_type", "gender")
 
        # Construct the ASR Prompt -- cue is the speaker's gender instead
        # of a background-noise category.
        prompt = (
            f"Listen to the audio carefully. It contains exactly TWO distinct problems played sequentially, "
            f"each spoken by a speaker of a different gender.\n"
            f"Find the segment spoken by the {target_gender} speaker. "
            f"Please transcribe EXACTLY what is spoken in that specific segment. "
            f"You MUST wrap your final transcription inside <asr> and </asr> tags. "
            f"Do not solve the problem and do not output any other explanations."
        )
 
        prediction_raw = model.query(audio_path, prompt).strip()
        prediction_clean = extract_asr_result(prediction_raw)
 
        total_tested += 1
 
        # Pack results
        result_dict = {
            "concat_id": concat_id,
            "cue_type": cue_type,
            "task": task,
            "target_index": target_index,
            "target_gender": target_gender,
            "ground_truth_text": ground_truth_text,
            "ground_truth_1": ground_truth_1,
            "ground_truth_2": ground_truth_2,
            "prediction_raw": prediction_raw,
            "prediction_clean": prediction_clean,
            "all_distractors_info": [
                {
                    "gender": d["gender"],
                    "text": d["question"],
                } for d in distractors
            ],
        }
 
        # Save immediately in append mode
        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
 
    print(f"\n=== ASR Generation Complete (gender cue). Total Tested: {total_tested} ===")

def evaluate_conditional_asr_emotion(model, dataset_dir, metadata_path, output_path):
    print("\n--- Starting Conditional ASR Evaluation (emotion cue) ---")
 
    # Load metadata
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f]
 
    # Checkpoint and resume logic
    processed_ids = set()
    total_tested = 0
 
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data["concat_id"])
                    total_tested += 1
                except json.JSONDecodeError:
                    continue
 
    if processed_ids:
        print(f"Detected historical progress. {len(processed_ids)} tests completed.")
        print("Automatically skipping completed items...\n")
 
    # Evaluation loop
    for record in tqdm(records, desc="Evaluating Conditional ASR (emotion)"):
        concat_id = record["concat_id"]
 
        # Skip if already evaluated
        if concat_id in processed_ids:
            continue
 
        audio_path = os.path.join(dataset_dir, record["audio_path"])
 
        # Extract ordered ground truths directly from the sequence
        ground_truth_1 = record["sequence"][0]["question"]
        ground_truth_2 = record["sequence"][1]["question"]
 
        # Target is FIXED at dataset-construction time -- read it directly,
        # no random selection needed.
        target_index = record["target_index"]
        target_emotion = record["target_emotion"]
        target_gender = record.get("target_gender")  # informational only, not in prompt
        target_segment = record["sequence"][target_index]
        ground_truth_text = target_segment["question"]
 
        # Identify distractors: every segment other than target_index.
        distractors = [
            seg for i, seg in enumerate(record["sequence"]) if i != target_index
        ]
 
        # "math" if this segment has a bare-string expected_answer, "trivia"
        # if it has an expected_answers list -- purely informational output,
        # not used for grading here.
        if "expected_answer" in target_segment:
            task = "math"
        elif "expected_answers" in target_segment:
            task = "trivia"
        else:
            task = None
 
        cue_type = record.get("cue_type", "emotion")
 
        # Construct the ASR Prompt -- cue is the speaker's emotional tone
        # instead of gender or a background-noise category. Gender is
        # deliberately never mentioned, even though it's also controlled.
        adj = EMOTION_ADJ.get(target_emotion, target_emotion)
        prompt = (
            f"Listen to the audio carefully. It contains exactly TWO distinct problems played sequentially, "
            f"each spoken with a different emotional tone of voice.\n"
            f"Find the segment spoken by the {adj}-sounding speaker. "
            f"Please transcribe EXACTLY what is spoken in that specific segment. "
            f"You MUST wrap your final transcription inside <asr> and </asr> tags. "
            f"Do not solve the problem and do not output any other explanations."
        )
 
        prediction_raw = model.query(audio_path, prompt).strip()
        prediction_clean = extract_asr_result(prediction_raw)
 
        total_tested += 1
 
        # Pack results
        result_dict = {
            "concat_id": concat_id,
            "cue_type": cue_type,
            "task": task,
            "target_index": target_index,
            "target_emotion": target_emotion,
            "target_gender": target_gender,
            "ground_truth_text": ground_truth_text,
            "ground_truth_1": ground_truth_1,
            "ground_truth_2": ground_truth_2,
            "prediction_raw": prediction_raw,
            "prediction_clean": prediction_clean,
            "all_distractors_info": [
                {
                    "emotion": d["emotion"],
                    "gender": d["gender"],
                    "text": d["question"],
                } for d in distractors
            ],
        }
 
        # Save immediately in append mode
        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
 
    print(f"\n=== ASR Generation Complete (emotion cue). Total Tested: {total_tested} ===")
 

if __name__ == "__main__":
    pass