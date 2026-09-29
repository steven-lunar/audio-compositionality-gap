import os
import json
from tqdm import tqdm
import string
from src.evaluators.extractor import *
from whisper.normalizers import EnglishTextNormalizer

whisper_normalizer = EnglishTextNormalizer()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = whisper_normalizer(s)
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def evaluate_dual_trivia(model, dataset_dir, metadata_path, output_path):
    print("Starting Sequential Trivia-QA Evaluation")
    
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
                    
    for record in tqdm(records, desc="Evaluating Dual Trivia Capability"):
        concat_id = record["concat_id"]
        
        if concat_id in processed_ids:
            continue
            
        audio_path = os.path.join(dataset_dir, record["audio_path"])
        
        # Extract ground truth expected answer lists from metadata sequence
        norm_gt_list_1 = record["sequence"][0].get("expected_answers", [])
        norm_gt_list_2 = record["sequence"][1].get("expected_answers", [])
        
        # Apply normalization to all valid answers in the ground truth lists
        # norm_gt_list_1 = [normalizer(str(ans)) for ans in gt_ans_list_1]
        # norm_gt_list_2 = [normalizer(str(ans)) for ans in gt_ans_list_2]
        
        # prompt = (
        #     "Listen to the audio carefully. It contains exactly TWO distinct trivia questions played sequentially.\n"
        #     "Answer both questions in the exact order they appear.\n"
        #     "Output ONLY the final answer for the first question inside <ans_1> and </ans_1> tags.\n"
        #     "Output ONLY the final answer for the second question inside <ans_2> and </ans_2> tags.\n"
        #     "Do not output any reasoning or other text."
        # )

        prompt = (
            "Listen to the audio carefully. It contains exactly TWO distinct trivia questions played sequentially.\n"
            "Answer both questions in the exact order they appear.\n"
            "CRITICAL INSTRUCTION: Provide the absolute shortest possible answer. Output ONLY the core entity, name, or word. "
            "DO NOT use full sentences.\n"
            "Output the concise answer for the first question inside <ans_1> and </ans_1> tags.\n"
            "Output the concise answer for the second question inside <ans_2> and </ans_2> tags.\n"
            "Do not output any reasoning or conversational text."
        )
        
        prediction_raw = model.query(audio_path, prompt)
        extracted_ans_1, extracted_ans_2 = extract_dual_trivia_answers(prediction_raw)
        
        # Normalize the extracted answers
        norm_extracted_1 = normalize_answer(str(extracted_ans_1))
        norm_extracted_2 = normalize_answer(str(extracted_ans_2))
        
        # Check if the normalized prediction is in the normalized ground truth list
        is_correct_1 = norm_extracted_1 in norm_gt_list_1 if norm_extracted_1 else False
        is_correct_2 = norm_extracted_2 in norm_gt_list_2 if norm_extracted_2 else False
        
        correct_count = sum([is_correct_1, is_correct_2])
        is_exact_match = (correct_count == 2)
        
        # Record both raw and normalized values for thorough debugging
        result_dict = {
            "concat_id": concat_id,
            # "ground_truth_raw": [gt_ans_list_1, gt_ans_list_2],
            "ground_truth_norm": [norm_gt_list_1, norm_gt_list_2],
            "prediction_raw": prediction_raw,
            "extracted_answers_raw": [extracted_ans_1, extracted_ans_2],
            "extracted_answers_norm": [norm_extracted_1, norm_extracted_2],
            "is_correct_1": is_correct_1,
            "is_correct_2": is_correct_2,
            "correct_count": correct_count,
            "is_exact_match": is_exact_match
        }
        
        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    pass