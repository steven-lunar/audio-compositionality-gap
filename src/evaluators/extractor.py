import re

def extract_dual_environment_events(prediction_text):
    """
    Extracts environmental events from <event1>/<event_1> and <event2>/<event_2> tags.
    Robustly handles missing closing tags and optional underscores.
    Returns a list of extracted strings (lowercased) or None if a tag is missing to maintain index alignment.
    """
    results = []
    
    if not isinstance(prediction_text, str):
        return [None, None]
        
    prediction_text = prediction_text.strip()
    
    # Extract Event 1
    # event_?1 supports both <event1> and <event_1>
    # The lookahead (?=(?:</event_?1>|<event_?2>|$)) safely truncates if the closing tag is omitted
    match_1 = re.search(r'<event_?1>(.*?)(?=(?:</event_?1>|<event_?2>|$))', prediction_text, re.IGNORECASE | re.DOTALL)
    if match_1:
        results.append(match_1.group(1).strip().lower())
    else:
        results.append(None)
        
    # Extract Event 2
    match_2 = re.search(r'<event_?2>(.*?)(?=(?:</event_?2>|$))', prediction_text, re.IGNORECASE | re.DOTALL)
    if match_2:
        results.append(match_2.group(1).strip().lower())
    else:
        results.append(None)
        
    return results

def extract_asr_result(prediction_text):
    """
    Extract the transcription from the model's verbose output using XML tags.
    Falls back to the raw text if tags are missing.
    """
    prediction_text = prediction_text.strip()
    match = re.search(r'<asr>(.*?)</asr>', prediction_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return prediction_text

def extract_dual_transcriptions(prediction_text):
    """
    Extracts the transcriptions from <asr1>/<asr_1> and <asr2>/<asr_2> tags.
    Robustly handles missing closing tags and optional underscores in the tag names.
    Returns empty strings if tags are missing or empty.
    """
    asr_1, asr_2 = "", ""
    
    if not isinstance(prediction_text, str):
        return asr_1, asr_2
        
    prediction_text = prediction_text.strip()
    
    # Extract Transcription 1
    # asr_?1 supports both <asr1> and <asr_1>
    # (?=(?:</asr_?1>|<asr_?2>|$)) acts as a lookahead to stop at the next tag or end of string
    match_1 = re.search(r'<asr_?1>(.*?)(?=(?:</asr_?1>|<asr_?2>|$))', prediction_text, re.IGNORECASE | re.DOTALL)
    if match_1:
        asr_1 = match_1.group(1).strip()
        
    # Extract Transcription 2
    match_2 = re.search(r'<asr_?2>(.*?)(?=(?:</asr_?2>|$))', prediction_text, re.IGNORECASE | re.DOTALL)
    if match_2:
        asr_2 = match_2.group(1).strip()
        
    return asr_1, asr_2

def extract_math_answer(prediction_text):
    """
    Extract the final numerical answer from a verbose model prediction.
    Implements fallback strategies and strict decimal boundary checks.
    """
    if not prediction_text:
        return None
        
    prediction_text = prediction_text.strip()
    
    # Priority 1: Strict XML tag extraction (using the safe decimal regex)
    xml_match = re.search(r'<ans>(.*?)</ans>', prediction_text, flags=re.IGNORECASE | re.DOTALL)
    if xml_match:
        content = xml_match.group(1)
        # Use the corrected regex: -?\d+(?:\.\d+)?
        numbers = re.findall(r'-?\d+(?:\.\d+)?', content)
        if numbers:
            return numbers[-1]
        return content.strip()
        
    # Priority 2: Conversational fallback (e.g., "The answer is 32.7.")
    fallback_matches = re.findall(r'(?:answer is|is)[:\s]*(-?\d+(?:\.\d+)?)', prediction_text, flags=re.IGNORECASE)
    if fallback_matches:
        return fallback_matches[-1]
        
    # Priority 3: Absolute fallback, grab the absolute last valid number in the text
    all_numbers = re.findall(r'-?\d+(?:\.\d+)?', prediction_text)
    if all_numbers:
        return all_numbers[-1]
        
    return "PARSING_ERROR"

def extract_dual_math_answers(prediction_text):
    """
    Extracts the final numerical answers from <ans1>/<ans_1> and <ans2>/<ans_2> tags.
    Robustly handles missing closing tags and grabs the last valid number even if the model hallucinates equations.
    """
    ans_1, ans_2 = None, None
    
    if not isinstance(prediction_text, str):
        return ans_1, ans_2
        
    prediction_text = prediction_text.strip()
    
    # Extract Answer 1
    # ans_?1 supports both <ans1> and <ans_1>
    # (?=(?:</ans_?1>|<ans_?2>|$)) acts as a lookahead to stop at the next tag or end of string
    match_1 = re.search(r'<ans_?1>(.*?)(?=(?:</ans_?1>|<ans_?2>|$))', prediction_text, re.IGNORECASE | re.DOTALL)
    if match_1:
        # Extract all numeric values and select the last one
        numbers_1 = re.findall(r'-?\d+\.?\d*', match_1.group(1))
        if numbers_1:
            ans_1 = numbers_1[-1]
            
    # Extract Answer 2
    match_2 = re.search(r'<ans_?2>(.*?)(?=(?:</ans_?2>|$))', prediction_text, re.IGNORECASE | re.DOTALL)
    if match_2:
        # Extract all numeric values and select the last one
        numbers_2 = re.findall(r'-?\d+\.?\d*', match_2.group(1))
        if numbers_2:
            ans_2 = numbers_2[-1]
            
    return ans_1, ans_2

def extract_trivia_answer(prediction_raw):
    """
    Parses the raw prediction string to extract text within <ans> tags.
    """
    if not isinstance(prediction_raw, str):
        return ""
        
    match = re.search(r"<ans>(.*?)</ans>", prediction_raw, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def extract_dual_trivia_answers(prediction_raw):
    """
    Robustly parses text within <ans1> and <ans2>, handling missing or correct closing tags.
    """
    ans_1 = ""
    ans_2 = ""
    
    if not isinstance(prediction_raw, str):
        return ans_1, ans_2
        
    match_1 = re.search(r"<ans_1>(.*?)(?=(?:</ans_1>|<ans_2>|$))", prediction_raw, flags=re.DOTALL | re.IGNORECASE)
    if match_1:
        ans_1 = match_1.group(1).strip()
        
    match_2 = re.search(r"<ans_2>(.*?)(?=(?:</ans_2>|$))", prediction_raw, flags=re.DOTALL | re.IGNORECASE)
    if match_2:
        ans_2 = match_2.group(1).strip()
        
    return ans_1, ans_2

if __name__ == "__main__":
    pass