import re
from word2number import w2n

def gsm8k_extract_pattern(text):
    pattern = r'####\s+(\d+)'
    matches = re.findall(pattern, text)
    return matches

# def gsm8k_extract_pattern_from_array(text_array):
#     pattern = r'####\s+(\d+)'
#     results = []
#     for string in text_array:
#         matches = re.findall(pattern, string)
#         results.append(matches)
#     return results

def gsm8k_extract_pattern_from_array(text_array):
    # pattern = '### The answer is:'
    pattern = '### A:'
    results = []
    for string in text_array:
        split_strings = string.split(pattern)
        # print('split_strings: ', split_strings)
        matches = '####' if len(split_strings) == 1 else split_strings[1] 
        results.append(matches)
    return results

def mmlu_extract_integers_after_hashes(text_array):
    pattern = r'####.*?(\d+)'
    results = []
    for string in text_array:
        matches = re.findall(pattern, string)
        results.append(matches[0] if len(matches) > 0 else '4')
    return results

def mmlu_extract_letters_after_hashes(text_array):
    pattern = r'####.*?\b(A|B|C|D|a|b|c|d)\b'
    results = []
    for string in text_array:
        matches = re.findall(pattern, string)
        if len(matches) > 0:
            results.append([m.upper() for m in matches])
        else:
            results.append(['E'])
    return results

def mmlu_extract_text_answer(text):
    pattern = '####'
    split_strings = text.split(pattern, 1)
    matches = '#### ' if len(split_strings) == 1 else split_strings[1] 
    return matches

def mmlu_extract_between(text):
    pattern = re.escape('A: ') + r'(.*?)' + re.escape('B')
    match = re.search(pattern, text)
    a_text = match.group(1) if match else ""
    
    pattern = re.escape('B: ') + r'(.*?)' + re.escape('C')
    match = re.search(pattern, text)
    b_text = match.group(1) if match else ""
    
    pattern = re.escape('C: ') + r'(.*?)' + re.escape('D')
    match = re.search(pattern, text)
    c_text = match.group(1) if match else ""
    
    pattern = re.escape('D: ') + r'(.*?)' + re.escape('. The right')
    match = re.search(pattern, text)
    d_text = match.group(1) if match else ""
    
    return {'A': a_text, 'B': b_text, 'C': c_text, 'D': d_text}

def convert_words_to_numbers(text):
    words = text.split()
    converted_words = []
    
    for word in words:
        try:
            converted_words.append(str(w2n.word_to_num(word)))
        except ValueError:
            converted_words.append(word)
    
    return ' '.join(converted_words)

def extract_answer_from_array(text_array):
    pattern = '\n'
    results = []
    for string in text_array:
        matches = string.rsplit(pattern, 1)[1]
        processed_numbers_string = convert_words_to_numbers(matches)
        results.append(processed_numbers_string)
    return results

def classification_extract_answer_from_array(text_array):
    pattern = '### Class:'
    results = []
    for string in text_array:
        matches = string.rsplit(pattern, 1)[1]
        results.append(matches)
    return results

def vqav2_extract_answer_from_chat_template(text_array):
    pattern = r'ASSISTANT: (.*?)</s>'
    extracted_strings = []
    
    for text in text_array:
        match = re.search(pattern, text)
        if match:
            extracted_strings.append(match.group(1))
    
    return extracted_strings