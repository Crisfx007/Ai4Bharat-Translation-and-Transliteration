import warnings
import cld3
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from ai4bharat.transliteration import XlitEngine
import torch
import math


warnings.filterwarnings("ignore", category=FutureWarning)


xlit_engine = XlitEngine("hi", beam_width=10)


lang_codes = {
    "en": "eng_Latn", "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu",
    "mr": "mar_Deva", "pa": "pan_Guru", "bn": "ben_Beng", "gu": "guj_Gujr",
    "ml": "mal_Mlym", "kn": "kan_Knda", "as": "asm_Beng"
}


cld3_to_lang = {
    "eng": "en", "hin": "hi", "tam": "ta", "tel": "te", "mar": "mr",
    "pan": "pa", "ben": "bn", "guj": "gu", "mal": "ml", "kan": "kn", "asm": "as",
    "hi-Latn": "hi"  
}


script_ranges = {
    "hi": range(0x0900, 0x097F),  # Devanagari (Hindi, Marathi)
    "ta": range(0x0B80, 0x0BFF),  # Tamil
    "te": range(0x0C00, 0x0C7F),  # Telugu
    "mr": range(0x0900, 0x097F),  # Devanagari (Marathi)
    "pa": range(0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    "bn": range(0x0980, 0x09FF),  # Bengali (also Assamese)
    "gu": range(0x0A80, 0x0AFF),  # Gujarati
    "ml": range(0x0D00, 0x0D7F),  # Malayalam
    "kn": range(0x0C80, 0x0CFF),  # Kannada
    "as": range(0x0980, 0x09FF)   # Bengali script (Assamese)
}

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variable for translator
translator = None

# Function to initialize translator lazily with GPU support
def get_translator():
    global translator
    if translator is None:
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0 if device.type == "cuda" else -1)
    return translator

# Function to detect script (improved Hinglish detection)
def detect_script(text):
    if not text.strip():  # Handle empty strings
        return "en"
    for char in text:
        for lang, char_range in script_ranges.items():
            if ord(char) in char_range:
                return lang
    all_ascii = all(ord(c) < 128 for c in text)
    if all_ascii:
        result = cld3.get_language(text)
        if result and result.language == "hi-Latn":
            return "hi"
        elif result and result.probability < 0.7:  # Low confidence ASCII text
            print("Low confidence ASCII text, assuming Romanized Hindi.")
            return "hi"  # Assume Hinglish for uncertain ASCII cases
    return "en"

# Function to detect language using CLD3 with fallback
def detect_language(text):
    if not text.strip():  # Handle empty strings
        return "en", 1.0
    result = cld3.get_language(text)
    if result and result.probability > 0.5:
        detected_lang = cld3_to_lang.get(result.language, result.language)
        if detected_lang not in lang_codes or (detected_lang == "en" and result.probability < 0.9):
            script_lang = detect_script(text)
            if script_lang != detected_lang:
                print(f"CLD3 detected '{detected_lang}', but script suggests '{script_lang}', overriding.")
                return script_lang, max(result.probability, 0.7)
        return detected_lang, result.probability
    script_lang = detect_script(text)
    return script_lang, 0.7  # Default to 0.7 for fallback cases

# Function to process text with GPU acceleration
def process_text(text):
    try:
        detected_lang, confidence = detect_language(text)
        print(f"Detected: {{'label': '{detected_lang}', 'score': {confidence}}}")

        if confidence >= 0.5:  # Adjusted to include 0.5
            if detected_lang == "en":
                print(f"English detected, returning original text: {text}")
                return text
            elif detected_lang == "hi":
                if any(ord(c) in range(0x0900, 0x097F) for c in text):
                    print(f"Devanagari Hindi detected, translating directly.")
                    translator = get_translator()
                    translation = translator(text, src_lang=lang_codes["hi"], tgt_lang="eng_Latn")
                    print(f"Translated to English: {translation[0]['translation_text']}")
                    return translation[0]['translation_text']
                else:
                    print(f"Romanized Hindi detected, transliterating to Devanagari.")
                    translit_result = xlit_engine.translit_sentence(text)
                    devanagari_text = translit_result['hi']
                    print(f"Transliterated to Devanagari: {devanagari_text}")
                    translator = get_translator()
                    translation = translator(devanagari_text, src_lang=lang_codes["hi"], tgt_lang="eng_Latn")
                    print(f"Translated to English: {translation[0]['translation_text']}")
                    return translation[0]['translation_text']
            elif detected_lang in lang_codes:
                print(f"{detected_lang.capitalize()} detected, translating directly to English.")
                translator = get_translator()
                translation = translator(text, src_lang=lang_codes[detected_lang], tgt_lang="eng_Latn")
                print(f"Translated to English: {translation[0]['translation_text']}")
                return translation[0]['translation_text']
            else:
                print(f"Language '{detected_lang}' not supported in this pipeline.")
                return text
        else:
            print("Confidence too low, returning original text.")
            return text
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return text

# Function to split JSON data into parts and process them
def process_json_file_in_parts(input_file, output_prefix="translated_part_", num_parts=20, start_part=1, end_part=None):
    # Load JSON data from file
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return

    # Calculate the size of each part
    total_tweets = len(json_data)
    part_size = math.ceil(total_tweets / num_parts)
    print(f"Total tweets: {total_tweets}, Part size: {part_size}, Number of parts: {num_parts}")

    # Adjust end_part if not specified or exceeds num_parts
    end_part = min(end_part or num_parts, num_parts)

    # Process each part within the specified range
    for part in range(start_part - 1, end_part):
        start_idx = part * part_size
        end_idx = min((part + 1) * part_size, total_tweets)
        part_data = json_data[start_idx:end_idx]
        
        print(f"\nProcessing part {part + 1} (tweets {start_idx + 1} to {end_idx})")
        
        # Process each tweet and its comments in this part
        for tweet in part_data:
            original_content = tweet.get("content", "")
            if original_content:
                translated_content = process_text(original_content)
                tweet["content"] = translated_content
                print(f"Tweet ID {tweet['tweet_id']} translated: {translated_content}")
            
            if "comments" in tweet:
                for comment in tweet["comments"]:
                    original_comment = comment.get("content", "")
                    if original_comment:
                        translated_comment = process_text(original_comment)
                        comment["content"] = translated_comment
                        print(f"Comment translated: {translated_comment}")

        # Save the translated part to a file
        output_file = f"{output_prefix}{part + 1}.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(part_data, f, ensure_ascii=False, indent=4)
            print(f"Part {part + 1} saved to '{output_file}'")
        except Exception as e:
            print(f"Error saving part {part + 1}: {str(e)}")

# Specify input file and processing options
input_file = "india_training_merged_filtered.json"  # Replace with your actual input file path
output_prefix = "translated_part_"  # Prefix for output files
num_parts = 20  # Total number of parts to split into

# Process the file in parts (e.g., process parts 1 to 5, as specified)
process_json_file_in_parts(input_file, output_prefix, num_parts, start_part=1, end_part=20)

# To resume from a specific part, e.g., part 3 to part 5:
# process_json_file_in_parts(input_file, output_prefix, num_parts, start_part=3, end_part=5)