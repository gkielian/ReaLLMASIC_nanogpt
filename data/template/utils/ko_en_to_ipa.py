import subprocess
from konlpy.tag import Okt
import inflect
import argparse
import re

# English number conversion engine
inflect_engine = inflect.engine()

# Korean number dictionary
korean_numbers = {
    0: "영",
    1: "일",
    2: "이",
    3: "삼",
    4: "사",
    5: "오",
    6: "육",
    7: "칠",
    8: "팔",
    9: "구",
    10: "십"
}

def number_to_korean(num):
    """Convert numbers to Korean words."""
    if num <= 10:
        return korean_numbers[num]
    else:
        result = []
        if num >= 1000:
            result.append(korean_numbers[num // 1000] + "천")
            num %= 1000
        if num >= 100:
            result.append(korean_numbers[num // 100] + "백")
            num %= 100
        if num >= 10:
            result.append(korean_numbers[num // 10] + "십")
            num %= 10
        if num > 0:
            result.append(korean_numbers[num])
        return "".join(result)

def transcribe_english(sentence):
    """Transcribe an English sentence to phonemes using espeak-ng."""
    try:
        result = subprocess.run(
            ["espeak-ng", "-q", "--ipa", sentence],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error in transcribing English: {str(e)}"

def transcribe_korean(sentence):
    """Transcribe a Korean sentence into its phonemes using KoNLPy (Okt)."""
    okt = Okt()
    tokens = okt.morphs(sentence)
    tokenized_sentence = ' '.join(tokens)
    
    try:
        result = subprocess.run(
            ["espeak-ng", "-q", "-v", "ko", "--ipa", tokenized_sentence],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error in transcribing Korean: {str(e)}"

def handle_mixed_language(word):
    """Handle a word with potential Korean, English, or number content."""
    if word.isdigit():  # Detect numbers
        number_in_korean = number_to_korean(int(word))
        return transcribe_korean(number_in_korean)
    elif any('가' <= char <= '힣' for char in word):  # Detect Korean
        return transcribe_korean(word)
    else:  # English word
        return transcribe_english(word)

def transcribe_multilingual(sentences, output_file):
    """Transcribe multilingual sentences (English and Korean, with numbers) and save to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            result = []
            # Split sentence but keep punctuation (preserve spaces, commas, etc.)
            words = re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)
            for word in words:
                if re.match(r'\w+', word):  # Only process words (skip punctuation)
                    result.append(handle_mixed_language(word))
                else:
                    result.append(word)  # Preserve punctuation as is
            transcription_result = " ".join(result)
            f.write(transcription_result + "\n")
            print(transcription_result)  # Print to console for reference

def main():
    parser = argparse.ArgumentParser(description='Transcribe multilingual sentences into IPA phonemes.')
    parser.add_argument('input_file', type=str, help='Path to the input file containing sentences.')
    parser.add_argument('output_file', type=str, help='Path to the output file for IPA transcription.')

    args = parser.parse_args()

    # Read input sentences
    with open(args.input_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    # Transcribe and save to the output file
    transcribe_multilingual(sentences, args.output_file)

if __name__ == '__main__':
    main()

