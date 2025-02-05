import argparse
import json
import re

def categorize_tokens(input_file, output_file):
    """
    Categorizes tokens based on their language and writes to a JSON file.

    Args:
      input_file: Path to the input file containing tokens and IDs.
      output_file: Path to the output JSON file.
    """

    categories = {
        "chinese": [],
        "english": [],
        "korean": [],
        "japanese": [],
        "misc": []
    }

    cjk_ranges = [
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F), # CJK Unified Ideographs Extension C
        (0x2B740, 0x2B81F), # CJK Unified Ideographs Extension D
        (0x2B820, 0x2CEAF), # CJK Unified Ideographs Extension E
        (0x2CEB0, 0x2EBEF), # CJK Unified Ideographs Extension F
        (0x30000, 0x3134F),  # CJK Unified Ideographs Extension G
        (0x31350, 0x323AF)   # CJK Unified Ideographs Extension H
        ]

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                parts = line.strip().split(":")
                if len(parts) < 2:  # Handle cases with no colon or missing ID
                  print(f"Skipping invalid line (missing ID or colon): {line.strip()}")
                  continue
                token = ":".join(parts[:-1]).strip()
                token_id = parts[-1].strip() #getting the id here

                # Basic categorization based on character sets
                if any(0x3040 <= ord(c) <= 0x30FF for c in token):  # Japanese Hiragana/Katakana
                    categories["japanese"].append({"token": token, "id": token_id})
                elif any(0xAC00 <= ord(c) <= 0xD7A3 for c in token):  # Korean characters
                    categories["korean"].append({"token": token, "id": token_id})
                elif any(0x4E00 <= ord(c) <= 0x9FFF for c in token):  # Chinese characters
                    categories["chinese"].append({"token": token, "id": token_id})
                elif re.search(r'[a-zA-Z]', token):  # Check for English letters
                    categories["english"].append({"token": token, "id": token_id})
                else:
                    is_misc = True
                    for start, end in cjk_ranges:
                        if any(start <= ord(c) <= end for c in token):  # Chinese characters
                            categories["chinese"].append({"token": token, "id": token_id})
                            is_misc = False
                            break

                    if is_misc:
                        categories["misc"].append({"token": token, "id": token_id}) # Catchall

            except ValueError:
                print(f"Skipping invalid line (ValueError): {line.strip()}")
            except IndexError as e:
                 print(f"Skipping invalid line (IndexError {e}): {line.strip()}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(categories, f, indent=4, ensure_ascii=False)

    print(f"Categorized tokens written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Token Categorizer')
    parser.add_argument('-i', '--input', type=str, default='output.txt', help='Path to the input file with tokens and IDs.')
    parser.add_argument('-o', '--output', type=str, default='categorized_tokens.json', help='Path to the output JSON file.')
    args = parser.parse_args()
    categorize_tokens(args.input, args.output)

if __name__ == "__main__":
    main()
