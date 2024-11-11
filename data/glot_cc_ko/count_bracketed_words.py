import re
from collections import Counter
import argparse

def find_bracketed_words(filename="output.txt"):
    # Regular expression to match words within [[[[[ ]]]]] brackets
    pattern = r"\[\[\[\[\[\s*(\w+)\s*\]\]\]\]\]"

    # Read the file content
    with open(filename, 'r') as file:
        content = file.read()

    # Find all matches in the file
    words = re.findall(pattern, content)

    # Count the frequency of each word
    word_count = Counter(words)

    # Print each word and its frequency
    for word, count in word_count.items():
        print(f"{word}: {count}")

if __name__ == "__main__":
    # Setting up argument parser
    parser = argparse.ArgumentParser(description="Count frequency of words in [[[[[ ]]]]] brackets.")
    parser.add_argument("--file", type=str, default="output.txt", help="The file to parse (default: output.txt)")
    args = parser.parse_args()

    # Run the function
    find_bracketed_words(args.file)

