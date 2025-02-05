import argparse
from transformers import AutoTokenizer

def extract_all_tokens(tokenizer):
    # Get the vocabulary size
    vocab_size = tokenizer.vocab_size

    all_tokens = []
    all_token_ids = []

    for token_id in range(vocab_size):
        # Decode the token ID to get the token string
        token = tokenizer.decode([token_id])
        all_tokens.append(token)
        all_token_ids.append(token_id)

    return all_tokens, all_token_ids

def write_tokens_to_file(tokens, token_ids, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for token, token_id in zip(tokens, token_ids):
            f.write(f"{token}: {token_id}\n")
    print(f"Tokens and IDs written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Tokenizer_To_File')
    parser.add_argument('--model', type=str, default="google/gemma-7b", help='Tokenizer model to use')
    parser.add_argument('-o', '--output', type=str, default='output.txt', help='Path to the output file for tokens.')

    args = parser.parse_args()

    print(f"Obtaining tokens from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Extract tokens
    all_tokens, all_token_ids = extract_all_tokens(tokenizer)

    # Write to file
    write_tokens_to_file(all_tokens, all_token_ids, args.output)
    print(f"Tokens written to {args.output}")

if __name__ == "__main__":
    main()
