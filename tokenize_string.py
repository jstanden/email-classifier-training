#!/usr/bin/env python3
"""
Tokenize arbitrary strings using the trained model's tokenizer.

Usage:
    python tokenize_string.py "This is a test." "Another string."
    # Or with no arguments, enter strings interactively
    python tokenize_string.py
"""
import sys
from transformers import AutoTokenizer

MODEL_DIR = "./models/email_classifier_final"

def print_tokenization(text, tokenizer):
    print(f"Input: {text}")
    encoding = tokenizer(text)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {encoding['input_ids']}")
    print(f"Attention mask: {encoding['attention_mask']}")
    print("-" * 60)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print_tokenization(arg, tokenizer)
    else:
        print("Enter text to tokenize (Ctrl+D to finish):")
        try:
            for line in sys.stdin:
                line = line.strip()
                if line:
                    print_tokenization(line, tokenizer)
        except KeyboardInterrupt:
            print()
            return

if __name__ == "__main__":
    main() 