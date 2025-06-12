#!/usr/bin/env python3
"""
Test the inference_hook.py customization logic on a JSONL dataset.

Usage:
    python test_inference_hook.py data/emails.jsonl
"""
import sys
import json
import difflib
from colorama import init, Fore, Back, Style
from inference_hook import customize_email_for_inference

# Initialize colorama for cross-platform colored output
init()

def print_example(idx, original, processed):
    print(f"{Fore.CYAN}Example {idx+1}:{Style.RESET_ALL}")
    
    if isinstance(original, dict):
        print(f"{Fore.YELLOW}Original subject:{Style.RESET_ALL}", original.get('subject', ''))
        original_body = original.get('body', '')
        processed_body = processed.get('body', '') if isinstance(processed, dict) else str(processed)
        
        print(f"{Fore.YELLOW}Original body:{Style.RESET_ALL}")
        print_diff(original_body, processed_body, "Original", "Processed")
        
    else:
        print(f"{Fore.YELLOW}Original:{Style.RESET_ALL}")
        print_diff(str(original), str(processed), "Original", "Processed")
    
    print("-" * 80)

def print_diff(original, processed, original_label, processed_label):
    """Print a colored diff between original and processed text."""
    # Split into lines for better diff visualization
    original_lines = original.splitlines()
    processed_lines = processed.splitlines()
    
    # Generate diff
    diff = difflib.unified_diff(
        original_lines, 
        processed_lines,
        fromfile=original_label,
        tofile=processed_label,
        lineterm='',
        n=3  # Context lines
    )
    
    # Print diff with colors
    for line in diff:
        if line.startswith('---') or line.startswith('+++'):
            print(f"{Fore.BLUE}{line}{Style.RESET_ALL}")
        elif line.startswith('@@'):
            print(f"{Fore.MAGENTA}{line}{Style.RESET_ALL}")
        elif line.startswith('+'):
            print(f"{Fore.GREEN}+ {line[1:]}{Style.RESET_ALL}")
        elif line.startswith('-'):
            print(f"{Fore.RED}- {line[1:]}{Style.RESET_ALL}")
        else:
            print(f"  {line}")
    
    # If no differences, show a message
    if original == processed:
        print(f"{Fore.YELLOW}No changes detected{Style.RESET_ALL}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_inference_hook.py <input_jsonl>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    print(f"{Fore.GREEN}Testing inference hook on: {input_path}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Showing first 10 examples...{Style.RESET_ALL}\n")
    
    with open(input_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= 10:
                break
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"{Fore.RED}Skipping line {idx+1}: {e}{Style.RESET_ALL}")
                continue
            
            # Only test if 'subject' and 'body' are present
            if not (isinstance(obj, dict) and 'subject' in obj and 'body' in obj):
                continue
            
            processed = customize_email_for_inference({'subject': obj['subject'], 'body': obj['body']})
            print_example(idx, obj, processed)

if __name__ == "__main__":
    main() 