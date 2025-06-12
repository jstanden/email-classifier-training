#!/usr/bin/env python3
"""
Testing/inference script for the email classifier.
Usage: 
  python test.py "Your email text here"
  python test.py --input_file emails_to_classify.txt
  python test.py --interactive
"""

import argparse
import os
import sys
from classifier import EmailClassifier

def main():
    parser = argparse.ArgumentParser(description='Test email classifier on new emails')
    parser.add_argument('--model_path', default='./models/email_classifier_final',
                       help='Path to trained model')
    parser.add_argument('--input_file', 
                       help='File containing emails to classify (one per line)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('emails', nargs='*', 
                       help='Email texts to classify')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first using train.py")
        return
    
    # Initialize classifier
    classifier = EmailClassifier()
    
    # Get emails to classify
    emails_to_classify = []
    
    if args.interactive:
        print("Interactive mode - enter emails to classify (press Ctrl+D when done):")
        print("Enter email text:")
        for line in sys.stdin:
            email = line.strip()
            if email:
                emails_to_classify.append(email)
                print("Enter email text (or Ctrl+D to finish):")
    
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}")
            return
        
        with open(args.input_file, 'r') as f:
            emails_to_classify = [line.strip() for line in f if line.strip()]
    
    elif args.emails:
        emails_to_classify = args.emails
    
    else:
        print("No emails provided. Use --interactive, --input_file, or provide emails as arguments.")
        return
    
    if not emails_to_classify:
        print("No emails to classify.")
        return
    
    # Classify emails
    print(f"\nClassifying {len(emails_to_classify)} emails...")
    results = classifier.classify_emails(emails_to_classify, args.model_path)
    
    # Display results
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. Email: {result['email'][:60]}...")
        print(f"   Predicted Folder: {result['folder']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print("   All scores:")
        for score in result['all_scores']:
            print(f"     {score['label']}: {score['score']:.3f}")
        print("-" * 50)

if __name__ == "__main__":
    main() 