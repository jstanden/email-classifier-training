#!/usr/bin/env python3
"""
Training script for the email classifier.
Usage: python train.py [--data_path data/emails.jsonl] [--model_name distilbert-base-uncased]
"""

import argparse
import os
import json
from classifier import EmailClassifier, load_dataset
from inference_hook import customize_email_for_inference
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Train email classifier')
    parser.add_argument('--data_path', default='data/emails.jsonl', 
                       help='Path to JSONL training data file')
    parser.add_argument('--model_name', default='distilbert-base-uncased',
                       help='Base model to fine-tune')
    parser.add_argument('--output_dir', default='./models/email_classifier_final',
                       help='Directory to save the trained model')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of examples to sample from dataset (for quick testing)')
    parser.add_argument('--no_hierarchical', action='store_true',
                       help='Disable hierarchical label processing (use full labels)')
    
    args = parser.parse_args()
    
    # Load training data
    print(f"Loading dataset from {args.data_path}")
    print(f"Hierarchical label processing: {'Disabled' if args.no_hierarchical else 'Enabled'}")
    
    training_emails, training_labels = load_dataset(
        args.data_path, 
        sample_size=args.sample_size,
        use_hierarchical_labels=not args.no_hierarchical
    )
    
    # Apply inference hook to training data for consistency
    print("Applying inference hook to training data...")
    processed_emails = []
    for email in training_emails:
        processed_email = customize_email_for_inference(email)
        processed_emails.append(processed_email)
    
    print(f"Training with {len(processed_emails)} examples")
    print(f"Label distribution: {dict(zip(*np.unique(training_labels, return_counts=True)))}")
    
    # Initialize classifier
    classifier = EmailClassifier(model_name=args.model_name)
    
    # Prepare data with train/eval/test split
    train_dataset, eval_dataset, test_dataset, label2id, id2label = classifier.prepare_data(
        processed_emails, training_labels
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(eval_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Number of classes: {len(label2id)}")
    print(f"Label mapping: {label2id}")
    
    # Save test data for later evaluation (JSONL only)
    os.makedirs(args.output_dir, exist_ok=True)
    test_jsonl_file = os.path.join(args.output_dir, 'test_data.jsonl')
    with open(test_jsonl_file, 'w') as f:
        for i in range(len(test_dataset)):
            obj = {
                'text': test_dataset[i]['text'],
                'label': id2label[test_dataset[i]['labels']]
            }
            f.write(json.dumps(obj) + '\n')
    print(f"Test data saved as JSONL to {test_jsonl_file}")
    
    # Save label mappings separately
    metadata = {
        'label2id': label2id,
        'id2label': id2label,
        'hierarchical_processing': not args.no_hierarchical
    }
    metadata_file = os.path.join(args.output_dir, 'test_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Test metadata saved to {metadata_file}")
    
    # Train the model
    print("\nStarting training...")
    classifier.train(train_dataset, eval_dataset, label2id, id2label, output_dir=args.output_dir)
    
    print(f"\nTraining completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main() 