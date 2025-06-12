#!/usr/bin/env python3
"""
Evaluation script for the email classifier.
Usage: python evaluate.py [--model_path ./email_classifier_final] [--test_data data/test_emails.jsonl]
"""

import argparse
import os
import json
from classifier import EmailClassifier, load_dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import init, Fore, Back, Style

# Initialize colorama for cross-platform colored output
init()

def plot_confusion_matrix(cm, labels, output_path=None):
    """Plot confusion matrix with proper labels"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate email classifier')
    parser.add_argument('--model_path', default='./models/email_classifier_final',
                       help='Path to trained model')
    parser.add_argument('--test_data', 
                       help='Path to test data JSONL file (if not using saved test data)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of examples to sample from test data (for quick evaluation)')
    parser.add_argument('--no_hierarchical', action='store_true',
                       help='Disable hierarchical label processing (use full labels)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first using train.py")
        return
    
    # Load test data
    test_file = os.path.join(args.model_path, 'test_data.json')
    metadata_file = os.path.join(args.model_path, 'test_metadata.json')
    
    if os.path.exists(test_file) and not args.test_data:
        # Use the saved test data from training (recommended)
        print(f"Loading saved test data from {test_file}")
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        test_emails = test_data['emails']
        test_labels = test_data['labels']
        label2id = test_data['label2id']
        id2label = test_data['id2label']
        hierarchical_processing = test_data.get('hierarchical_processing', True)
        
        print(f"Model was trained with hierarchical processing: {'Enabled' if hierarchical_processing else 'Disabled'}")
        
        # Sample if requested
        if args.sample_size and args.sample_size < len(test_emails):
            import random
            random.seed(42)
            indices = random.sample(range(len(test_emails)), args.sample_size)
            test_emails = [test_emails[i] for i in indices]
            test_labels = [test_labels[i] for i in indices]
            
    elif os.path.exists(os.path.join(args.model_path, 'test_data.jsonl')) and not args.test_data:
        # Use the new JSONL format with separate metadata
        test_jsonl_file = os.path.join(args.model_path, 'test_data.jsonl')
        print(f"Loading saved test data from {test_jsonl_file}")
        
        test_emails = []
        test_labels = []
        with open(test_jsonl_file, 'r') as f:
            for line in f:
                obj = json.loads(line.strip())
                test_emails.append(obj['text'])
                test_labels.append(obj['label'])
        
        # Load metadata
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            label2id = metadata['label2id']
            id2label = metadata['id2label']
            hierarchical_processing = metadata.get('hierarchical_processing', True)
            print(f"Model was trained with hierarchical processing: {'Enabled' if hierarchical_processing else 'Disabled'}")
        else:
            label2id = None
            id2label = None
            print("Warning: No metadata file found, label mappings will be inferred from test data")
        
        # Sample if requested
        if args.sample_size and args.sample_size < len(test_emails):
            import random
            random.seed(42)
            indices = random.sample(range(len(test_emails)), args.sample_size)
            test_emails = [test_emails[i] for i in indices]
            test_labels = [test_labels[i] for i in indices]
    elif args.test_data:
        # Use provided test data file
        print(f"Loading test data from {args.test_data}")
        
        # Check if it's a JSON file (like saved test_data.json) or JSONL file
        if args.test_data.endswith('.json'):
            # Load as JSON file (like the saved test_data.json from training)
            with open(args.test_data, 'r') as f:
                test_data = json.load(f)
            
            test_emails = test_data['emails']
            test_labels = test_data['labels']
            label2id = test_data.get('label2id')
            id2label = test_data.get('id2label')
            hierarchical_processing = test_data.get('hierarchical_processing', True)
            
            print(f"Model was trained with hierarchical processing: {'Enabled' if hierarchical_processing else 'Disabled'}")
            
        else:
            # Load as JSONL file (original format)
            print(f"Hierarchical label processing: {'Disabled' if args.no_hierarchical else 'Enabled'}")
            
            test_emails, test_labels = load_dataset(
                args.test_data, 
                sample_size=args.sample_size,
                use_hierarchical_labels=not args.no_hierarchical
            )
            # Note: label2id and id2label will be inferred from test data
            label2id = None
            id2label = None
        
        # Sample if requested
        if args.sample_size and args.sample_size < len(test_emails):
            import random
            random.seed(42)
            indices = random.sample(range(len(test_emails)), args.sample_size)
            test_emails = [test_emails[i] for i in indices]
            test_labels = [test_labels[i] for i in indices]
    else:
        print("Error: No test data found. Either:")
        print("1. Train a model first to generate test_data.json, or")
        print("2. Provide --test_data argument")
        return
    
    print(f"Evaluating on {len(test_emails)} examples")
    print(f"Label distribution: {dict(zip(*np.unique(test_labels, return_counts=True)))}")
    
    # Initialize classifier and load trained model
    classifier = EmailClassifier()
    
    # Classify test emails
    print("\nRunning predictions...")
    results = classifier.classify_emails(test_emails, args.model_path)
    
    # Extract predictions and true labels
    predicted_labels = [result['folder'] for result in results]
    true_labels = test_labels
    
    # Calculate metrics
    print("\n" + "="*60)
    print(f"{Fore.CYAN}EVALUATION RESULTS{Style.RESET_ALL}")
    print("="*60)
    
    # Classification report
    print(f"\n{Fore.YELLOW}Classification Report:{Style.RESET_ALL}")
    print(classification_report(true_labels, predicted_labels, zero_division=0))
    
    # Confusion matrix
    print(f"\n{Fore.YELLOW}Confusion Matrix:{Style.RESET_ALL}")
    cm = confusion_matrix(true_labels, predicted_labels)
    print(cm)
    
    # Plot confusion matrix
    if id2label:
        # Get unique labels in the order they appear in the confusion matrix
        unique_labels = sorted(list(set(true_labels + predicted_labels)))
        plot_confusion_matrix(cm, unique_labels, 
                             output_path=os.path.join(args.model_path, 'confusion_matrix.png'))
    else:
        print("Skipping confusion matrix plot - no label mappings available")
    
    # Overall accuracy
    accuracy = np.mean(np.array(predicted_labels) == np.array(true_labels))
    accuracy_color = Fore.GREEN if accuracy >= 0.8 else Fore.YELLOW if accuracy >= 0.6 else Fore.RED
    print(f"\n{Fore.YELLOW}Overall Accuracy:{Style.RESET_ALL} {accuracy_color}{accuracy:.4f}{Style.RESET_ALL}")
    
    # Per-email results (show random 20 for brevity)
    print(f"\n{Fore.YELLOW}Detailed Results (random 20):{Style.RESET_ALL}")
    print("-" * 80)
    
    # Select random 20 examples
    import random
    random.seed(42)  # For reproducibility
    if len(test_emails) > 20:
        indices = random.sample(range(len(test_emails)), 20)
        sample_emails = [test_emails[i] for i in indices]
        sample_labels = [true_labels[i] for i in indices]
        sample_results = [results[i] for i in indices]
    else:
        sample_emails = test_emails
        sample_labels = true_labels
        sample_results = results
    
    for i, (email, true_label, result) in enumerate(zip(sample_emails, sample_labels, sample_results)):
        predicted_label = result['folder']
        confidence = result['confidence']
        correct = Fore.GREEN + "✓" + Style.RESET_ALL if predicted_label == true_label else Fore.RED + "✗" + Style.RESET_ALL
        
        print(f"{i+1:2d}. {correct} {email[:50]}...")
        print(f"     True: {true_label:15s} | Predicted: {predicted_label:15s} | Confidence: {confidence:.3f}")
        print()
    
    if len(test_emails) > 20:
        print(f"... and {len(test_emails) - 20} more examples")

if __name__ == "__main__":
    main() 