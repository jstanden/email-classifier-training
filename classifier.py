from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import json
from inference_hook import customize_email_for_inference

class EmailClassifier:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        
    def prepare_data(self, emails, labels, train_size=0.7, eval_size=0.15, test_size=0.15, random_state=42):
        """
        Prepare your labeled email data with proper train/eval/test split
        emails: list of email texts
        labels: list of folder names/categories (will be converted to integers)
        train_size: proportion for training (default 0.7)
        eval_size: proportion for evaluation (default 0.15)
        test_size: proportion for testing (default 0.15)
        """
        assert abs(train_size + eval_size + test_size - 1.0) < 1e-6, "Split proportions must sum to 1.0"
        
        # Create label mapping
        unique_labels = list(set(labels))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        # Convert labels to integers
        label_ids = [label2id[label] for label in labels]
        
        # Check if stratification is possible (need at least 2 samples per class)
        from collections import Counter
        label_counts = Counter(label_ids)
        min_samples_per_class = min(label_counts.values())
        
        print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        print(f"Minimum samples per class: {min_samples_per_class}")
        
        # Use stratification only if we have enough samples per class
        use_stratify = min_samples_per_class >= 2
        
        if not use_stratify:
            print("Warning: Some classes have fewer than 2 samples. Using non-stratified split.")
        
        # First split: separate test set
        try:
            train_eval_emails, test_emails, train_eval_labels, test_labels = train_test_split(
                emails, label_ids, test_size=test_size, random_state=random_state, 
                stratify=label_ids if use_stratify else None
            )
        except ValueError as e:
            print(f"Stratification failed: {e}")
            print("Falling back to non-stratified split.")
            train_eval_emails, test_emails, train_eval_labels, test_labels = train_test_split(
                emails, label_ids, test_size=test_size, random_state=random_state, stratify=None
            )
        
        # Second split: separate train and eval from remaining data
        eval_ratio = eval_size / (train_size + eval_size)
        try:
            train_emails, eval_emails, train_labels, eval_labels = train_test_split(
                train_eval_emails, train_eval_labels, test_size=eval_ratio, 
                random_state=random_state, stratify=train_eval_labels if use_stratify else None
            )
        except ValueError as e:
            print(f"Stratification failed: {e}")
            print("Falling back to non-stratified split.")
            train_emails, eval_emails, train_labels, eval_labels = train_test_split(
                train_eval_emails, train_eval_labels, test_size=eval_ratio, 
                random_state=random_state, stratify=None
            )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            "text": train_emails,
            "labels": train_labels
        })
        
        eval_dataset = Dataset.from_dict({
            "text": eval_emails,
            "labels": eval_labels
        })
        
        test_dataset = Dataset.from_dict({
            "text": test_emails,
            "labels": test_labels
        })
        
        # Tokenize the data with proper padding
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",  # Pad to max_length
                max_length=512,
                return_tensors=None  # Return lists, not tensors
            )
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
        tokenized_test = test_dataset.map(tokenize_function, batched=True)
        
        return tokenized_train, tokenized_eval, tokenized_test, label2id, id2label
    
    def train(self, train_dataset, eval_dataset, label2id, id2label, output_dir="./email_classifier_final"):
        """Train the model on your labeled data"""
        # Load model for classification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )
        
        # Import data collator and callbacks for proper batching and early stopping
        from transformers import DataCollatorWithPadding, EarlyStoppingCallback
        
        # Create data collator for consistent padding
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding='max_length',  # Changed from padding=True to padding='max_length'
            max_length=512,
            return_tensors="pt"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=2e-5,
            logging_steps=1,
        )
        
        # Create trainer with data collator and early stopping callback
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],  # Stop if no improvement for 1 epoch
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model(output_dir)
        
        # Print final evaluation results
        eval_results = trainer.evaluate()
        print(f"\nFinal evaluation loss: {eval_results['eval_loss']:.4f}")
        
    def classify_emails(self, emails, model_path="./models/email_classifier_final"):
        """Classify new emails into folders"""
        # Load the trained model
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=self.tokenizer,
            top_k=None,  # Get all scores
            truncation=True,  # Enable truncation
            max_length=512,  # Set max length
            padding=True
        )
        
        results = []
        for email in emails:
            # --- HOOK: customize email content before truncation ---
            email = customize_email_for_inference(email)
            # Truncate email if it's too long
            if len(email) > 400:  # Rough character limit
                email = email[:400] + "..."
            prediction = classifier(email)
            # Get the highest confidence prediction
            best_prediction = max(prediction[0], key=lambda x: x['score'])
            results.append({
                'email': email,
                'folder': best_prediction['label'],
                'confidence': best_prediction['score'],
                'all_scores': prediction[0]  # Include all scores for debugging
            })
        
        return results

    def load_test_data(self, emails, labels, test_size=0.15, random_state=42):
        """
        Load test data using the same split as prepare_data
        This ensures consistency between training and evaluation
        """
        from sklearn.model_selection import train_test_split
        
        # Create label mapping (should match the one used in training)
        unique_labels = list(set(labels))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        # Convert labels to integers
        label_ids = [label2id[label] for label in labels]
        
        # Use the same split logic as prepare_data
        train_eval_emails, test_emails, train_eval_labels, test_labels = train_test_split(
            emails, label_ids, test_size=test_size, random_state=random_state, stratify=label_ids
        )
        
        return test_emails, test_labels, label2id, id2label

def load_dataset(jsonl_path, sample_size=None, random_state=42, use_hierarchical_labels=True):
    """
    Load dataset from JSONL file
    sample_size: if provided, randomly sample this many examples
    use_hierarchical_labels: if True, extract top-level category from hierarchical labels
    """
    emails = []
    labels = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise Exception(f"Invalid JSON: {line}")
            if 'subject' in obj and 'label' in obj:
                subject = obj['subject']
                body = obj.get('body', '')
                # Concatenate subject and body, but limit length
                combined = f"{subject} {body}".strip()
                if len(combined) > 400:
                    combined = combined[:400] + "..."
                emails.append(combined)
                label = obj['label']
            else:
                print(f"Warning: Skipping line with unknown format: {obj}")
                continue
            # Handle hierarchical labels
            if use_hierarchical_labels and '/' in label:
                top_level_label = label.split('/')[0].strip()
                labels.append(top_level_label)
            else:
                labels.append(label)
    if sample_size and sample_size < len(emails):
        import random
        random.seed(random_state)
        indices = random.sample(range(len(emails)), sample_size)
        emails = [emails[i] for i in indices]
        labels = [labels[i] for i in indices]
    return emails, labels