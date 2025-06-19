# Email Classifier with Hugging Face Transformers

A modular email classification system using Hugging Face Transformers. This project allows you to train, evaluate, and test email classifiers that can categorize emails into different folders (Work, Shopping, IT, etc.).

## Project Structure

```
├── classifier.py      # Core EmailClassifier class and utilities
├── train.py           # Training script
├── evaluate.py        # Evaluation script  
├── test.py            # Testing/inference script
├── inference_hook.py  # Customizable email processing hook
├── tokenize_string.py # Script to tokenize arbitrary strings
├── models/            # Directory containing trained models
├── data/
│   └── emails.jsonl   # Training dataset in JSONL format
└── README.md          # This file
```

## Requirements

- Python 3.7+
- transformers
- datasets
- scikit-learn
- torch
- numpy
- colorama (for colored test output)

## Installation

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Dataset Format

The training data should be in JSONL format with three columns:
- `subject`: The email subject line
- `body`: The email body content
- `label`: The category/folder name (supports hierarchical labels)

### Hierarchical Labels

The system supports hierarchical labels using the format `Category/Subcategory`. By default, only the top-level category is used for training.

**Examples:**

**Simple labels:**
```json
{"subject": "Meeting with client tomorrow at 3 PM", "body": "Please prepare the quarterly report for our meeting.", "label": "Work"}
{"subject": "Your Amazon order has been delivered", "body": "Your package has been delivered to your doorstep.", "label": "Shopping"}
```

**Hierarchical labels (top-level category extracted):**
```json
{"subject": "Server maintenance scheduled for tonight", "body": "We will be performing routine maintenance from 2-4 AM.", "label": "Technical/Infrastructure"}
{"subject": "Password reset required", "body": "Your password has expired and needs to be reset.", "label": "Technical/Security"}
{"subject": "Critical security update required", "body": "Please install the latest security patches.", "label": "Technical/Security"}
```

In the above examples, all emails with `Technical/*` labels will be classified as `Technical` during training.

**Note**: 
- The classifier automatically concatenates subject and body fields, and truncates long texts to fit within the model's token limit (512 tokens).
- Use `--no_hierarchical` flag to disable hierarchical processing and use full labels.

## Usage

### Training

Train a new model on your dataset:

```bash
# Use default settings (hierarchical labels enabled)
python train.py

# Specify custom dataset and model
python train.py --data_path data/my_emails.jsonl --model_name bert-base-uncased

# Disable hierarchical label processing
python train.py --no_hierarchical

# Quick training with sample data
python train.py --sample_size 1000
```

### Evaluation

Evaluate a trained model:

```bash
# Evaluate on the saved test data (recommended)
python evaluate.py

# Quick evaluation with sample
python evaluate.py --sample_size 100

# Evaluate with hierarchical processing disabled
python evaluate.py --no_hierarchical

# Evaluate on a specific test dataset
python evaluate.py --test_data data/test_emails.jsonl
```

### Testing/Inference

Classify new emails:

```bash
# Classify a single email
python test.py "Urgent: Server maintenance required"

# Classify multiple emails
python test.py "Meeting tomorrow" "Sale: 50% off" "Password reset"

# Interactive mode
python test.py --interactive

# Classify emails from a file (one per line)
python test.py --input_file emails_to_classify.txt
```

## Hierarchical Classification Strategy

The system is designed to support a two-stage classification approach:

### Stage 1: Top-Level Classification
- Train a classifier on top-level categories (e.g., `Technical`, `Work`, `Shopping`)
- Use hierarchical labels like `Technical/Security`, `Technical/Infrastructure`

### Stage 2: Subcategory Classification (Future)
- Train specialized classifiers for each top-level category
- Use only the subcategory part of the label (e.g., `Security`, `Infrastructure`)
- Apply the appropriate subcategory classifier based on Stage 1 results

### Benefits:
- **Reduced complexity**: Fewer classes in the main classifier
- **Better performance**: More focused training data per classifier
- **Scalability**: Easy to add new subcategories without retraining the main classifier
- **Flexibility**: Can use simple or hierarchical labels as needed

## Scripts Overview

### `classifier.py`
Contains the core `EmailClassifier` class with methods for:
- `prepare_data()`: Split data into train/eval/test sets (70/15/15)
- `train()`: Train the model
- `classify_emails()`: Make predictions on new emails

### `train.py`
Command-line training script with options for:
- Custom dataset path
- Different base models
- Output directory specification
- Data sampling for quick experiments
- Automatic 70/15/15 train/eval/test split
- Hierarchical label processing

### `evaluate.py`
Comprehensive evaluation script that provides:
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Per-email detailed results
- Overall accuracy
- Uses saved test data for consistency
- Hierarchical label support

### `test.py`
Flexible testing script supporting:
- Command-line email input
- Interactive mode
- File-based input
- Detailed confidence scores

## Data Splits

The system uses a consistent 70/15/15 split:
- **70% Training**: Used to train the model
- **15% Validation**: Used during training for validation and early stopping
- **15% Test**: Withheld completely, used only for final evaluation

This ensures:
- Proper validation during training
- Unbiased final performance evaluation
- Reproducible results with fixed random seed

## Customization

### Using Different Models
You can use any Hugging Face model compatible with sequence classification:

```bash
python train.py --model_name bert-base-uncased
python train.py --model_name roberta-base
python train.py --model_name distilbert-base-uncased
```

### Adding New Categories
Simply add new labels to your JSONL dataset:

```json
{"subject": "Your flight has been confirmed", "body": "Flight details and boarding information.", "label": "Travel/Flights"}
{"subject": "Medical appointment reminder", "body": "Your appointment is scheduled for tomorrow.", "label": "Health/Appointments"}
```

### Training Parameters
Modify training parameters in `classifier.py`:
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimization
- `batch_size`: Batch size for training
- `warmup_steps`: Number of warmup steps

## Customizing Email Processing

The system includes a customizable hook for processing emails before training and inference. This allows you to apply consistent preprocessing across your entire pipeline.

### Using `inference_hook.py`

The `inference_hook.py` file contains a single function `customize_email_for_inference(email)` that is called on every email during both training and inference. By default, it returns the input unchanged.

### Common Customizations

**1. Anonymize Email Addresses and URLs:**
```python
# Uncomment and modify these patterns in inference_hook.py
email_pattern = r'[\w\.-]+@[\w\.-]+'
url_pattern = r'https?://\S+'
# Apply re.sub() to replace with [EMAIL] and [URL]
```

**2. Remove Quoted Text and Reply Headers:**
```python
# Remove lines starting with '>' and "On ... wrote:" patterns
quote_pattern = r'^>.*$'
reply_header_pattern = r'^On .+ wrote:'
```

**3. Clean Ada Chat Transcripts:**
```python
# Remove rote questions from Ada chat transcripts
# Find first USER: that doesn't have "Selected:" after it
```

### Testing Your Customizations

Use the test script to see how your changes affect the emails:

```bash
# Test with colored diff output
python test_inference_hook.py data/your_data.jsonl
```

This will show you exactly what changes are being made to each email with a colored diff format.

### Example Workflow with Custom Processing

1. **Edit `inference_hook.py`** to add your custom processing logic
2. **Test your changes** with `python test_inference_hook.py data/your_data.jsonl`
3. **Train your model** - the same processing will be applied during training
4. **Run inference** - the same processing will be applied during inference

This ensures consistent preprocessing across your entire pipeline without modifying the core classifier code.

## Deployment

### Production Inference API

For production deployments, you can use the dedicated inference API server that provides a FastAPI-based REST API for email classification:

**Repository**: https://github.com/jstanden/email-classifier-inference/

The inference API offers:
- **REST API endpoints** for single and batch email classification
- **Docker containerization** for easy deployment
- **GPU acceleration** with automatic hardware detection (MPS, CUDA, CPU)
- **Batch processing** for efficient handling of multiple emails
- **Health checks** and monitoring capabilities
- **Interactive documentation** with Swagger UI
- **Consistent preprocessing** using the same `inference_hook.py` from this training repository

The API is designed to work seamlessly with models trained using this repository, providing a production-ready interface for your email classification models.

### Basic Deployment Workflow

1. **Train your model** using this repository:
   ```bash
   python train.py --data_path data/your_emails.jsonl
   ```

2. **Copy your trained model** to the inference API's `models/` directory

3. **Deploy the inference API** using Docker:
   ```bash
   git clone https://github.com/jstanden/email-classifier-inference/
   cd email-classifier-inference
   ./deploy.sh run
   ```

4. **Use the API** to classify emails via HTTP requests:
   ```bash
   curl -X POST "http://localhost:8000/classify" \
        -H "Content-Type: application/json" \
        -d '{"subject": "Meeting tomorrow", "body": "Please prepare the quarterly report"}'
   ```

See the [inference API repository](https://github.com/jstanden/email-classifier-inference/) for complete deployment instructions and API documentation.

## Troubleshooting

### Model Not Learning
If the classifier predicts the same label for all emails:
- **Too little data**: Add more training examples (hundreds per class)
- **Class imbalance**: Ensure balanced representation across classes
- **Poor data quality**: Check for clear, distinct patterns in each category

### Memory Issues
- Reduce batch size: `per_device_train_batch_size=4`
- Use a smaller model: `distilbert-base-uncased` instead of `bert-base-uncased`
- Reduce max sequence length in tokenization

### Performance Issues
- Use GPU if available (automatically detected)
- Consider using a smaller model for faster training
- Reduce the number of epochs for quick experiments

### Tokenization Errors
- Long emails are automatically truncated to fit within 512 tokens
- Subject and body are concatenated with a space separator
- Very long emails may lose some content during truncation

## Example Workflow

1. **Prepare your dataset:**
   ```bash
   # Create your JSONL file with subject, body, label columns
   echo '{"subject": "Your email subject", "body": "Your email body", "label": "Category"}' > data/my_emails.jsonl
   ```

2. **Train the model:**
   ```bash
   python train.py --data_path data/my_emails.jsonl
   ```

3. **Evaluate performance:**
   ```bash
   python evaluate.py
   ```

4. **Test on new emails:**
   ```bash
   python test.py "New email to classify"
   ```
