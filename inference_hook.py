import re

def anonymize_emails_and_urls(email):
    """Anonymize email addresses and URLs in the email content."""
    email_pattern = r'[\w\.-]+@[\w\.-]+'
    url_pattern = r'https?://\S+'
    
    if isinstance(email, dict):
        for k in ['subject', 'body']:
            if k in email:
                email[k] = re.sub(email_pattern, '[EMAIL]', email[k])
                email[k] = re.sub(url_pattern, '[URL]', email[k])
    elif isinstance(email, str):
        email = re.sub(email_pattern, '[EMAIL]', email)
        email = re.sub(url_pattern, '[URL]', email)
    
    return email

def remove_quoted_and_reply_lines(email):
    """Remove quoted text (lines starting with '>') and reply headers ('On ... wrote:')."""
    quote_pattern = r'^>.*$'
    reply_header_pattern = r'^On .+ wrote:'
    
    def clean_text(text):
        lines = text.split('\n')
        filtered = [line for line in lines if not re.match(quote_pattern, line) and not re.match(reply_header_pattern, line)]
        return '\n'.join(filtered)
    
    if isinstance(email, dict):
        for k in ['body']:
            if k in email:
                email[k] = clean_text(email[k])
    elif isinstance(email, str):
        email = clean_text(email)
    
    return email

def clean_ada_transcripts(email):
    """Remove rote questions from Ada chat transcripts, keeping only the actual user input."""
    # Get the body content regardless of input type
    if isinstance(email, dict):
        body = email.get('body', '')
    else:
        body = email
    
    # Process the body if it contains USER: patterns
    if 'USER:' in body:
        # Find the first USER: that doesn't have "Selected:" after it
        lines = body.split('\n')
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == 'USER:' and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if not next_line.startswith('Selected:'):
                    start_idx = i
                    break
        
        if start_idx is not None:
            processed_body = '\n'.join(lines[start_idx:])
            
            # Update the appropriate field based on input type
            if isinstance(email, dict):
                email['body'] = processed_body
            else:
                email = processed_body
    
    return email

def customize_email_for_inference(email):
    """
    Customize the content of an email before it is processed for inference or training.
    This function is called on each email in both classify_emails() and load_dataset().
    By default, it returns the input unchanged.
    
    Args:
        email: The email content (string or dict) to be customized.
    Returns:
        The customized email content (same type as input).
    
    Users can modify this function to apply any custom logic needed for
    inference/training-time email processing (e.g., regex cleanup, formatting, etc).
    """
    
    # Uncomment the lines below to enable specific processing:
    
    # email = anonymize_emails_and_urls(email)  # Replace emails and URLs with placeholders
    # email = remove_quoted_and_reply_lines(email)  # Remove quoted text and reply headers
    # email = clean_ada_transcripts(email)  # Clean Ada chat transcripts
    
    return email 