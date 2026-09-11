"""Text preprocessing module for cleaning Twitter customer support messages."""

import re
import html

# Regex patterns for Twitter cleaning
MENTION_PATTERN = re.compile(r"@\w+", re.UNICODE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EXTRA_WHITESPACE_PATTERN = re.compile(r"\s+")
SPECIAL_CHARS_PATTERN = re.compile(r"[\r\n\t]+")


def clean_text(text: str) -> str:
    """Clean a customer tweet by stripping mentions, URLs, unescaping HTML, and normalizing whitespace.
    
    Args:
        text: Raw tweet text string.
        
    Returns:
        Cleaned, normalized text string.
    """
    if not isinstance(text, str):
        return ""

    # Unescape HTML entities (e.g. &amp; -> &)
    text = html.unescape(text)

    # Remove @mentions (e.g. @AmazonHelp)
    text = MENTION_PATTERN.sub(" ", text)

    # Remove URLs (http://, https://, www.)
    text = URL_PATTERN.sub(" ", text)

    # Replace newlines, tabs with spaces
    text = SPECIAL_CHARS_PATTERN.sub(" ", text)

    # Collapse multiple whitespaces
    text = EXTRA_WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def is_valid_message(text: str, min_tokens: int = 4) -> bool:
    """Check if cleaned text has enough substance for evaluation/retrieval.
    
    Args:
        text: Preprocessed text.
        min_tokens: Minimum number of whitespace-delimited tokens required.
        
    Returns:
        True if the message is deemed valid and non-trivial.
    """
    tokens = text.split()
    return len(tokens) >= min_tokens
