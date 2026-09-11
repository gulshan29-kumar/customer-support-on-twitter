"""Reconstruct customer-support pairs from Kaggle dataset for AmazonHelp.

Extracts AmazonHelp support responses, maps them to inbound customer tweets,
cleans texts, curates the 200-example golden evaluation set, and saves the
retrieval/training corpus with zero leakage.
"""

import os
import zipfile
import re
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    DATA_DIR,
    HISTORICAL_PAIRS_PATH,
    GOLDEN_SET_PATH,
    TARGET_BRAND,
    INTENT_CLASSES,
)
from src.preprocessing import clean_text, is_valid_message

RAW_ZIP_PATH = Path(r"C:\Users\Gulshan Kumar\Downloads\twcs.csv.zip")
RANDOM_SEED = 42


def extract_amazon_pairs(zip_path: Path, max_pairs: int = 40000) -> pd.DataFrame:
    """Stream twcs.csv from zip and reconstruct AmazonHelp (customer, support) pairs."""
    print(f"Reading from {zip_path}...")
    
    # Phase 1: Collect AmazonHelp support tweets and their target customer tweet IDs
    amazon_support = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("twcs.csv") as f:
            for chunk in pd.read_csv(
                f,
                chunksize=150000,
                usecols=["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"],
                dtype={"tweet_id": str, "author_id": str, "inbound": bool, "text": str, "in_response_to_tweet_id": str},
            ):
                mask = (chunk["author_id"] == TARGET_BRAND) & (~chunk["inbound"]) & chunk["in_response_to_tweet_id"].notna()
                filtered = chunk[mask]
                for _, row in filtered.iterrows():
                    cust_id = str(row["in_response_to_tweet_id"]).replace(".0", "").strip()
                    if cust_id and cust_id not in amazon_support:
                        amazon_support[cust_id] = {
                            "support_tweet_id": str(row["tweet_id"]).replace(".0", "").strip(),
                            "support_text": row["text"],
                            "support_author_id": TARGET_BRAND,
                            "brand": TARGET_BRAND,
                        }
                    if len(amazon_support) >= max_pairs * 2:
                        break
                if len(amazon_support) >= max_pairs * 2:
                    break

    print(f"Found {len(amazon_support)} candidate AmazonHelp replies pointing to customer tweets.")
    target_cust_ids = set(amazon_support.keys())

    # Phase 2: Find the matching customer inbound tweets
    records = []
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("twcs.csv") as f:
            for chunk in pd.read_csv(
                f,
                chunksize=150000,
                usecols=["tweet_id", "inbound", "text"],
                dtype={"tweet_id": str, "inbound": bool, "text": str},
            ):
                cust_chunk = chunk[chunk["inbound"] & chunk["tweet_id"].isin(target_cust_ids)]
                for _, row in cust_chunk.iterrows():
                    cid = str(row["tweet_id"]).replace(".0", "").strip()
                    sup = amazon_support[cid]
                    raw_text = str(row["text"])
                    cleaned = clean_text(raw_text)
                    if is_valid_message(cleaned, min_tokens=4):
                        records.append({
                            "customer_tweet_id": cid,
                            "customer_text": cleaned,
                            "raw_customer_text": raw_text,
                            "support_tweet_id": sup["support_tweet_id"],
                            "support_text": sup["support_text"],
                            "support_author_id": sup["support_author_id"],
                            "brand": TARGET_BRAND,
                        })
                    if len(records) >= max_pairs:
                        break
                if len(records) >= max_pairs:
                    break

    df = pd.DataFrame(records).drop_duplicates(subset=["customer_text"])
    print(f"Reconstructed {len(df)} unique valid customer-support pairs.")
    return df


def classify_text_intent(text: str) -> str:
    """Accurately classify an Amazon customer message into the 9-class taxonomy based on issue signals."""
    t = text.lower()
    
    # 1. Delivery or Delay
    if any(k in t for k in [
        "late delivery", "delayed", "delay", "tracking", "track my", "shipment",
        "shipping", "hasn't arrived", "not arrived", "haven't received", "package",
        "courier", "where is my order", "order status", "out for delivery", "dispatch",
        "delivered but", "carrier", "parcel", "estimated delivery", "deliver"
    ]):
        return "delivery_or_delay"

    # 2. Refund or Compensation
    if any(k in t for k in [
        "refund", "money back", "reimburse", "compensation", "credit note",
        "return refund", "cashback", "refunded", "deducted refund", "claim refund"
    ]):
        return "refund_or_compensation"

    # 3. Payment or Billing
    if any(k in t for k in [
        "charged", "charge", "billing", "payment", "unauthorized transaction",
        "credit card", "debit card", "pay balance", "bank account", "invoice",
        "deducted", "double charge", "charged twice", "gift card balance", "prime fee"
    ]):
        return "payment_or_billing"

    # 4. Account or Access
    if any(k in t for k in [
        "login", "log in", "sign in", "password", "otp", "hacked", "suspended",
        "blocked account", "2fa", "two-factor", "verification code", "reset password",
        "access my account", "locked out", "compromised"
    ]):
        return "account_or_access"

    # 5. Technical Issue
    if any(k in t for k in [
        "app crashing", "app crash", "website error", "error code", "kindle",
        "fire tv", "firestick", "alexa", "echo", "prime video", "streaming error",
        "loading issue", "server error", "bug", "glitch", "crash", "not working",
        "stuck on loading", "video quality", "audio out of sync", "firmware"
    ]):
        return "technical_issue"

    # 6. Booking or Travel (Amazon Travel / Flights / Bus tickets edge cases or travel tickets)
    if any(k in t for k in [
        "flight", "ticket", "booking", "boarding", "reservation", "train ticket",
        "bus ticket", "pnr", "hotel booking", "travel", "seat selection", "reschedule flight"
    ]):
        return "booking_or_travel"

    # 7. Product or Service Info
    if any(k in t for k in [
        "available", "in stock", "restock", "warranty", "specification", "specs",
        "compatible", "dimensions", "how to use", "offer", "discount code", "promo",
        "price match", "trade-in", "difference between", "prime membership benefits"
    ]):
        return "product_or_service_info"

    # 8. Complaint or Feedback
    if any(k in t for k in [
        "worst service", "terrible", "disappointed", "pathetic", "useless support",
        "rude", "horrible", "ridiculous", "poor service", "waste of money",
        "bad customer service", "unacceptable", "scam", "shame on you", "hate amazon"
    ]):
        return "complaint_or_feedback"

    # 9. Other (ambiguous, generic, social)
    return "other"


def create_datasets():
    """Build golden set (200 examples) and historical retrieval corpus (strictly disjoint)."""
    np.random.seed(RANDOM_SEED)

    if not RAW_ZIP_PATH.exists():
        raise FileNotFoundError(f"Raw Kaggle zip not found at {RAW_ZIP_PATH}")

    df_pairs = extract_amazon_pairs(RAW_ZIP_PATH, max_pairs=30000)

    # Assign high-quality intent labels
    df_pairs["assigned_intent"] = df_pairs["customer_text"].apply(classify_text_intent)

    # Inspect initial intent breakdown
    counts = df_pairs["assigned_intent"].value_counts()
    print("Class breakdown in raw pool:\n", counts)

    # Target balanced golden set of exactly 200 examples across the 9 classes
    # Aim for roughly 20-25 examples per class, with 'other' and rare classes adequately represented
    target_distribution = {
        "delivery_or_delay": 28,
        "refund_or_compensation": 24,
        "payment_or_billing": 24,
        "technical_issue": 24,
        "account_or_access": 22,
        "product_or_service_info": 22,
        "complaint_or_feedback": 22,
        "other": 20,
        "booking_or_travel": 14,  # travel/booking is rarer in retail support
    }

    # Verify total
    total_target = sum(target_distribution.values())
    assert total_target == 200, f"Total must be 200, got {total_target}"

    golden_rows = []
    golden_cust_ids = set()

    # Sample for each intent category
    for intent, n_samples in target_distribution.items():
        subset = df_pairs[df_pairs["assigned_intent"] == intent]
        if len(subset) < n_samples:
            sampled = subset
        else:
            # Sample with varying lengths
            subset = subset.copy()
            subset["char_len"] = subset["customer_text"].str.len()
            subset = subset.sort_values(by="char_len")
            # Sample across length quantiles
            indices = np.linspace(0, len(subset) - 1, n_samples, dtype=int)
            sampled = subset.iloc[indices]
            
        for _, row in sampled.iterrows():
            golden_rows.append({
                "id": len(golden_rows) + 1,
                "customer_text": row["customer_text"],
                "gold_intent": intent,
                "brand": TARGET_BRAND,
                "support_tweet_id": row["support_tweet_id"],
                "customer_tweet_id": row["customer_tweet_id"],
            })
            golden_cust_ids.add(row["customer_tweet_id"])

    # If short of 200 due to rare classes, top up from remaining pool
    if len(golden_rows) < 200:
        remaining_pool = df_pairs[~df_pairs["customer_tweet_id"].isin(golden_cust_ids)]
        needed = 200 - len(golden_rows)
        topup = remaining_pool.sample(n=needed, random_state=RANDOM_SEED)
        for _, row in topup.iterrows():
            golden_rows.append({
                "id": len(golden_rows) + 1,
                "customer_text": row["customer_text"],
                "gold_intent": row["assigned_intent"],
                "brand": TARGET_BRAND,
                "support_tweet_id": row["support_tweet_id"],
                "customer_tweet_id": row["customer_tweet_id"],
            })
            golden_cust_ids.add(row["customer_tweet_id"])

    df_golden = pd.DataFrame(golden_rows).iloc[:200]
    df_golden["id"] = range(1, 201)

    # CRITICAL: Exclude all golden set customer tweets from historical corpus!
    df_corpus = df_pairs[~df_pairs["customer_tweet_id"].isin(set(df_golden["customer_tweet_id"]))].copy()
    
    # Also exclude any identical customer texts to prevent data leakage
    df_corpus = df_corpus[~df_corpus["customer_text"].isin(set(df_golden["customer_text"]))].copy()
    df_corpus.rename(columns={"assigned_intent": "intent"}, inplace=True)

    # Save to data directory
    df_golden.to_csv(GOLDEN_SET_PATH, index=False)
    df_corpus.to_csv(HISTORICAL_PAIRS_PATH, index=False)

    print(f"\nSuccessfully created Golden Set: {len(df_golden)} rows -> {GOLDEN_SET_PATH}")
    print(df_golden["gold_intent"].value_counts())
    print(f"\nSuccessfully created Historical Corpus: {len(df_corpus)} rows -> {HISTORICAL_PAIRS_PATH}")
    print("Leakage verification: intersection of customer IDs =", len(set(df_golden['customer_tweet_id']) & set(df_corpus['customer_tweet_id'])))


if __name__ == "__main__":
    create_datasets()
