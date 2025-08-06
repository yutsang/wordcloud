#!/usr/bin/env python3
"""
Demo script for the sentiment processor
Shows how the program works with a small sample of data
"""

import os
import json
from sentiment_processor import SentimentProcessor

def create_demo_files():
    """Create demo phrase files for testing."""
    
    # Demo positive phrases
    positive_content = """Positive Phrases for Demo Bank
==================================================

Total phrases: 8
Top phrases shown: 8

'easy to use' (frequency: 15)
'very good' (frequency: 12)
'great app' (frequency: 8)
'user friendly' (frequency: 6)
'convenient' (frequency: 4)
'fast and reliable' (frequency: 3)
'fast' (frequency: 2)
'reliable' (frequency: 2)
"""
    
    # Demo negative phrases
    negative_content = """Negative Phrases for Demo Bank
==================================================

Total phrases: 6
Top phrases shown: 6

'not working' (frequency: 10)
'failed to' (frequency: 8)
'app crashes' (frequency: 6)
'poor customer service' (frequency: 4)
'bad experience' (frequency: 3)
'very slow' (frequency: 2)
"""
    
    # Write demo files
    with open("demo_bank_positive_phrases.txt", "w", encoding="utf-8") as f:
        f.write(positive_content)
    
    with open("demo_bank_negative_phrases.txt", "w", encoding="utf-8") as f:
        f.write(negative_content)
    
    print("✓ Created demo phrase files")

def run_demo():
    """Run a demo of the sentiment processor."""
    print("=" * 60)
    print("SENTIMENT PROCESSOR DEMO")
    print("=" * 60)
    
    # Create demo files
    create_demo_files()
    
    # Initialize processor with dummy API key (won't make actual API calls)
    processor = SentimentProcessor("demo_key")
    
    print("\n1. Testing file parsing...")
    
    # Parse demo files
    positive_phrases = processor.parse_phrase_file("demo_bank_positive_phrases.txt")
    negative_phrases = processor.parse_phrase_file("demo_bank_negative_phrases.txt")
    
    print(f"   Positive phrases: {positive_phrases}")
    print(f"   Negative phrases: {negative_phrases}")
    
    print("\n2. Testing phrase merging...")
    
    # Test phrase merging
    merged_positive = processor.merge_similar_phrases(positive_phrases, similarity_threshold=0.7)
    merged_negative = processor.merge_similar_phrases(negative_phrases, similarity_threshold=0.7)
    
    print(f"   Original positive phrases: {len(positive_phrases)}")
    print(f"   Merged positive phrases: {len(merged_positive)}")
    print(f"   Merged positive: {merged_positive}")
    
    print(f"   Original negative phrases: {len(negative_phrases)}")
    print(f"   Merged negative phrases: {len(merged_negative)}")
    print(f"   Merged negative: {merged_negative}")
    
    print("\n3. Testing English detection...")
    
    # Test English detection
    test_phrases = [
        "easy to use",
        "very good",
        "非常好用",
        "easy to use 非常好用",
        "123 easy"
    ]
    
    for phrase in test_phrases:
        is_eng = processor.is_english(phrase)
        print(f"   '{phrase}' -> {'English' if is_eng else 'Non-English'}")
    
    print("\n4. Testing word cloud generation...")
    
    # Create results structure
    results = {
        "demo_bank": {
            "positive": merged_positive,
            "negative": merged_negative
        }
    }
    
    # Generate word clouds
    try:
        processor.generate_all_wordclouds(results)
        print("   ✓ Word clouds generated successfully")
        print("   Check the 'wordclouds' directory for generated images")
    except Exception as e:
        print(f"   ✗ Word cloud generation failed: {e}")
    
    print("\n5. Testing results saving...")
    
    # Save results
    processor.save_processed_results(results, "demo_results.json")
    print("   ✓ Results saved to demo_results.json")
    
    # Clean up demo files
    cleanup_demo_files()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print("\nThis demo shows the basic functionality of the sentiment processor.")
    print("To process your actual data:")
    print("1. Get your DeepSeek API key")
    print("2. Add your API key to config.py: DEEPSEEK_API_KEY = 'your_key_here'")
    print("3. Run: python sentiment_processor.py")

def cleanup_demo_files():
    """Clean up demo files."""
    demo_files = [
        "demo_bank_positive_phrases.txt",
        "demo_bank_negative_phrases.txt",
        "demo_results.json"
    ]
    
    for filename in demo_files:
        if os.path.exists(filename):
            os.remove(filename)
    
    print("   ✓ Cleaned up demo files")

if __name__ == "__main__":
    run_demo() 