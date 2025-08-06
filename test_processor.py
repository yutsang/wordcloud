#!/usr/bin/env python3
"""
Test script for the sentiment processor
Tests basic functionality without requiring API calls
"""

import os
import sys
import json
from sentiment_processor import SentimentProcessor

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    try:
        import requests
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        import numpy as np
        from difflib import SequenceMatcher
        print("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        return False

def test_file_parsing():
    """Test file parsing functionality."""
    print("\nTesting file parsing...")
    
    # Create a test processor with dummy API key
    processor = SentimentProcessor("dummy_key")
    
    # Test with a sample phrase file
    test_content = """Positive Phrases for Test Bank
==================================================

Total phrases: 3
Top phrases shown: 3

'easy to use' (frequency: 10)
'very good' (frequency: 5)
'great app' (frequency: 3)
"""
    
    # Write test file
    with open("test_phrases.txt", "w", encoding="utf-8") as f:
        f.write(test_content)
    
    try:
        phrases = processor.parse_phrase_file("test_phrases.txt")
        expected = {'easy to use': 10, 'very good': 5, 'great app': 3}
        
        if phrases == expected:
            print("✓ File parsing works correctly")
            success = True
        else:
            print(f"✗ File parsing failed. Expected: {expected}, Got: {phrases}")
            success = False
            
    except Exception as e:
        print(f"✗ File parsing error: {e}")
        success = False
    
    # Clean up
    if os.path.exists("test_phrases.txt"):
        os.remove("test_phrases.txt")
    
    return success

def test_phrase_merging():
    """Test phrase merging functionality."""
    print("\nTesting phrase merging...")
    
    processor = SentimentProcessor("dummy_key")
    
    # Test phrases with similar meanings
    test_phrases = {
        'easy to use': 10,
        'easy to': 5,
        'very good': 8,
        'very good app': 3,
        'great app': 7,
        'great': 4
    }
    
    try:
        merged = processor.merge_similar_phrases(test_phrases, similarity_threshold=0.8)
        print(f"✓ Phrase merging completed. Original: {len(test_phrases)}, Merged: {len(merged)}")
        print(f"  Merged phrases: {merged}")
        return True
    except Exception as e:
        print(f"✗ Phrase merging error: {e}")
        return False

def test_english_detection():
    """Test English detection functionality."""
    print("\nTesting English detection...")
    
    processor = SentimentProcessor("dummy_key")
    
    test_cases = [
        ("easy to use", True),
        ("very good", True),
        ("非常好用", False),
        ("easy to use 非常好用", False),
        ("123 easy", True),
        ("", False)  # Empty string should not be considered English
    ]
    
    success = True
    for text, expected in test_cases:
        result = processor.is_english(text)
        if result == expected:
            print(f"✓ '{text}' -> {result}")
        else:
            print(f"✗ '{text}' -> {result} (expected {expected})")
            success = False
    
    return success

def test_file_existence():
    """Test if required phrase files exist."""
    print("\nTesting file existence...")
    
    required_files = [
        'mox_bank_positive_phrases.txt',
        'mox_bank_negative_phrases.txt',
        'welab_bank_positive_phrases.txt',
        'welab_bank_negative_phrases.txt',
        'za_bank_positive_phrases.txt',
        'za_bank_negative_phrases.txt'
    ]
    
    missing_files = []
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} (missing)")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} files are missing:")
        for filename in missing_files:
            print(f"  - {filename}")
        return False
    else:
        print("\n✓ All required files found")
        return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("SENTIMENT PROCESSOR TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_file_parsing,
        test_phrase_merging,
        test_english_detection,
        test_file_existence
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The processor is ready to use.")
        print("\nNext steps:")
        print("1. Get your DeepSeek API key")
        print("2. Run: python sentiment_processor.py")
    else:
        print("✗ Some tests failed. Please fix the issues before running the processor.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 