#!/usr/bin/env python3
"""
Mox Bank Negative Focused Processor
Features:
- Focused on Mox Bank negative phrases only
- AI-powered phrase screening and merging
- Removes unmeaningful content for better wordcloud readability
- Custom colors for negative word clouds (RGB(0,184,245))
- Square word clouds without titles
- Saves processed text files
"""

import os
import re
import json
import requests
import shutil
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import time
import logging
from tqdm import tqdm
from config import *

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class MoxNegativeFocusedProcessor:
    def __init__(self, deepseek_api_key: str):
        self.deepseek_api_key = deepseek_api_key
        self.api_base_url = DEEPSEEK_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        # Custom color for negative word clouds
        self.negative_color = (0, 184, 245)  # RGB for negative
        
        # Comprehensive stop words
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'would', 'could', 'should', 'ought',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn',
            'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
            'weren', 'won', 'wouldn', 'im', 'youre', 'hes', 'shes', 'its', 'were', 'theyre',
            'ive', 'youve', 'weve', 'theyve', 'id', 'youd', 'hed', 'shed', 'wed', 'theyd',
            'ill', 'youll', 'hell', 'shell', 'well', 'theyll', 'isnt', 'arent', 'wasnt',
            'weren', 'hasnt', 'havent', 'hadnt', 'doesnt', 'dont', 'didnt', 'wont', 'wouldnt',
            'couldnt', 'shouldnt', 'lets', 'thats', 'whos', 'whats', 'heres', 'theres', 'whens',
            'wheres', 'whys', 'hows', 'us', 'him', 'her', 'them', 'their', 'ours', 'yours',
            'mine', 'yours', 'his', 'hers', 'theirs', 'myself', 'yourself', 'himself', 'herself',
            'itself', 'ourselves', 'yourselves', 'themselves'
        }
    
    def backup_existing_wordclouds(self):
        """Backup existing wordclouds before generating new ones."""
        backup_dir = os.path.join(OUTPUT_DIR, 'backup')
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = os.path.join(backup_dir, f"backup_{timestamp}")
        os.makedirs(backup_subdir, exist_ok=True)
        
        if os.path.exists(OUTPUT_DIR):
            for file in os.listdir(OUTPUT_DIR):
                if file.endswith('.png') and 'wordcloud' in file and 'mox' in file and 'negative' in file:
                    src = os.path.join(OUTPUT_DIR, file)
                    dst = os.path.join(backup_subdir, file)
                    shutil.copy2(src, dst)
                    logger.info(f"Backed up: {file}")
        
        logger.info(f"Backup completed: {backup_subdir}")
        return backup_subdir
    
    def is_english(self, text: str) -> bool:
        """Check if text is primarily English."""
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        ascii_ratio = sum(1 for c in cleaned if ord(c) < 128) / len(cleaned) if cleaned else 0
        return ascii_ratio > ENGLISH_DETECTION_THRESHOLD
    
    def translate_to_english(self, text: str) -> str:
        """Translate non-English text to English using DeepSeek API."""
        if self.is_english(text):
            return text
            
        try:
            prompt = f"""Translate the following text to English. If it's already in English, return it as is. 
            Only return the translated text, nothing else.
            
            Text: "{text}"
            
            Translation:"""
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": TRANSLATION_MAX_TOKENS,
                "temperature": TRANSLATION_TEMPERATURE
            }
            
            response = requests.post(self.api_base_url, headers=self.headers, json=payload, timeout=API_TIMEOUT)
            response.raise_for_status()
            
            result = response.json()
            translated = result['choices'][0]['message']['content'].strip()
            translated = re.sub(r'^["\']|["\']$', '', translated)
            
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed for '{text}': {e}")
            return text
    
    def ai_screen_phrases(self, phrases: Dict[str, int]) -> Dict[str, int]:
        """Use AI to screen phrases and remove unmeaningful content."""
        if not phrases:
            return phrases
        
        # Sort phrases by frequency
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        
        # Group phrases into batches for AI processing
        batch_size = 15
        screened_phrases = {}
        
        print("AI screening phrases for meaningful content...")
        
        for i in range(0, len(sorted_phrases), batch_size):
            batch = sorted_phrases[i:i + batch_size]
            
            # Create prompt for AI screening
            phrase_list = [f"'{phrase}' (frequency: {freq})" for phrase, freq in batch]
            phrases_text = "\n".join(phrase_list)
            
            prompt = f"""Analyze these negative sentiment phrases about a banking app. 
            Keep only phrases that are meaningful, clear, and represent genuine negative feedback.
            Remove phrases that are:
            1. Incomplete or fragmented
            2. Too generic or vague
            3. Technical jargon without clear meaning
            4. Repetitive or redundant
            5. Unclear or confusing
            
            Phrases to analyze:
            {phrases_text}
            
            Return the result as a JSON array of objects with 'phrase' and 'frequency' fields.
            Only include phrases that are meaningful and clear.
            
            Example format:
            [
                {{"phrase": "meaningful phrase", "frequency": original_frequency}},
                {{"phrase": "another meaningful phrase", "frequency": original_frequency}}
            ]
            
            Result:"""
            
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.2
                }
                
                response = requests.post(self.api_base_url, headers=self.headers, json=payload, timeout=API_TIMEOUT)
                response.raise_for_status()
                
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                
                # Try to parse JSON response
                try:
                    screened_batch = json.loads(ai_response)
                    for item in screened_batch:
                        if 'phrase' in item and 'frequency' in item:
                            phrase = item['phrase'].strip()
                            if phrase and len(phrase) >= 3:
                                screened_phrases[phrase] = item['frequency']
                except json.JSONDecodeError:
                    # If JSON parsing fails, keep original phrases
                    for phrase, freq in batch:
                        if phrase.strip():
                            screened_phrases[phrase] = freq
                
                time.sleep(API_RETRY_DELAY)
                
            except Exception as e:
                logger.error(f"AI screening failed for batch: {e}")
                # Keep original phrases if AI fails
                for phrase, freq in batch:
                    if phrase.strip():
                        screened_phrases[phrase] = freq
        
        return screened_phrases
    
    def ai_merge_similar_phrases(self, phrases: Dict[str, int]) -> Dict[str, int]:
        """Use AI to merge phrases with similar meanings."""
        if len(phrases) <= 1:
            return phrases
        
        # Sort phrases by frequency
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        
        # Group phrases into batches for AI processing
        batch_size = 12
        merged_phrases = {}
        
        print("AI merging similar phrases...")
        
        for i in range(0, len(sorted_phrases), batch_size):
            batch = sorted_phrases[i:i + batch_size]
            
            if len(batch) == 1:
                phrase, freq = batch[0]
                if phrase.strip():
                    merged_phrases[phrase] = freq
                continue
            
            # Create prompt for AI merging
            phrase_list = [f"'{phrase}' (frequency: {freq})" for phrase, freq in batch]
            phrases_text = "\n".join(phrase_list)
            
            prompt = f"""Analyze these negative sentiment phrases about a banking app and merge those with similar meanings. 
            Combine frequencies for merged phrases. Choose the most appropriate phrase to represent each group.
            
            Phrases:
            {phrases_text}
            
            Return the result as a JSON array of objects with 'phrase' and 'frequency' fields.
            Only merge phrases that are truly similar in meaning. If phrases are distinct, keep them separate.
            
            Example format:
            [
                {{"phrase": "merged phrase", "frequency": total_frequency}},
                {{"phrase": "separate phrase", "frequency": original_frequency}}
            ]
            
            Result:"""
            
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.3
                }
                
                response = requests.post(self.api_base_url, headers=self.headers, json=payload, timeout=API_TIMEOUT)
                response.raise_for_status()
                
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                
                # Try to parse JSON response
                try:
                    merged_batch = json.loads(ai_response)
                    for item in merged_batch:
                        if 'phrase' in item and 'frequency' in item:
                            phrase = item['phrase'].strip()
                            if phrase and len(phrase) >= 3:
                                merged_phrases[phrase] = merged_phrases.get(phrase, 0) + item['frequency']
                except json.JSONDecodeError:
                    # If JSON parsing fails, keep original phrases
                    for phrase, freq in batch:
                        if phrase.strip():
                            merged_phrases[phrase] = merged_phrases.get(phrase, 0) + freq
                
                time.sleep(API_RETRY_DELAY)
                
            except Exception as e:
                logger.error(f"AI merging failed for batch: {e}")
                # Keep original phrases if AI fails
                for phrase, freq in batch:
                    if phrase.strip():
                        merged_phrases[phrase] = merged_phrases.get(phrase, 0) + freq
        
        return merged_phrases
    
    def parse_phrase_file(self, filepath: str) -> Dict[str, int]:
        """Parse a phrase file and extract phrases with their frequencies."""
        phrases = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            pattern = r"'([^']+)' \(frequency: (\d+)\)"
            matches = re.findall(pattern, content)
            
            for phrase, freq in matches:
                phrases[phrase] = int(freq)
                
            logger.info(f"Parsed {len(phrases)} phrases from {filepath}")
            return phrases
            
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return {}
    
    def save_processed_phrases_to_txt(self, phrases: Dict[str, int], filename: str):
        """Save processed phrases to a text file."""
        os.makedirs('processed_phrases', exist_ok=True)
        filepath = os.path.join('processed_phrases', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Mox Bank Negative - AI Focused Processed Phrases\n")
            f.write("="*50 + "\n\n")
            
            # Sort by frequency (descending)
            sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
            
            for phrase, freq in sorted_phrases:
                f.write(f"'{phrase}' (frequency: {freq})\n")
        
        logger.info(f"Saved processed phrases to {filepath}")
    
    def translate_phrases(self, phrases: Dict[str, int]) -> Dict[str, int]:
        """Translate phrases to English."""
        if not phrases:
            return {}
        
        translated_phrases = {}
        
        print("Translating non-English phrases...")
        with tqdm(total=len(phrases), desc="Translating phrases") as pbar:
            for phrase, freq in phrases.items():
                translated = self.translate_to_english(phrase)
                translated_phrases[translated] = translated_phrases.get(translated, 0) + freq
                pbar.update(1)
                time.sleep(0.1)  # Small delay to avoid rate limiting
        
        return translated_phrases
    
    def basic_filtering(self, phrases: Dict[str, int]) -> Dict[str, int]:
        """Basic filtering of phrases."""
        filtered_phrases = {}
        
        print("Basic filtering...")
        with tqdm(total=len(phrases), desc="Filtering phrases") as pbar:
            for phrase, freq in phrases.items():
                # Basic filtering
                if len(phrase.strip()) >= 3 and len(phrase.strip()) <= 80:
                    # Check for stop words dominance
                    words = phrase.lower().split()
                    stop_word_count = sum(1 for word in words if word in self.stop_words)
                    if stop_word_count / len(words) <= 0.7:
                        filtered_phrases[phrase] = freq
                pbar.update(1)
        
        logger.info(f"Filtered {len(phrases)} phrases to {len(filtered_phrases)} phrases")
        return filtered_phrases
    
    def process_mox_negative(self) -> Dict[str, int]:
        """Process Mox Bank negative phrases with AI enhancement."""
        filename = "mox_bank_negative_phrases.txt"
        filepath = os.path.join('.', filename)
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return {}
        
        print(f"Processing {filename}...")
        
        # Parse phrases
        phrases = self.parse_phrase_file(filepath)
        
        if not phrases:
            return {}
        
        # Round 1: Translate non-English phrases
        print("Round 1: Translating phrases...")
        translated_phrases = self.translate_phrases(phrases)
        
        # Round 2: Basic filtering
        print("Round 2: Basic filtering...")
        filtered_phrases = self.basic_filtering(translated_phrases)
        
        # Round 3: AI screening for meaningful content
        print("Round 3: AI screening for meaningful content...")
        screened_phrases = self.ai_screen_phrases(filtered_phrases)
        
        # Round 4: AI merging similar phrases
        print("Round 4: AI merging similar phrases...")
        final_phrases = self.ai_merge_similar_phrases(screened_phrases)
        
        # Save processed phrases to text file
        processed_filename = "mox_bank_negative_ai_focused_processed_phrases.txt"
        self.save_processed_phrases_to_txt(final_phrases, processed_filename)
        
        return final_phrases
    
    def generate_wordcloud(self, phrases: Dict[str, int], filename: str):
        """Generate a word cloud from phrases with custom colors."""
        if not phrases:
            logger.warning(f"No phrases to generate wordcloud for {filename}")
            return
        
        # Create custom color function
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return f"rgb({self.negative_color[0]}, {self.negative_color[1]}, {self.negative_color[2]})"
        
        wordcloud = WordCloud(
            width=1000,  # Larger square format
            height=1000,  # Larger square format
            background_color='white',
            max_words=150,  # More words for better visibility
            relative_scaling=0.8,  # Better word size distribution
            collocations=False,  # Avoid repeated words
            prefer_horizontal=0.7,  # Mix horizontal and vertical text
            min_font_size=12,  # Larger minimum font size
            max_font_size=120,  # Larger maximum font size
            color_func=color_func
        ).generate_from_frequencies(phrases)
        
        plt.figure(figsize=(12, 12))  # Larger square figure
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # Higher DPI for better quality
        plt.close()
        
        logger.info(f"Generated wordcloud: {filename}")
    
    def print_summary(self, phrases: Dict[str, int]):
        """Print a summary of the processing results."""
        print("\n" + "="*60)
        print("MOX BANK NEGATIVE - AI FOCUSED PROCESSING SUMMARY")
        print("="*60)
        
        total_freq = sum(phrases.values())
        print(f"Total unique phrases: {len(phrases)}")
        print(f"Total frequency: {total_freq}")
        
        # Show top 10 phrases
        top_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 phrases:")
        for i, (phrase, freq) in enumerate(top_phrases, 1):
            print(f"  {i:2d}. '{phrase}' (frequency: {freq})")

def main():
    """Main function to run the Mox Bank negative focused processor."""
    api_key = DEEPSEEK_API_KEY
    
    if not api_key:
        api_key = input("Please enter your DeepSeek API key: ").strip()
        
    if not api_key:
        print("Error: API key is required")
        print("You can either:")
        print("1. Add your API key to the DEEPSEEK_API_KEY variable in config.py")
        print("2. Enter it when prompted")
        return
    
    print("Mox Bank Negative - AI Focused Processor")
    print("="*50)
    print(f"Negative color: RGB{MoxNegativeFocusedProcessor(api_key).negative_color}")
    print(f"Backing up existing wordclouds...")
    
    processor = MoxNegativeFocusedProcessor(api_key)
    
    # Backup existing wordclouds
    backup_dir = processor.backup_existing_wordclouds()
    
    print("Processing Mox Bank negative phrases with AI enhancement...")
    start_time = time.time()
    
    results = processor.process_mox_negative()
    
    if results:
        print("Generating AI-focused word cloud...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"{OUTPUT_DIR}/mox_bank_negative_ai_focused_wordcloud.png"
        processor.generate_wordcloud(results, filename)
        
        processor.print_summary(results)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\nAI-focused processing complete!")
        print(f"Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
        print(f"Backup location: {backup_dir}")
        print(f"Processed phrases saved to 'processed_phrases/' directory")
        print(f"Check the '{OUTPUT_DIR}' directory for the generated wordcloud.")
    else:
        print("No results generated. Please check the input file.")

if __name__ == "__main__":
    main() 