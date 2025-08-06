#!/usr/bin/env python3
"""
AI-Enhanced Sentiment Phrase Processor
Features:
- AI-powered phrase merging using DeepSeek API
- Cleans up problematic phrases like "t, xxxxx" and redundant phrases
- Multiple rounds of AI-based consolidation
- Custom colors for word clouds (positive: RGB(12,35,60), negative: RGB(0,184,245))
- Square word clouds without titles
- Saves processed text files
- Threading-based processing with progress bars
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
from difflib import SequenceMatcher
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from config import *

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class AIEnhancedProcessor:
    def __init__(self, deepseek_api_key: str, num_threads: int = 4):
        self.deepseek_api_key = deepseek_api_key
        self.api_base_url = DEEPSEEK_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
        }
        self.num_threads = num_threads
        
        # Custom colors for word clouds
        self.positive_color = (12, 35, 60)  # RGB for positive
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
        
        # Neutral/technical terms
        self.neutral_terms = {
            'customer service', 'service', 'app', 'bank', 'banking', 'account', 'money', 'payment',
            'transfer', 'transaction', 'login', 'password', 'verification', 'security', 'update',
            'version', 'system', 'process', 'procedure', 'function', 'feature', 'interface', 'ui',
            'ux', 'design', 'layout', 'screen', 'page', 'button', 'menu', 'option', 'setting',
            'configuration', 'data', 'information', 'file', 'document', 'record', 'history',
            'log', 'report', 'status', 'result', 'response', 'message', 'notification', 'alert',
            'error', 'warning', 'success', 'failure', 'problem', 'issue', 'bug', 'glitch', 'crash',
            'freeze', 'hang', 'slow', 'fast', 'speed', 'performance', 'efficiency', 'quality',
            'reliability', 'stability', 'compatibility', 'accessibility', 'usability', 'functionality'
        }
        
        # Positive sentiment words
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect', 'awesome',
            'brilliant', 'outstanding', 'superb', 'terrific', 'fabulous', 'marvelous', 'splendid',
            'magnificent', 'exceptional', 'extraordinary', 'incredible', 'unbelievable', 'remarkable',
            'impressive', 'satisfying', 'pleasing', 'enjoyable', 'delightful', 'lovely', 'beautiful',
            'nice', 'pleasant', 'comfortable', 'convenient', 'easy', 'simple', 'smooth', 'fast',
            'quick', 'efficient', 'effective', 'reliable', 'stable', 'secure', 'safe', 'trustworthy',
            'helpful', 'useful', 'valuable', 'beneficial', 'advantageous', 'profitable', 'successful',
            'working', 'functioning', 'operational', 'available', 'accessible', 'user-friendly',
            'intuitive', 'straightforward', 'clear', 'understandable', 'transparent', 'honest',
            'fair', 'reasonable', 'affordable', 'cheap', 'inexpensive', 'economical', 'cost-effective'
        }
        
        # Negative sentiment words
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'dreadful', 'atrocious', 'abysmal', 'appalling',
            'disgusting', 'revolting', 'nauseating', 'sickening', 'vile', 'foul', 'rotten', 'corrupt',
            'broken', 'damaged', 'defective', 'faulty', 'malfunctioning', 'non-working', 'useless',
            'worthless', 'pointless', 'meaningless', 'unnecessary', 'redundant', 'repetitive',
            'boring', 'tedious', 'monotonous', 'dull', 'uninteresting', 'unappealing', 'unattractive',
            'ugly', 'hideous', 'repulsive', 'offensive', 'insulting', 'rude', 'impolite', 'disrespectful',
            'unprofessional', 'incompetent', 'inefficient', 'ineffective', 'unreliable', 'unstable',
            'insecure', 'unsafe', 'dangerous', 'risky', 'hazardous', 'harmful', 'damaging', 'destructive',
            'expensive', 'costly', 'overpriced', 'unaffordable', 'unreasonable', 'unfair', 'dishonest',
            'deceptive', 'misleading', 'confusing', 'complicated', 'complex', 'difficult', 'hard',
            'challenging', 'frustrating', 'annoying', 'irritating', 'bothersome', 'troublesome',
            'problematic', 'worrisome', 'concerning', 'alarming', 'disturbing', 'upsetting',
            'disappointing', 'dissatisfying', 'unsatisfactory', 'inadequate', 'insufficient',
            'incomplete', 'partial', 'limited', 'restricted', 'blocked', 'prevented', 'stopped',
            'failed', 'crashed', 'froze', 'hung', 'stuck', 'trapped', 'lost', 'missing', 'gone',
            'disappeared', 'vanished', 'erased', 'deleted', 'removed', 'eliminated', 'destroyed',
            'ruined', 'wasted', 'squandered', 'thrown away', 'discarded', 'abandoned', 'neglected',
            'ignored', 'overlooked', 'forgotten', 'unknown', 'unclear', 'uncertain', 'doubtful',
            'suspicious', 'questionable', 'untrustworthy', 'unreliable', 'unstable', 'inconsistent',
            'unpredictable', 'uncontrollable', 'unmanageable', 'unusable', 'inaccessible', 'unavailable'
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
                if file.endswith('.png') and 'wordcloud' in file:
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
    
    def clean_problematic_phrases(self, phrase: str) -> str:
        """Clean up problematic phrases like 't, xxxxx' and redundant phrases."""
        # Remove phrases that start with single letters followed by comma
        if re.match(r'^[a-z],\s*', phrase, re.IGNORECASE):
            return ""
        
        # Remove phrases with excessive punctuation or special characters
        if re.search(r'[^\w\s,\.!?\'"()-]', phrase):
            return ""
        
        # Remove phrases that are just repeated words
        words = phrase.lower().split()
        if len(words) >= 2:
            # Check for repeated words like "good very good"
            for i in range(len(words) - 1):
                if words[i] == words[i + 1]:
                    return ""
            
            # Check for similar words like "good" and "vgood"
            for i in range(len(words) - 1):
                if (words[i] in self.positive_words and words[i + 1].startswith('v') and 
                    words[i + 1][1:] in self.positive_words):
                    return ""
        
        # Remove phrases that are too short or too long
        if len(phrase.strip()) < 3 or len(phrase.strip()) > 100:
            return ""
        
        # Remove phrases with unbalanced parentheses
        if phrase.count('(') != phrase.count(')'):
            return ""
        
        return phrase.strip()
    
    def ai_merge_similar_phrases(self, phrases: Dict[str, int], sentiment: str) -> Dict[str, int]:
        """Use AI to merge phrases with similar meanings."""
        if len(phrases) <= 1:
            return phrases
        
        # Sort phrases by frequency
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        
        # Group phrases into batches for AI processing
        batch_size = 10
        merged_phrases = {}
        
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
            
            prompt = f"""Analyze these {sentiment} sentiment phrases and merge those with similar meanings. 
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
    
    def ai_clean_and_consolidate(self, phrases: Dict[str, int], sentiment: str) -> Dict[str, int]:
        """Use AI to clean and consolidate phrases."""
        if not phrases:
            return phrases
        
        # First, clean problematic phrases
        cleaned_phrases = {}
        for phrase, freq in phrases.items():
            cleaned = self.clean_problematic_phrases(phrase)
            if cleaned:
                cleaned_phrases[cleaned] = cleaned_phrases.get(cleaned, 0) + freq
        
        # Then use AI to merge similar phrases
        consolidated_phrases = self.ai_merge_similar_phrases(cleaned_phrases, sentiment)
        
        return consolidated_phrases
    
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
            f.write(f"AI-Enhanced Processed Phrases - {filename}\n")
            f.write("="*50 + "\n\n")
            
            # Sort by frequency (descending)
            sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
            
            for phrase, freq in sorted_phrases:
                f.write(f"'{phrase}' (frequency: {freq})\n")
        
        logger.info(f"Saved processed phrases to {filepath}")
    
    def translate_phrases_with_threading(self, phrases: Dict[str, int]) -> Dict[str, int]:
        """Translate phrases using threading."""
        if not phrases:
            return {}
        
        phrase_items = list(phrases.items())
        translated_phrases = {}
        
        def translate_single_phrase(item):
            phrase, freq = item
            translated = self.translate_to_english(phrase)
            return translated, freq
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(translate_single_phrase, item) for item in phrase_items]
            
            with tqdm(total=len(futures), desc="Translating phrases") as pbar:
                for future in as_completed(futures):
                    try:
                        translated, freq = future.result()
                        translated_phrases[translated] = translated_phrases.get(translated, 0) + freq
                    except Exception as e:
                        logger.error(f"Error in translation: {e}")
                    pbar.update(1)
        
        return translated_phrases
    
    def process_phrases_with_filtering(self, phrases: Dict[str, int], sentiment: str) -> Dict[str, int]:
        """Process phrases with enhanced filtering."""
        filtered_phrases = {}
        
        with tqdm(total=len(phrases), desc=f"Filtering {sentiment} phrases") as pbar:
            for phrase, freq in phrases.items():
                # Basic filtering
                if len(phrase.strip()) >= 3 and len(phrase.strip()) <= 100:
                    # Check for stop words dominance
                    words = phrase.lower().split()
                    stop_word_count = sum(1 for word in words if word in self.stop_words)
                    if stop_word_count / len(words) <= 0.6:
                        filtered_phrases[phrase] = freq
                pbar.update(1)
        
        logger.info(f"Filtered {len(phrases)} phrases to {len(filtered_phrases)} phrases for {sentiment}")
        return filtered_phrases
    
    def process_all_files(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Process all sentiment phrase files with AI-enhanced filtering and merging."""
        banks = BANKS
        sentiments = SENTIMENTS
        
        results = {}
        
        print(f"\nProcessing with {self.num_threads} threads and AI enhancement...")
        
        for bank in banks:
            results[bank] = {}
            
            for sentiment in sentiments:
                filename = f"{bank}_{sentiment}_phrases.txt"
                filepath = os.path.join('.', filename)
                
                if not os.path.exists(filepath):
                    logger.warning(f"File not found: {filepath}")
                    continue
                
                print(f"\nProcessing {filename}...")
                
                # Parse phrases
                phrases = self.parse_phrase_file(filepath)
                
                if not phrases:
                    continue
                
                # Round 1: Translate non-English phrases
                print("Round 1: Translating phrases...")
                translated_phrases = self.translate_phrases_with_threading(phrases)
                
                # Round 2: Basic filtering
                print("Round 2: Basic filtering...")
                filtered_phrases = self.process_phrases_with_filtering(translated_phrases, sentiment)
                
                # Round 3: AI-powered cleaning and consolidation
                print("Round 3: AI-powered cleaning and consolidation...")
                ai_consolidated = self.ai_clean_and_consolidate(filtered_phrases, sentiment)
                
                # Round 4: Final AI merging
                print("Round 4: Final AI merging...")
                final_phrases = self.ai_merge_similar_phrases(ai_consolidated, sentiment)
                
                # Save processed phrases to text file
                processed_filename = f"{bank}_{sentiment}_ai_enhanced_processed_phrases.txt"
                self.save_processed_phrases_to_txt(final_phrases, processed_filename)
                
                results[bank][sentiment] = final_phrases
                
                time.sleep(API_RETRY_DELAY)
        
        return results
    
    def generate_wordcloud(self, phrases: Dict[str, int], sentiment: str, filename: str):
        """Generate a word cloud from phrases with custom colors."""
        if not phrases:
            logger.warning(f"No phrases to generate wordcloud for {filename}")
            return
        
        # Choose color based on sentiment
        if sentiment == 'positive':
            color = self.positive_color
        else:
            color = self.negative_color
        
        # Create custom color function
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return f"rgb({color[0]}, {color[1]}, {color[2]})"
        
        wordcloud = WordCloud(
            width=800,  # Square format
            height=800,  # Square format
            background_color='white',
            max_words=WORDCLOUD_MAX_WORDS,
            relative_scaling=WORDCLOUD_RELATIVE_SCALING,
            collocations=False,  # Avoid repeated words
            prefer_horizontal=0.7,  # Mix horizontal and vertical text
            min_font_size=10,
            max_font_size=100,
            color_func=color_func
        ).generate_from_frequencies(phrases)
        
        plt.figure(figsize=(10, 10))  # Square figure
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(filename, dpi=WORDCLOUD_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated wordcloud: {filename}")
    
    def generate_all_wordclouds(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Generate word clouds for all banks and sentiments."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        print("\nGenerating AI-enhanced word clouds...")
        
        for bank, sentiments in results.items():
            for sentiment, phrases in sentiments.items():
                if phrases:
                    filename = f"{OUTPUT_DIR}/{bank}_{sentiment}_ai_enhanced_wordcloud.png"
                    self.generate_wordcloud(phrases, sentiment, filename)
    
    def save_processed_results(self, results: Dict[str, Dict[str, Dict[str, int]]], filename: str = "ai_enhanced_processed_phrases.json"):
        """Save processed results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved AI-enhanced processed results to {filename}")
    
    def print_summary(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Print a summary of the processing results."""
        print("\n" + "="*60)
        print("AI-ENHANCED PROCESSING SUMMARY")
        print("="*60)
        
        for bank, sentiments in results.items():
            print(f"\n{bank.replace('_', ' ').title()}:")
            for sentiment, phrases in sentiments.items():
                total_freq = sum(phrases.values())
                print(f"  {sentiment.title()}: {len(phrases)} unique phrases, {total_freq} total frequency")
                
                # Show top 5 phrases
                top_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)[:5]
                print("    Top phrases:")
                for phrase, freq in top_phrases:
                    print(f"      '{phrase}' (frequency: {freq})")

def main():
    """Main function to run the AI-enhanced sentiment processor."""
    parser = argparse.ArgumentParser(description='AI-Enhanced Sentiment Processor')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads to use (default: 4)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    api_key = DEEPSEEK_API_KEY
    
    if not api_key:
        api_key = input("Please enter your DeepSeek API key: ").strip()
        
    if not api_key:
        print("Error: API key is required")
        print("You can either:")
        print("1. Add your API key to the DEEPSEEK_API_KEY variable in config.py")
        print("2. Enter it when prompted")
        return
    
    print("AI-Enhanced Sentiment Processor")
    print("="*50)
    print(f"Using {args.threads} threads")
    print(f"Positive color: RGB{AIEnhancedProcessor(api_key).positive_color}")
    print(f"Negative color: RGB{AIEnhancedProcessor(api_key).negative_color}")
    print(f"Backing up existing wordclouds...")
    
    processor = AIEnhancedProcessor(api_key, num_threads=args.threads)
    
    # Backup existing wordclouds
    backup_dir = processor.backup_existing_wordclouds()
    
    print("Processing sentiment phrases with AI-enhanced filtering and merging...")
    start_time = time.time()
    
    results = processor.process_all_files()
    
    processor.save_processed_results(results)
    
    print("Generating AI-enhanced word clouds...")
    processor.generate_all_wordclouds(results)
    
    processor.print_summary(results)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nAI-enhanced processing complete!")
    print(f"Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
    print(f"Backup location: {backup_dir}")
    print(f"Processed phrases saved to 'processed_phrases/' directory")
    print(f"Check the '{OUTPUT_DIR}' directory for generated images.")

if __name__ == "__main__":
    main() 