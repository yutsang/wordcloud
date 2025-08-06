#!/usr/bin/env python3
"""
Final Enhanced Sentiment Phrase Processor
Features:
- Multi-core processing with progress bars
- Automatic backup system
- Enhanced phrase cleaning and merging
- Better word cloud generation
- Comprehensive similarity detection
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
from config import *

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class FinalSentimentProcessor:
    def __init__(self, deepseek_api_key: str, num_cores: int = 4):
        self.deepseek_api_key = deepseek_api_key
        self.api_base_url = DEEPSEEK_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
        }
        self.num_cores = min(num_cores, mp.cpu_count())
        
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
            'werent', 'hasnt', 'havent', 'hadnt', 'doesnt', 'dont', 'didnt', 'wont', 'wouldnt',
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
    
    def clean_phrase(self, phrase: str) -> str:
        """Clean and normalize a phrase."""
        # Remove leading/trailing whitespace and quotes
        phrase = phrase.strip().strip("'\"")
        
        # Remove incomplete phrases
        incomplete_patterns = [
            r'^\s*and\s+',  # Starts with "and"
            r'^\s*or\s+',   # Starts with "or"
            r'^\s*but\s+',  # Starts with "but"
            r'^\s*the\s+',  # Starts with "the"
            r'^\s*a\s+',    # Starts with "a"
            r'^\s*an\s+',   # Starts with "an"
            r'^\s*in\s+',   # Starts with "in"
            r'^\s*on\s+',   # Starts with "on"
            r'^\s*at\s+',   # Starts with "at"
            r'^\s*to\s+',   # Starts with "to"
            r'^\s*for\s+',  # Starts with "for"
            r'^\s*of\s+',   # Starts with "of"
            r'^\s*with\s+', # Starts with "with"
            r'^\s*by\s+',   # Starts with "by"
        ]
        
        for pattern in incomplete_patterns:
            phrase = re.sub(pattern, '', phrase, flags=re.IGNORECASE)
        
        # Remove trailing incomplete words
        phrase = re.sub(r'\s+\w+\s*$', '', phrase)
        
        # Remove phrases with unbalanced parentheses
        if phrase.count('(') != phrase.count(')'):
            phrase = re.sub(r'\([^)]*$', '', phrase)
            phrase = re.sub(r'^[^(]*\)', '', phrase)
        
        # Remove phrases that end with incomplete words
        if re.search(r'\s+\w{1,2}\s*$', phrase):
            phrase = re.sub(r'\s+\w{1,2}\s*$', '', phrase)
        
        return phrase.strip()
    
    def filter_phrase(self, phrase: str, sentiment: str) -> bool:
        """Filter out phrases that don't meet quality criteria."""
        # Clean the phrase first
        phrase = self.clean_phrase(phrase)
        
        if not phrase or len(phrase) < 3:
            return False
        
        phrase_lower = phrase.lower()
        words = phrase_lower.split()
        
        # 1. Filter out very short phrases
        if len(words) <= 2:
            return False
        
        # 2. Filter out phrases that are mostly stop words
        stop_word_count = sum(1 for word in words if word in self.stop_words)
        if stop_word_count / len(words) > 0.6:  # More than 60% stop words
            return False
        
        # 3. Filter out personal pronouns
        personal_pronouns = {'i', 'me', 'my', 'myself', 'we', 'us', 'our', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves'}
        if any(pronoun in words for pronoun in personal_pronouns):
            return False
        
        # 4. Filter out neutral/technical terms
        if any(neutral in phrase_lower for neutral in self.neutral_terms):
            return False
        
        # 5. Check for sentiment consistency
        if sentiment == 'positive':
            # Check for negative words in positive phrases
            negative_count = sum(1 for word in words if word in self.negative_words)
            if negative_count > 0:
                return False
        elif sentiment == 'negative':
            # Check for positive words in negative phrases
            positive_count = sum(1 for word in words if word in self.positive_words)
            if positive_count > 0:
                return False
        
        # 6. Filter out very long phrases
        if len(phrase) > 80:
            return False
        
        # 7. Filter out phrases with too many special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z\s]', phrase)) / len(phrase)
        if special_char_ratio > 0.25:  # More than 25% special characters
            return False
        
        # 8. Filter out incomplete phrases
        if re.search(r'^\s*(and|or|but|the|a|an|in|on|at|to|for|of|with|by)\s+', phrase, re.IGNORECASE):
            return False
        
        # 9. Filter out phrases with unbalanced punctuation
        if phrase.count('(') != phrase.count(')') or phrase.count('[') != phrase.count(']'):
            return False
        
        return True
    
    def similarity_score(self, phrase1: str, phrase2: str) -> float:
        """Calculate similarity between two phrases using multiple methods."""
        # Method 1: Sequence matcher
        seq_similarity = SequenceMatcher(None, phrase1.lower(), phrase2.lower()).ratio()
        
        # Method 2: Word overlap
        words1 = set(phrase1.lower().split())
        words2 = set(phrase2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        word_similarity = len(intersection) / len(union) if union else 0.0
        
        # Method 3: Substring similarity
        shorter = phrase1.lower() if len(phrase1) < len(phrase2) else phrase2.lower()
        longer = phrase2.lower() if len(phrase1) < len(phrase2) else phrase1.lower()
        
        if shorter in longer:
            substring_similarity = len(shorter) / len(longer)
        else:
            substring_similarity = 0.0
        
        # Combine similarities with weights
        final_similarity = (seq_similarity * 0.4 + word_similarity * 0.4 + substring_similarity * 0.2)
        
        return final_similarity
    
    def merge_similar_phrases(self, phrases: Dict[str, int], similarity_threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, int]:
        """Merge phrases with similar meanings and add up their frequencies."""
        if not phrases:
            return phrases
            
        # Sort phrases by frequency (descending)
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        
        merged = {}
        used_indices = set()
        
        with tqdm(total=len(sorted_phrases), desc="Merging phrases") as pbar:
            for i, (phrase1, freq1) in enumerate(sorted_phrases):
                if i in used_indices:
                    pbar.update(1)
                    continue
                    
                total_freq = freq1
                best_phrase = phrase1
                merged_phrases = [phrase1]
                
                for j, (phrase2, freq2) in enumerate(sorted_phrases[i+1:], i+1):
                    if j in used_indices:
                        continue
                        
                    similarity = self.similarity_score(phrase1, phrase2)
                    if similarity >= similarity_threshold:
                        total_freq += freq2
                        used_indices.add(j)
                        merged_phrases.append(phrase2)
                        
                        # Choose the best phrase (prefer shorter, more frequent)
                        if len(phrase2) < len(best_phrase) and freq2 >= freq1 * 0.8:
                            best_phrase = phrase2
                        elif freq2 > freq1 * 1.2:
                            best_phrase = phrase2
                
                # Only keep if we have meaningful content
                if len(best_phrase.strip()) >= 3:
                    merged[best_phrase] = total_freq
                
                used_indices.add(i)
                pbar.update(1)
        
        logger.info(f"Merged {len(phrases)} phrases into {len(merged)} phrases")
        return merged
    
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
    
    def process_phrases_parallel(self, phrases: Dict[str, int], sentiment: str) -> Dict[str, int]:
        """Process phrases with parallel filtering."""
        if not phrases:
            return {}
        
        # Prepare data for parallel processing
        phrase_items = list(phrases.items())
        
        def process_phrase_batch(batch):
            results = {}
            for phrase, freq in batch:
                if self.filter_phrase(phrase, sentiment):
                    results[phrase] = freq
            return results
        
        # Split into batches for parallel processing
        batch_size = max(1, len(phrase_items) // (self.num_cores * 4))
        batches = [phrase_items[i:i + batch_size] for i in range(0, len(phrase_items), batch_size)]
        
        filtered_phrases = {}
        
        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            futures = [executor.submit(process_phrase_batch, batch) for batch in batches]
            
            with tqdm(total=len(futures), desc=f"Filtering {sentiment} phrases") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        filtered_phrases.update(batch_results)
                    except Exception as e:
                        logger.error(f"Error in parallel processing: {e}")
                    pbar.update(1)
        
        logger.info(f"Filtered {len(phrases)} phrases to {len(filtered_phrases)} phrases for {sentiment}")
        return filtered_phrases
    
    def translate_phrases_parallel(self, phrases: Dict[str, int]) -> Dict[str, int]:
        """Translate phrases in parallel."""
        if not phrases:
            return {}
        
        phrase_items = list(phrases.items())
        
        def translate_phrase_batch(batch):
            results = {}
            for phrase, freq in batch:
                translated = self.translate_to_english(phrase)
                results[translated] = results.get(translated, 0) + freq
            return results
        
        # Split into batches
        batch_size = max(1, len(phrase_items) // (self.num_cores * 4))
        batches = [phrase_items[i:i + batch_size] for i in range(0, len(phrase_items), batch_size)]
        
        translated_phrases = {}
        
        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            futures = [executor.submit(translate_phrase_batch, batch) for batch in batches]
            
            with tqdm(total=len(futures), desc="Translating phrases") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        for phrase, freq in batch_results.items():
                            translated_phrases[phrase] = translated_phrases.get(phrase, 0) + freq
                    except Exception as e:
                        logger.error(f"Error in parallel translation: {e}")
                    pbar.update(1)
        
        return translated_phrases
    
    def process_all_files(self, similarity_threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Process all sentiment phrase files with enhanced filtering."""
        banks = BANKS
        sentiments = SENTIMENTS
        
        results = {}
        
        print(f"\nProcessing with {self.num_cores} CPU cores...")
        
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
                
                # Translate non-English phrases (parallel)
                print("Translating phrases...")
                translated_phrases = self.translate_phrases_parallel(phrases)
                
                # Apply enhanced filtering (parallel)
                print("Filtering phrases...")
                filtered_phrases = self.process_phrases_parallel(translated_phrases, sentiment)
                
                # Merge similar phrases
                print("Merging similar phrases...")
                merged_phrases = self.merge_similar_phrases(filtered_phrases, similarity_threshold)
                
                results[bank][sentiment] = merged_phrases
                
                time.sleep(API_RETRY_DELAY)
        
        return results
    
    def generate_wordcloud(self, phrases: Dict[str, int], title: str, filename: str):
        """Generate a word cloud from phrases."""
        if not phrases:
            logger.warning(f"No phrases to generate wordcloud for {title}")
            return
        
        wordcloud = WordCloud(
            width=WORDCLOUD_WIDTH,
            height=WORDCLOUD_HEIGHT,
            background_color='white',
            max_words=WORDCLOUD_MAX_WORDS,
            colormap=WORDCLOUD_COLORMAP,
            relative_scaling=WORDCLOUD_RELATIVE_SCALING,
            collocations=False,  # Avoid repeated words
            prefer_horizontal=0.7,  # Mix horizontal and vertical text
            min_font_size=10,
            max_font_size=100
        ).generate_from_frequencies(phrases)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20, pad=20)
        plt.tight_layout()
        
        plt.savefig(filename, dpi=WORDCLOUD_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated wordcloud: {filename}")
    
    def generate_all_wordclouds(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Generate word clouds for all banks and sentiments."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        print("\nGenerating word clouds...")
        
        for bank, sentiments in results.items():
            for sentiment, phrases in sentiments.items():
                if phrases:
                    title = f"{bank.replace('_', ' ').title()} - {sentiment.title()} Phrases (Final)"
                    filename = f"{OUTPUT_DIR}/{bank}_{sentiment}_final_wordcloud.png"
                    self.generate_wordcloud(phrases, title, filename)
    
    def save_processed_results(self, results: Dict[str, Dict[str, Dict[str, int]]], filename: str = "final_processed_phrases.json"):
        """Save processed results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved final processed results to {filename}")
    
    def print_summary(self, results: Dict[str, Dict[str, Dict[str, int]]]):
        """Print a summary of the processing results."""
        print("\n" + "="*60)
        print("FINAL PROCESSING SUMMARY")
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
    """Main function to run the final sentiment processor."""
    parser = argparse.ArgumentParser(description='Final Enhanced Sentiment Processor')
    parser.add_argument('--cores', type=int, default=4, help='Number of CPU cores to use (default: 4)')
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
    
    print("Final Enhanced Sentiment Processor")
    print("="*50)
    print(f"Using {args.cores} CPU cores")
    print(f"Backing up existing wordclouds...")
    
    processor = FinalSentimentProcessor(api_key, num_cores=args.cores)
    
    # Backup existing wordclouds
    backup_dir = processor.backup_existing_wordclouds()
    
    print("Processing sentiment phrases with enhanced filtering...")
    start_time = time.time()
    
    results = processor.process_all_files(similarity_threshold=SIMILARITY_THRESHOLD)
    
    processor.save_processed_results(results)
    
    print("Generating final word clouds...")
    processor.generate_all_wordclouds(results)
    
    processor.print_summary(results)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nFinal processing complete!")
    print(f"Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
    print(f"Backup location: {backup_dir}")
    print(f"Check the '{OUTPUT_DIR}' directory for generated images.")

if __name__ == "__main__":
    main() 