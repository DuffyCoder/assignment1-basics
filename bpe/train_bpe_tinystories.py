#!/usr/bin/env python3
"""
BPE Training Script for TinyStories Dataset

This script trains a byte-level BPE tokenizer on the TinyStories dataset
with the following specifications:
- Maximum vocabulary size: 10,000
- Special token: <|endoftext|>
- Performance requirements: ≤30 minutes, ≤30GB RAM

The script provides comprehensive analysis including:
- Training time and memory usage
- Performance profiling to identify bottlenecks
- Vocabulary analysis including longest token
- Serialization of results to disk
"""

import os
import sys
import time
import json
import pickle
import cProfile
import pstats
import psutil
import tracemalloc
from pathlib import Path
from typing import Tuple, Dict, List

# Add the current directory to Python path to import tests modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Import the BPE components directly
    from tests.custom.bpe_counter import read_tokens, BPECounter

    def run_train_bpe(input_path, vocab_size, special_tokens):
        """Local implementation of run_train_bpe to avoid import issues"""
        tokens = read_tokens(input_path, special_tokens)
        bpe_counter = BPECounter(tokens, vocab_size, special_tokens)
        bpe_counter.merge()
        return bpe_counter.vocab, bpe_counter.merges

except ImportError as e:
    print(f"Error importing BPE components: {e}")
    print("Please ensure you're running this script from the assignment1-basics directory")
    print("and that all required dependencies are installed.")
    sys.exit(1)


class MemoryMonitor:
    """Monitor memory usage during training"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.start_memory = 0
    
    def start(self):
        """Start monitoring memory"""
        tracemalloc.start()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        print(f"Initial memory usage: {self.start_memory:.2f} MB")
    
    def update_peak(self):
        """Update peak memory usage"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        return current_memory
    
    def get_stats(self):
        """Get memory statistics"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'start_memory_mb': self.start_memory,
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'tracemalloc_current_mb': current / 1024 / 1024,
            'tracemalloc_peak_mb': peak / 1024 / 1024
        }


def save_tokenizer_results(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], 
                          output_dir: str = "bpe_results") -> Tuple[str, str]:
    """
    Save vocabulary and merges to disk
    
    Args:
        vocab: Vocabulary mapping from token ID to bytes
        merges: List of BPE merge rules
        output_dir: Directory to save results
    
    Returns:
        Tuple of (vocab_path, merges_path)
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save vocabulary as JSON (convert bytes to list of ints for serialization)
    vocab_path = os.path.join(output_dir, "vocab.json")
    vocab_serializable = {str(k): list(v) for k, v in vocab.items()}
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_serializable, f, indent=2, ensure_ascii=False)
    
    # Save merges as pickle (preserves bytes type)
    merges_path = os.path.join(output_dir, "merges.pkl")
    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)
    
    print(f"Vocabulary saved to: {vocab_path}")
    print(f"Merges saved to: {merges_path}")
    
    return vocab_path, merges_path


def analyze_vocabulary(vocab: Dict[int, bytes]) -> Dict:
    """
    Analyze the trained vocabulary
    
    Args:
        vocab: Vocabulary mapping from token ID to bytes
    
    Returns:
        Dictionary with analysis results
    """
    # Find longest token
    longest_token = max(vocab.values(), key=len)
    longest_token_id = [k for k, v in vocab.items() if v == longest_token][0]
    
    # Analyze token lengths
    token_lengths = [len(token) for token in vocab.values()]
    avg_length = sum(token_lengths) / len(token_lengths)
    
    # Try to decode longest token for readability analysis
    try:
        longest_token_str = longest_token.decode('utf-8', errors='replace')
        is_readable = longest_token_str.isprintable()
    except:
        longest_token_str = str(longest_token)
        is_readable = False
    
    analysis = {
        'vocab_size': len(vocab),
        'longest_token': longest_token,
        'longest_token_id': longest_token_id,
        'longest_token_length': len(longest_token),
        'longest_token_str': longest_token_str,
        'longest_token_readable': is_readable,
        'avg_token_length': avg_length,
        'min_token_length': min(token_lengths),
        'max_token_length': max(token_lengths)
    }
    
    return analysis


def profile_bpe_training(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple:
    """
    Train BPE tokenizer with comprehensive profiling
    
    Args:
        input_path: Path to training data
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens
    
    Returns:
        Tuple of (vocab, merges, training_time, memory_stats, profile_stats)
    """
    print("="*60)
    print("STARTING BPE TRAINING ON TINYSTORIES DATASET")
    print("="*60)
    print(f"Input file: {input_path}")
    print(f"Target vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print()
    
    # Initialize monitoring
    memory_monitor = MemoryMonitor()
    memory_monitor.start()
    
    # Setup profiler
    profiler = cProfile.Profile()
    
    # Start training
    print("Starting BPE training...")
    start_time = time.time()
    
    profiler.enable()
    try:
        vocab, merges = run_train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        profiler.disable()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Get memory statistics
    memory_stats = memory_monitor.get_stats()
    
    # Process profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    print(f"\nTraining completed successfully!")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Peak memory usage: {memory_stats['peak_memory_mb']:.2f} MB ({memory_stats['peak_memory_mb']/1024:.2f} GB)")
    
    return vocab, merges, training_time, memory_stats, stats


def main():
    """Main function to run BPE training and analysis"""

    # Configuration
    input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    output_dir = "bpe_results"

    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running in test mode with smaller dataset...")
        input_path = "data/test_tinystories.txt"
        vocab_size = 500  # Smaller vocab for testing

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        print("\nPlease download the TinyStories dataset first:")
        print("mkdir -p data")
        print("cd data")
        print("wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt")
        print("cd ..")
        print("\nAlternatively, you can use the validation file for testing:")
        print("wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt")

        # Check if validation file exists as fallback
        validation_path = "data/TinyStoriesV2-GPT4-valid.txt"
        if os.path.exists(validation_path):
            print(f"\nFound validation file: {validation_path}")
            response = input("Would you like to use the validation file instead? (y/n): ")
            if response.lower() == 'y':
                input_path = validation_path
                print(f"Using validation file: {input_path}")
            else:
                sys.exit(1)
        else:
            sys.exit(1)
    
    try:
        # Train BPE tokenizer with profiling
        vocab, merges, training_time, memory_stats, profile_stats = profile_bpe_training(
            input_path, vocab_size, special_tokens
        )
        
        # Analyze vocabulary
        print("\n" + "="*60)
        print("VOCABULARY ANALYSIS")
        print("="*60)
        
        vocab_analysis = analyze_vocabulary(vocab)
        
        print(f"Vocabulary size: {vocab_analysis['vocab_size']}")
        print(f"Longest token: {vocab_analysis['longest_token']}")
        print(f"Longest token ID: {vocab_analysis['longest_token_id']}")
        print(f"Longest token length: {vocab_analysis['longest_token_length']} bytes")
        print(f"Longest token (as string): '{vocab_analysis['longest_token_str']}'")
        print(f"Is longest token readable: {vocab_analysis['longest_token_readable']}")
        print(f"Average token length: {vocab_analysis['avg_token_length']:.2f} bytes")
        
        # Analyze if longest token makes sense
        print(f"\nLongest token analysis:")
        if vocab_analysis['longest_token_readable']:
            print(f"✓ The longest token '{vocab_analysis['longest_token_str']}' appears to be readable text")
            print("✓ This makes sense as BPE should merge frequently occurring character sequences")
        else:
            print(f"⚠ The longest token contains non-printable characters")
            print("This might indicate special characters or encoding artifacts")
        
        # Save results
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        save_tokenizer_results(vocab, merges, output_dir)
        
        # Performance analysis
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Memory usage:")
        print(f"  - Peak memory: {memory_stats['peak_memory_mb']:.2f} MB ({memory_stats['peak_memory_mb']/1024:.2f} GB)")
        print(f"  - Start memory: {memory_stats['start_memory_mb']:.2f} MB")
        print(f"  - Memory increase: {memory_stats['peak_memory_mb'] - memory_stats['start_memory_mb']:.2f} MB")
        
        # Check if requirements are met
        time_ok = training_time <= 30 * 60  # 30 minutes
        memory_ok = memory_stats['peak_memory_mb'] <= 30 * 1024  # 30 GB
        
        print(f"\nRequirement compliance:")
        print(f"  - Time ≤ 30 minutes: {'✓' if time_ok else '✗'} ({training_time/60:.2f} min)")
        print(f"  - Memory ≤ 30 GB: {'✓' if memory_ok else '✗'} ({memory_stats['peak_memory_mb']/1024:.2f} GB)")
        
        # Most time-consuming operations
        print(f"\nTop 10 most time-consuming operations:")
        profile_stats.print_stats(10)
        
        # Summary for task questions
        print("\n" + "="*60)
        print("TASK ANSWERS SUMMARY")
        print("="*60)
        
        print(f"Q: How many hours and memory did training take?")
        print(f"A: Training took {training_time/3600:.4f} hours ({training_time:.2f} seconds)")
        print(f"   Peak memory usage was {memory_stats['peak_memory_mb']:.2f} MB ({memory_stats['peak_memory_mb']/1024:.2f} GB)")
        
        print(f"\nQ: What is the longest token in the vocabulary? Does it make sense?")
        print(f"A: The longest token is '{vocab_analysis['longest_token_str']}' ({vocab_analysis['longest_token_length']} bytes)")
        print(f"   This {'makes sense' if vocab_analysis['longest_token_readable'] else 'may not make complete sense'} as it represents")
        print(f"   {'a common character sequence that BPE identified' if vocab_analysis['longest_token_readable'] else 'potentially special characters or encoding artifacts'}")
        
        print(f"\nQ: What part of the tokenizer training process takes the most time?")
        print(f"A: Based on profiling, the most time-consuming operations are:")
        
        # Get top 3 functions by cumulative time
        stats_list = []
        profile_stats.print_stats(0)  # This populates the stats
        for func, (_, _, _, ct, _) in profile_stats.stats.items():
            stats_list.append((ct, func))
        stats_list.sort(reverse=True)
        
        for i, (cumtime, func) in enumerate(stats_list[:3]):
            _, _, func_name = func
            print(f"   {i+1}. {func_name} ({cumtime:.3f}s cumulative)")
        
        print(f"\nAll results have been saved to the '{output_dir}' directory.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
