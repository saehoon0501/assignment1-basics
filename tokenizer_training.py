import json
import os
from cs336_basics import train

def log_training_outputs(vocab, merges, output_prefix):
    """
    Log the vocabulary and merges to files.
    
    Args:
        vocab: Dictionary mapping token IDs to bytes
        merges: List of merge operations (tuples of bytes)
        output_prefix: Prefix for the output files
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Convert vocab to a JSON-serializable format
    vocab_json = {}
    for token_id, token_bytes in vocab.items():
        # Convert bytes to string for JSON serialization
        try:
            vocab_json[str(token_id)] = token_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # For non-UTF8 bytes, use hex representation
            vocab_json[str(token_id)] = token_bytes.hex()
    
    # Save vocabulary
    vocab_file = f"logs/{output_prefix}_vocab.json"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, indent=2, ensure_ascii=False)
    print(f"Vocabulary saved to {vocab_file} ({len(vocab)} tokens)")
    
    # Save merges
    merges_file = f"logs/{output_prefix}_merges.txt"
    with open(merges_file, 'w', encoding='utf-8') as f:
        f.write("# BPE Merge Operations\n")
        f.write("# Format: token1 token2\n")
        f.write(f"# Total merges: {len(merges)}\n\n")
        
        for i, (token1, token2) in enumerate(merges):
            try:
                # Try to decode as UTF-8
                token1_str = token1.decode('utf-8')
                token2_str = token2.decode('utf-8')
                f.write(f"{token1_str} {token2_str}\n")
            except UnicodeDecodeError:
                # Fall back to hex representation for non-UTF8 bytes
                token1_hex = token1.hex()
                token2_hex = token2.hex()
                f.write(f"[hex:{token1_hex}] [hex:{token2_hex}]\n")
    
    print(f"Merges saved to {merges_file} ({len(merges)} operations)")
    
    # Save summary statistics
    stats_file = f"logs/{output_prefix}_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Training Statistics for {output_prefix}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Vocabulary size: {len(vocab)}\n")
        f.write(f"Number of merges: {len(merges)}\n")
        f.write(f"Original bytes (0-255): 256\n")
        f.write(f"Special tokens: {len(vocab) - 256 - len(merges)}\n")
        f.write(f"Learned tokens: {len(merges)}\n")
    
    print(f"Statistics saved to {stats_file}")

if __name__ == "__main__":
    print("Training BPE tokenizers...")
    
    # Train on TinyStories dataset
    print("\nTraining on TinyStoriesV2-GPT4 dataset...")
    tiny_vocab, tiny_merges = train("data/TinyStoriesV2-GPT4-train.txt", 10_000, ['<|endoftext|>'])
    log_training_outputs(tiny_vocab, tiny_merges, "tinystories")
    
    # Train on OpenWebText dataset
    print("\nTraining on OpenWebText dataset...")
    owt_vocab, owt_merges = train("data/owt_train.txt", 32_000, ['<|endoftext|>'])
    log_training_outputs(owt_vocab, owt_merges, "owt")
    
    print("\nAll training outputs have been logged to the 'logs' directory.")
    
    