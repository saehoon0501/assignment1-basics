import os
import regex as re
import mmap
from typing import BinaryIO
from collections import Counter
import multiprocessing

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def flatten_tuple(t):
     for item in t:
        if isinstance(item, tuple):
            yield from flatten_tuple(item)
        else:
            yield item

def find_chunk_boundaries(file:BinaryIO, split_special_token:bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    num_chunks = 10
    chunk_size = file_size // num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
    
def pre_tokenize(content:BinaryIO, special_tokens:list[str]) -> dict[tuple[bytes], int]:
    escaped_tokens = [re.escape(token) for token in special_tokens]
    split = re.split("|".join(escaped_tokens), content.decode("utf-8"))
    
    result = Counter()
    for sc in split:
        print(sc)
        iter = re.finditer(PAT, sc)        
        for match in iter:
            pre_token = match.group()
            print(f"text: {pre_token}")

            b = pre_token.encode("utf-8")
            print(f"binary: {[hex(byte) for byte in b]}")

            key = tuple(hex(byte) for byte in b)
            result[key] += 1
        
    for key, value in result.items():
        print(f"{key}::: {value}")
    return result

def merge(pre_tokens:dict[tuple[bytes], int], vocab:dict[int, bytes], vocab_size:int):        
    pair_dict = Counter() #maps pair to the total count of the pair, k:pair, v: count
    pair_tkn_dict = {} #maps which preTokens a pair is from, k: pair, v: preToken
    
    #initial pair by iterating all the btye sequence in pre_tokens
    for pt, count in pre_tokens.items():
        for i in range(len(pt)-1):
            pair = tuple(pt[i:i+2])
            pair_dict[pair] += count
            #memorize all the pretoken that contains the pair
            if pair not in pair_tkn_dict:                
                pair_tkn_dict[pair] = [pt]
            elif pt not in pair_tkn_dict[pair]:
                pair_tkn_dict[pair].append(pt)

    #Until satisfy the size merge
    while vocab_size - len(vocab) > 0:
        #pick most frequent pair
        merged_pair = None
        max_count = 0
        for pair, count in pair_dict.items():
            if count > max_count:
                merged_pair = pair
                max_count = count
            #if tie, pick lexicographically greater pair
            elif count == max_count and tuple(flatten_tuple(pair)) > tuple(flatten_tuple(merged_pair)):
                merged_pair = pair

        #update any pair that's related to merged pair
        for pre_token in pair_tkn_dict[merged_pair]:
            for i in range(len(pre_token)):
                if tuple(pre_token[i:i+2]) == merged_pair:
                    new_token = pre_token[:i] + (merged_pair,) + pre_token[i+2:]
                    #2 updated elements, the count and preTokens
                    if i > 0:
                        pair_dict[pre_token[i-1:i+1]] -= pre_tokens[tuple(flatten_tuple(pre_token))]
                        if pair_dict[pre_token[i-1:i+1]] == 0:
                            pair_dict.pop(pre_token[i-1:i+1])
                        pair_dict[new_token[i-1:i+1]] += pre_tokens[tuple(flatten_tuple(pre_token))]
                        if new_token[i-1:i+1] not in pair_tkn_dict:
                            pair_tkn_dict[new_token[i-1:i+1]] = [new_token]
                        else:
                            pair_tkn_dict[new_token[i-1:i+1]].append(new_token)

                    if i < len(pre_token) - 2:
                        pair_dict[pre_token[i+1:i+3]] -= pre_tokens[tuple(flatten_tuple(pre_token))]
                        if pair_dict[pre_token[i+1:i+3]] == 0:
                            pair_dict.pop(pre_token[i+1:i+3])
                        pair_dict[new_token[i:i+2]] += pre_tokens[tuple(flatten_tuple(pre_token))]
                        if new_token[i:i+2] not in pair_tkn_dict:
                            pair_tkn_dict[new_token[i:i+2]] = [new_token]
                        else:
                            pair_tkn_dict[new_token[i:i+2]].append(new_token)
        pair_dict.pop(merged_pair)            
        pair_tkn_dict.pop(merged_pair)
        vocab[len(vocab)] = merged_pair        
            

def train(input_path:str, vocab_size:int, speical_token:list[str]):
    #initialize vocab with 0~255 and special_token
    vocab = {i:hex(i) for i in range(256)}
    for st in speical_token:
        vocab[len(vocab.keys())-1] = st
    
    #open data and chunk it
    with open(input_path, 'rb') as f:
        chunks = []
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            boundaries = find_chunk_boundaries(mm, b"<|endoftext|>")

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunks.append(mm[start:end])
        
    #use multiprocess on chunks and aggregate the pre_token counts
    args = [(chunk, speical_token) for chunk in chunks]
    with multiprocessing.Pool(5) as pool:
        mp_results = pool.starmap(pre_tokenize, args)

    pre_tokens = Counter()
    for result in mp_results:
        pre_tokens.update(result)
    
    for key, value in pre_tokens.items():
        print(f"{key}::: {value}")

    merge(pre_tokens, vocab, vocab_size)

if __name__ == "__main__":
    train('/Users/sehoonbyun/Documents/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt', 263, '<|endoftext|>')