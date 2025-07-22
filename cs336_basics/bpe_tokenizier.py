import os
import regex as re
import mmap
from typing import BinaryIO
from collections import Counter
import multiprocessing

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def flatten_tuple(t):
    if not isinstance(t, tuple):
        yield t
        return
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
        iter = re.finditer(PAT, sc)        
        for match in iter:
            pre_token = match.group()
            b = pre_token.encode("utf-8")
            key = tuple(b[i:i+1] for i in range(len(b)))
            result[key] += 1
        
    return result

def merge(pre_tokens:dict[tuple[bytes], int], vocab:dict[int, bytes], vocab_size:int):        
    result: list[tuple[bytes, bytes]] = []# list of merges, 
    pair_dict = Counter() #maps pair to the total count of the pair, k:pair, v: count    
    
    #initial pair by iterating all the btye sequence in pre_tokens
    for pt, count in pre_tokens.items():
        for i in range(len(pt)-1):
            pair = tuple(pt[i:i+2])
            pair_dict[pair] += count                        

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
        
        if merged_pair is None:
            break # No more pairs to merge

        result.append((b"".join(flatten_tuple(merged_pair[0])), b"".join(flatten_tuple(merged_pair[1]))))
        
        new_merged_token = b"".join(flatten_tuple(merged_pair))
        vocab[len(vocab)] = new_merged_token
        
        # Update pre_tokens with the new merged token        
        new_pre_tokens = {}
        for pre_token, count in pre_tokens.items():
            i = 0
            new_token = pre_token
            while i < len(new_token) - 1:
                if new_token[i:i+2] == merged_pair:
                    new_token = new_token[:i] + (new_merged_token,) + new_token[i+2:]
                i += 1
            if new_token != pre_token:
                new_pre_tokens[pre_token] = new_token
        
        # Apply the updates to pre_tokens and only update the affected pair
        for old_token, new_token in new_pre_tokens.items():
            count = pre_tokens[old_token]
            del pre_tokens[old_token]
            pre_tokens[new_token] += count

            for i in range(len(old_token)-1):
                pair_dict[tuple(old_token[i:i+2])] -= count

                if pair_dict[tuple(old_token[i:i+2])] <= 0:
                    del pair_dict[tuple(old_token[i:i+2])]

            for i in range(len(new_token)-1):
                pair_dict[tuple(new_token[i:i+2])] += count
        
    return result

def train(input_path:str, vocab_size:int, special_token:list[str]):
    #initialize vocab with 0~255 and special_token
    vocab = {i: bytes([i]) for i in range(256)}
    for st in special_token:
        vocab[len(vocab)] = st.encode("utf-8")
    
    #open data and chunk it
    with open(input_path, 'rb') as f:
        chunks = []
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            boundaries = find_chunk_boundaries(mm, special_token[0].encode("utf-8"))

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunks.append(mm[start:end])
        
    #use multiprocess on chunks and aggregate the pre_token counts
    args = [(chunk, special_token) for chunk in chunks]
    with multiprocessing.Pool(20) as pool:
        mp_results = pool.starmap(pre_tokenize, args)

    pre_tokens = Counter()
    for result in mp_results:
        pre_tokens.update(result)    

    merges = merge(pre_tokens, vocab, vocab_size)

    return vocab, merges