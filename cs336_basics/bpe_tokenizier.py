import os
import regex as re
import mmap
from typing import BinaryIO, Iterable, Iterator
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

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens

        # Create reverse lookup: bytes -> int (for fast encoding)
        self.token_to_id = {
            token_bytes: token_id for token_id, token_bytes in vocab.items()
        }

    # Class method that constructs and return a Tokenizer from a serialized vocabulary
    # and list of merges
    # def from_files(
    #     cls, 
    #     vocab_filepath:str, 
    #     merges_filepath:str, 
    #     special_tokens: list[str] | None =None
    # ):

    def _tokenize_text(self, text: str) -> list[tuple[bytes, ...]]:
        """
        Tokenize text preserving order (for encoding), unlike pre_tokenize which counts.
        Returns a list of token tuples in original order.
        Should preserve special_tokens
        """
        if not self.special_tokens:
            special_tokens = []
        else:
            special_tokens = self.special_tokens
            
        # Sort special tokens by length (descending) to handle overlapping tokens correctly
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in sorted_tokens]
        # Use capturing groups to preserve special tokens in split result
        pattern = f"({('|'.join(escaped_tokens))})" if escaped_tokens else None
        split = re.split(pattern, text) if pattern else [text]
        
        tokens_in_order = []
        for sc in split:
            if not sc:  # Skip empty strings
                continue
                
            # Check if this segment is a special token
            if sc in special_tokens:
                # Special tokens are added directly as single tokens
                b = sc.encode("utf-8")
                key = tuple(b[i:i+1] for i in range(len(b)))
                tokens_in_order.append(key)
            else:
                # Regular text segments are processed with PAT regex
                iter = re.finditer(PAT, sc)
                for match in iter:
                    pre_token = match.group()
                    b = pre_token.encode("utf-8")
                    key = tuple(b[i:i+1] for i in range(len(b)))
                    tokens_in_order.append(key)
        
        return tokens_in_order

    # Encode an input text into a sequence of token IDs.
    def encode(self, text: str) -> list[int]:
        # Get tokens in order (not counts)
        tokens_in_order = self._tokenize_text(text)

        # Apply the merges for every preToken
        merged_tokens = []
        for pt in tokens_in_order:
            # MUST not merge any special tokens
            if self.special_tokens is not None:
                # Check if this token corresponds to a special token
                pt_str = b"".join(pt).decode("utf-8")
                if pt_str in self.special_tokens:
                    merged_tokens.append(pt)
                    continue

            while len(pt) >= 2:
                # Find the best pair to merge
                best_pair_idx = -1
                best_pair_rank = float("inf")

                for i in range(len(pt) - 1):
                    pair = (pt[i], pt[i + 1])
                    rank = self.merges.get(pair)
                    if rank is not None and rank < best_pair_rank:
                        best_pair_rank = rank
                        best_pair_idx = i

                if best_pair_idx == -1:
                    break  # No more merges possible

                # Merge the best pair
                i = best_pair_idx
                merged_bytes = pt[i] + pt[i + 1]
                pt = pt[:i] + (merged_bytes,) + pt[i + 2 :]

            merged_tokens.append(pt)

        # convert all the bytes into token ID
        result = []
        for mt in merged_tokens:
            bytes_token = b"".join(mt)
            
            if bytes_token in self.token_to_id:
                # Token exists in vocab
                token_id = self.token_to_id[bytes_token]
                result.append(token_id)
            else:
                # FALLBACK: Split into individual bytes (which MUST exist in vocab)
                for byte_piece in mt:
                    if byte_piece in self.token_to_id:
                        result.append(self.token_to_id[byte_piece])
                    else:
                        raise KeyError(f"Individual byte {byte_piece} not in vocab - this shouldn't happen!")
        
        return result


    # Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
    # This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            tokens_in_order = self._tokenize_text(line)

            # Apply the merges and yield token ID
            for pt in tokens_in_order:
                if self.special_tokens is not None:
                    # Check if this token corresponds to a special token
                    pt_str = b"".join(pt).decode("utf-8")
                    if pt_str in self.special_tokens:
                        bytes_token = b"".join(pt)
                        if bytes_token in self.token_to_id:
                            yield self.token_to_id[bytes_token]
                        else:
                            for byte_piece in pt:
                                if byte_piece in self.token_to_id:
                                    yield (self.token_to_id[byte_piece])
                                else:
                                    raise KeyError(
                                        f"Individual byte {byte_piece} not in vocab - this shouldn't happen!"
                                    )
                        continue

                while len(pt) >= 2:
                    # Find the best pair to merge
                    best_pair_idx = -1
                    best_pair_rank = float("inf")

                    for i in range(len(pt) - 1):
                        pair = (pt[i], pt[i + 1])
                        rank = self.merges.get(pair)
                        if rank is not None and rank < best_pair_rank:
                            best_pair_rank = rank
                            best_pair_idx = i

                    if best_pair_idx == -1:
                        break  # No more merges possible

                    # Merge the best pair
                    i = best_pair_idx
                    merged_bytes = pt[i] + pt[i + 1]
                    pt = pt[:i] + (merged_bytes,) + pt[i + 2 :]

                bytes_token = b"".join(pt)
                if bytes_token in self.token_to_id:
                    yield self.token_to_id[bytes_token]
                else:
                    for byte_piece in pt:
                        if byte_piece in self.token_to_id:
                            yield (self.token_to_id[byte_piece])
                        else:
                            raise KeyError(
                                f"Individual byte {byte_piece} not in vocab - this shouldn't happen!"
                            )

    # Decode a sequence of token IDs into text.
    def decode(self, ids: list[int]) -> str:
        # Collect all bytes first, then decode the entire sequence
        result_bytes = b""
        for token_id in ids:
            result_bytes += self.vocab[token_id]
        
        # Decode the entire byte sequence as UTF-8
        # Use 'replace' to handle any invalid UTF-8 sequences gracefully
        return result_bytes.decode('utf-8', errors='replace')
