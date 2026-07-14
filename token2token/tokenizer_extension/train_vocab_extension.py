from collections import defaultdict, Counter
from typing import Iterable, Optional
from tqdm import tqdm
from heapq import heappush, heappop


def fast_group_tokens_and_frequencies(corpus, tokenizer, batch_size=10000):
    """
    Bypasses slow Python loops by feeding batches directly to HF's Rust engine,
    utilizing multi-threading to compute word-token frequencies.
    """
    split_freqs = Counter()
    backend_tokenizer = tokenizer._tokenizer
    batch = []
    
    for text in tqdm(corpus, desc="Batch pre-tokenizing (Rust-accelerated)"):
        if backend_tokenizer.normalizer is not None:
            text = backend_tokenizer.normalizer.normalize_str(text)
        
        batch.append(text)
        
        if len(batch) >= batch_size:
            encodings = backend_tokenizer.encode_batch(batch)
            for encoding in encodings:
                words_dict = {}
                for token, word_id in zip(encoding.tokens, encoding.word_ids):
                    if word_id is None:
                        continue
                    if word_id not in words_dict:
                        words_dict[word_id] = []
                    words_dict[word_id].append(token)
                
                for word_tokens in words_dict.values():
                    split_freqs[tuple(word_tokens)] += 1
            batch = []

    if batch:
        encodings = backend_tokenizer.encode_batch(batch)
        for encoding in encodings:
            words_dict = {}
            for token, word_id in zip(encoding.tokens, encoding.word_ids):
                if word_id is None:
                    continue
                if word_id not in words_dict:
                    words_dict[word_id] = []
                words_dict[word_id].append(token)
            
            for word_tokens in words_dict.values():
                split_freqs[tuple(word_tokens)] += 1

    return split_freqs

def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    where_to_update = defaultdict(set)

    for word, split in splits.items():
        if len(split) < 2:
            continue
        freq = word_freqs[word]
        for pair in zip(split[:-1], split[1:]):
            pair_freqs[pair] += freq
            where_to_update[pair].add(word)
    return pair_freqs, where_to_update


def merge_pair(a, b, splits, word_freqs, pair_freqs, queue, where_to_update):
    updated_pairs = defaultdict(int)
    target_pair = (a, b)
    words_to_modify = list(where_to_update[target_pair])

    for word in words_to_modify:
        freq = word_freqs[word]
        split = splits[word]
        if len(split) < 2:
            continue

        new_split = []
        i = 0
        n = len(split)
        while i < n:
            if i < n - 1 and split[i] == a and split[i + 1] == b:
                new_split.append(a + b)
                i += 2
            else:
                new_split.append(split[i])
                i += 1
        
        new_split = tuple(new_split)
        
        prev_pairs = list(zip(split[:-1], split[1:]))
        new_pairs = list(zip(new_split[:-1], new_split[1:]))
        
        prev_set = set(prev_pairs)
        new_set = set(new_pairs)

        for pair in new_set:
            if pair not in prev_set:
                where_to_update[pair].add(word)
        for pair in prev_set:
            if pair not in new_set:
                where_to_update[pair].discard(word)

        for pair in prev_pairs:
            updated_pairs[pair] -= freq
        for pair in new_pairs:
            updated_pairs[pair] += freq
            
        splits[word] = new_split

    for pair, change in updated_pairs.items():
        if change == 0:
            continue
        new_freq = pair_freqs.get(pair, 0) + change
        pair_freqs[pair] = new_freq
        if new_freq > 0:
            heappush(queue, (-new_freq, pair))

    if target_pair in pair_freqs:
        del pair_freqs[target_pair]
    if target_pair in where_to_update:
        del where_to_update[target_pair]
        
    return splits


def train_vocab_extension(
        tokenizer,
        corpus: Iterable[str],
        extension_size: int,
        max_token_length: Optional[int] = None,
        batch_size: int = 10000,
) -> dict:
    """
    :param tokenizer: The tokenizer to continually train
    :param corpus: Training corpus
    :param extension_size: The Number of tokens to add to the vocabulary
    :param max_token_length: maximum length of a token created
    :return: Dictionary with keys: 'vocab', 'merges', 'pair_freqs', 'word_freqs'
    """
    split_freqs = defaultdict(int)

    check_token = lambda a, b: True
    
    split_freqs = fast_group_tokens_and_frequencies(
            corpus=corpus, 
            tokenizer=tokenizer, 
            batch_size=batch_size
        )

    splits = {"".join(split): split for split in split_freqs}
    word_freqs = {"".join(split): freq for split, freq in split_freqs.items()}
    pair_freqs, where_to_update = compute_pair_freqs(splits, word_freqs)

    pair_queue = []
    for pair, freq in pair_freqs.items():
        heappush(pair_queue, (-freq, pair))

    vocab_size = extension_size
    vocab = {}
    merges = []

    with tqdm(total=vocab_size, desc="training") as pbar:
        while len(vocab) < vocab_size:
            max_freq, best_pair = heappop(pair_queue)
            max_freq = -max_freq  # Convert back to positive

            # Skip stale entries
            if best_pair is None or pair_freqs.get(best_pair, None) != max_freq:
                continue

            new_token = "".join(best_pair)
            if (max_token_length is not None and len(new_token) > max_token_length) or not check_token(*best_pair):
                del pair_freqs[best_pair]
                del where_to_update[best_pair]
                continue

            splits = merge_pair(*best_pair, splits, word_freqs, pair_freqs, pair_queue, where_to_update)
            merges.append(best_pair)
            if new_token not in vocab:
                vocab[new_token] = len(vocab)

            if len(vocab) % 100 == 0 or len(vocab) == vocab_size:
                pbar.n = len(vocab)
                pbar.refresh()

    return {"vocab": vocab, "merges": merges, "pair_freqs": pair_freqs, "word_freqs": word_freqs}
