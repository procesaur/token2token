# -*- coding: utf-8 -*-

"""
token2token/methods.py: bilingual lexicon extraction methods

Speed comparison using OpenSubtitles.en-eo (64,485 sentences):
    - rerank (single process): 105s
    - rerank_mp (multiprocessing), 8 CPUs: 33s (3.2x faster)
    - rerank_mp (multiprocessing), 16 CPUs: 26s (4.0x faster)
    - rerank_mp (multiprocessing), 32 CPUs: 47s (2.2x faster)

Optimal number of CPUs may differ depending on corpus size.
"""

import numpy as np
import operator
from tqdm import tqdm
from collections import Counter
from itertools import chain, product, islice, combinations
import scipy.sparse as sp


def rerank(P_XX, P_XY, width=50, n_trans=10):
    """Re-ranks candidate translations using fast C++ level sparse matrix math.

    P_XX: Normalized P(X2|X) CSR matrix
    P_XY: Normalized P(Y|X) CSR matrix
    """
    # 1. Prune P_XY to top 'width' candidates per row
    if width is not None:
        P_XY_pruned = prune_top_k_per_row(P_XY, k=width)
    else:
        P_XY_pruned = P_XY

    # 2. Compute Indirect Penalty: P(X2|X) @ P(Y|X2)
    print("Computing matrix dot product (P_XX @ P_XY)...")
    P_indirect = P_XX @ P_XY_pruned

    # 3. Compute final CPE Score Matrix
    CPE_Matrix = P_XY_pruned - P_indirect

    # 4. Extract top n_trans scores per word ID (x)
    x2ys_cpe = {}

    for x in range(CPE_Matrix.shape[0]):
        row = CPE_Matrix.getrow(x)
        if row.nnz == 0:
            continue

        data = row.data
        indices = row.indices

        positive_mask = data > 0
        if not np.any(positive_mask):
            continue

        data = data[positive_mask]
        indices = indices[positive_mask]

        # Efficient partial sort for top n_trans candidates
        if len(data) > n_trans:
            top_idx = np.argpartition(data, -n_trans)[-n_trans:]
            top_idx = top_idx[np.argsort(-data[top_idx])]
        else:
            top_idx = np.argsort(-data)

        # Output tuple list: [(y_id, cpe_score), ...]
        x2ys_cpe[x] = [(int(indices[i]), float(data[i])) for i in top_idx]

    return x2ys_cpe


def get_trans_pmi(x2ys, x2cnt, y2cnt, Nxy, Nx, Ny, width, n_trans):
    """Use pointwise mutual information to compute scores."""
    x2ys_pmi = dict()
    pmi_diff = -np.log2(Nxy) + np.log2(Nx) + np.log2(Ny)
    for x, ys in tqdm(x2ys.items()):
        l_scores = []
        for y, cnt in sorted(ys.items(), key=operator.itemgetter(1),
                             reverse=True)[:width]:
            pmi = np.log2(cnt) - np.log2(x2cnt[x]) - np.log2(y2cnt[y])
            pmi += pmi_diff
            l_scores.append((y, pmi))
        trans = sorted(l_scores, key=operator.itemgetter(1, 0), reverse=True)[:n_trans]
        x2ys_pmi[x] = trans

    return x2ys_pmi


def get_vocab(dataset, column1, column2, tokenizer1=None, tokenizer2=None, min_frequency=None, batch_size=256):

    if tokenizer1 and tokenizer2:
        idx2cnt = Counter()
        idy2cnt = Counter()

        # Create iterator over streaming dataset
        dataset_iter = iter(dataset)

        pbar = tqdm(desc="Processing Stream", unit=" chunks")
        
        while True:
            chunk = list(islice(dataset_iter, batch_size))
            if not chunk:
                break

            # Fast C-level list flattening per chunk
            idx2cnt.update(chain.from_iterable(row[column1].ids for row in chunk))
            idy2cnt.update(chain.from_iterable(row[column2].ids for row in chunk))
            pbar.update(1)

        pbar.close()

        # Extract vocabs directly
        word2idx = tokenizer1.get_vocab()
        idx2word = {v: k for k, v in word2idx.items()}

        word2idy = tokenizer2.get_vocab()
        idy2word = {v: k for k, v in word2idy.items()}

        return word2idx, idx2word, idx2cnt, word2idy, idy2word, idy2cnt

    else:
        word2cnt1 = Counter()
        word2cnt2 = Counter()

        for batch in tqdm(dataset.iter(batch_size=batch_size), desc="Batch Counting"):
            # batch[column1] is a list of lists of tokens
            for doc in batch[column1]:
                word2cnt1.update(doc)
                
            for doc in batch[column2]:
                word2cnt2.update(doc)

        # 2. Filter & Sort efficiently
        # Note: most_common() already sorts by count descending.
        # Sorting by (count, word) is only needed to break count ties deterministically.
        if min_frequency:
            items1 = [
                item for item in word2cnt1.most_common() if item[1] >= min_frequency
            ]
            items2 = [
                item for item in word2cnt2.most_common() if item[1] >= min_frequency
            ]
        else:
            items1 = word2cnt1.most_common()
            items2 = word2cnt2.most_common()

        # Tie-breaking sort using key=lambda vs operator.itemgetter
        items1.sort(key=lambda x: (-x[1], x[0]))
        items2.sort(key=lambda x: (-x[1], x[0]))

        # 3. Fast bulk creation of mappings (1-indexed)
        word2idx = {word: idx for idx, (word, _) in enumerate(items1, 1)}
        idx2word = {idx: word for idx, (word, _) in enumerate(items1, 1)}
        idx2cnt = {idx: cnt for idx, (_, cnt) in enumerate(items1, 1)}

        word2idy = {word: idy for idy, (word, _) in enumerate(items2, 1)}
        idy2word = {idy: word for idy, (word, _) in enumerate(items2, 1)}
        idy2cnt = {idy: cnt for idy, (_, cnt) in enumerate(items2, 1)}

    return word2idx, idx2word, idx2cnt, word2idy, idy2word, idy2cnt


def prune_top_k_per_row(mat, k=50):
        """Keeps only the top 'k' highest probability values per row."""
        mat = mat.tocsr()
        new_data, new_indices, new_indptr = [], [], [0]

        for i in range(mat.shape[0]):
            r_start = mat.indptr[i]
            r_end = mat.indptr[i + 1]

            r_data = mat.data[r_start:r_end]
            r_cols = mat.indices[r_start:r_end]

            if len(r_data) > k:
                top_k_idx = np.argpartition(r_data, -k)[-k:]
                new_data.extend(r_data[top_k_idx])
                new_indices.extend(r_cols[top_k_idx])
                new_indptr.append(new_indptr[-1] + k)
            else:
                new_data.extend(r_data)
                new_indices.extend(r_cols)
                new_indptr.append(new_indptr[-1] + len(r_data))

        return sp.csr_matrix(
            (new_data, new_indices, new_indptr), shape=mat.shape, dtype=mat.dtype
        )


def update_dicts(dataset, lang1, lang2, n_lines, save_pmi, x2cnt, vocab1=None, vocab2=None):
    """Get monolingual and cross-lingual count dictionaries.

    'cutoff' determines how many collocates are considered in each language.
    """
    xx_counter = Counter()
    xy_counter = Counter()

    seqlens1 = []
    seqlens2 = []

    if vocab1 and vocab2:
        get_x = vocab1.get
        get_y = vocab2.get
        shape=(len(vocab1)+1, len(vocab2)+1)
        use_vocab = True
    else:
        shape=vocab1
        use_vocab = False

    xx_update = xx_counter.update
    xy_update = xy_counter.update

    for ex in tqdm(dataset, total=n_lines):
        if use_vocab:
            # Inline lookup is faster than list(filter(None, map(...)))
            xs = [x_id for w in ex[lang1] if (x_id := get_x(w)) is not None]
            ys = [y_id for w in ex[lang2] if (y_id := get_y(w)) is not None]
        else:
            xs = ex[lang1].ids
            ys = ex[lang2].ids

        len_x = len(xs)
        len_y = len(ys)

        if save_pmi:
            seqlens1.append(len_x)
            seqlens2.append(len_y)

        if len_x == 0:
            continue

        # 1. Monolingual X-X Co-occurrences
        if len_x > 1:
            # Pass combinations directly into C-level update
            # Generate forward (x1, x2) and reverse (x2, x1) in two C-speed updates
            combos = [
                (x1, x2) for x1, x2 in combinations(xs, 2) if x1 != x2
            ]  # Single pass allocation
            xx_update(combos)
            xx_update((x2, x1) for x1, x2 in combos)  # Reverse pass over pre-filtered list

        # 2. Cross-lingual X-Y Co-occurrences
        if len_y > 0:
            xy_update(product(xs, ys))

    def counter_to_p_matrix(counter, shape, global_counts, min_count=3, dtype=np.float32):
        """
        Normalizes CSR matrix using exact global word counts (x2cnt) 
        to guarantee identical probability scores as the original code.
        """
        if not counter:
            return sp.csr_matrix(shape, dtype=dtype)
            
        filtered_items = [(k[0], k[1], v) for k, v in counter.items() if v >= min_count]
        if not filtered_items:
            return sp.csr_matrix(shape, dtype=dtype)
            
        n_entries = len(filtered_items)
        
        rows = np.fromiter((item[0] for item in filtered_items), dtype=np.int32, count=n_entries)
        cols = np.fromiter((item[1] for item in filtered_items), dtype=np.int32, count=n_entries)
        data = np.fromiter((item[2] for item in filtered_items), dtype=dtype, count=n_entries)
        
        mat = sp.csr_matrix((data, (rows, cols)), shape=shape, dtype=dtype)
        
        # Use exact total corpus counts for each row ID
        # global_counts should be an array/list where index = x_id, value = x2cnt[x_id]

        # Check if global_counts is a Counter/dict or array/list
        if isinstance(global_counts, dict) or hasattr(global_counts, 'get'):
            row_totals = np.fromiter(
                (global_counts.get(i, 0) for i in range(shape[0])), 
                dtype=dtype, 
                count=shape[0]
            )
        else:
            row_totals = np.array(global_counts, dtype=dtype)
        row_totals[row_totals == 0.0] = 1.0  # Prevent zero division
        
        inv_row_sums = sp.diags(1.0 / row_totals, dtype=dtype)
        return inv_row_sums @ mat


    x_x_dict = counter_to_p_matrix(xx_counter, (shape[0], shape[0]), x2cnt)
    x_y_dict = counter_to_p_matrix(xy_counter, shape, x2cnt)

    return x_x_dict, x_y_dict, seqlens1, seqlens2
