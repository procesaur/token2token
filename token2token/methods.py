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

import itertools as it
import numpy as np
import operator
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter, defaultdict
from itertools import chain, product, islice, combinations


def rerank(x2ys, x2cnt, x2xs, width, n_trans):
    """Re-rank word translations by computing CPE scores."""
    x2ys_cpe = dict()
    for x, ys in tqdm(x2ys.items()):
        cntx = x2cnt[x]
        y_scores = []
        for y, cnty in sorted(ys.items(), key=operator.itemgetter(1), reverse=True)[:width]:
            ts = cnty / float(cntx)  # initial translation score
            if x in x2xs:
                for x2, cntx2 in x2xs[x].items():  # collocates
                    p_x_x2 = cntx2 / float(cntx)
                    p_x2_y2 = 0
                    if x2 in x2ys:
                        p_x2_y2 = x2ys[x2].get(y, 0) / float(x2cnt[x2])
                    ts -= (p_x_x2 * p_x2_y2)
            y_scores.append((y, ts))
        # keep top n_trans with scores
        _ys_ = sorted(y_scores, key=lambda y_score: y_score[1], reverse=True)[:n_trans]
        x2ys_cpe[x] = _ys_   # <-- now stores list of (y, score)
    return x2ys_cpe


def _rerank_mp(x_and_ys, shared_inputs):
    """Internal multiprocessing function for `rerank_fast()`."""
    x, ys = x_and_ys
    x2ys, x2cnt, x2xs, width, n_trans = shared_inputs

    sorted_ys = sorted(ys.items(),
                       key=operator.itemgetter(1),
                       reverse=True)[:width]
    if x not in x2xs:
        return x, sorted_ys[:n_trans]

    def _correction(y):
        return sum(
            cntx2 * x2ys[x2][y] / float(x2cnt[x2])
            for x2, cntx2 in x2xs[x].items() if x2 in x2ys and y in x2ys[x2]
        )

    y_scores = [(y, cnty - _correction(y)) for y, cnty in sorted_ys]
    y_scores = sorted(y_scores, key=operator.itemgetter(1), reverse=True)
    reranked_ys = [y for y in y_scores[:n_trans] if y[1]>0]  # keep (y, score) pairs
    return x, reranked_ys


def rerank_mp(x2ys, x2cnt, x2xs, width, n_trans, num_workers):
    """Re-rank word translations by computing CPE scores with multiprocessing."""
    shared_inputs = x2ys, x2cnt, x2xs, width, n_trans
    print(f"Entering multiprocessing with {num_workers} workers..."
          f" (#words={len(x2ys)})")

    with Pool(num_workers) as p:
        # tqdm wraps the zipped iterator so you see progress
        results = p.starmap(
            _rerank_mp,
            tqdm(zip(x2ys.items(), it.repeat(shared_inputs)),
                 total=len(x2ys),
                 desc="Re-ranking")
        )
    x2ys_cpe = dict(results)
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


def get_vocab(dataset, column1, column2, tokenizer1=None, tokenizer2=None):

    if tokenizer1 and tokenizer2:
        idx2cnt = Counter()
        idy2cnt = Counter()

        # Create iterator over streaming dataset
        dataset_iter = iter(dataset)

        pbar = tqdm(desc="Processing Stream", unit=" chunks")
        
        while True:
            chunk = list(islice(dataset_iter, 512))
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
        word2idx, idx2word, idx2cnt = dict(), dict(), dict()
        word2idy, idy2word, idy2cnt = dict(), dict(), dict()
        word2cnt1 = Counter()
        word2cnt2 = Counter()
        for example in tqdm(dataset):
            word2cnt1.update(example[column1])
            word2cnt2.update(example[column2])
        word2cnt1 = word2cnt1.most_common()
        word2cnt2 = word2cnt2.most_common()

        word2cnt1.sort(key=operator.itemgetter(1, 0), reverse=True)
        for idx, (word, cnt) in enumerate(tqdm(word2cnt1)):
            word2idx[word] = idx
            idx2word[idx] = word
            idx2cnt[idx] = cnt

        word2cnt2.sort(key=operator.itemgetter(1, 0), reverse=True)
        for idy, (word, cnt) in enumerate(tqdm(word2cnt2)):
            word2idy[word] = idy
            idy2word[idy] = word
            idy2cnt[idy] = cnt

    return word2idx, idx2word, idx2cnt, word2idy, idy2word, idy2cnt


def update_dicts(dataset, lang1, lang2, n_lines, save_pmi, vocab1=None, vocab2=None):
    """Get monolingual and cross-lingual count dictionaries.

    'cutoff' determines how many collocates are considered in each language.
    """
    xx_counter = Counter()
    xy_counter = Counter()

    seqlens1 = []
    seqlens2 = []

    for ex in tqdm(dataset, total=n_lines):
        if vocab1:
            xs = [vocab1[x] for x in ex[lang1]] 
        else:
            xs = ex[lang1].ids
        if vocab2:
            ys = [vocab2[x] for x in ex[lang2]] 
        else:
            ys = ex[lang2].ids

        if save_pmi:
            seqlens1.append(len(xs))
            seqlens2.append(len(ys))

        xx_counter.update((x1, x2) for x1, x2 in combinations(xs, 2) if x1 != x2)
        xx_counter.update((x2, x1) for x1, x2 in combinations(xs, 2) if x1 != x2)

        # 2. Cross-lingual X-Y co-occurrences
        xy_counter.update(product(xs, ys))

    def counter_to_nested_dict(counter):
        nested = defaultdict(dict)
        for (k1, k2), count in counter.items():
            nested[k1][k2] = count
        return {k: dict(v) for k, v in nested.items()}

    x_x_dict = counter_to_nested_dict(xx_counter)
    x_y_dict = counter_to_nested_dict(xy_counter)

    return x_x_dict, x_y_dict, seqlens1, seqlens2
