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
from itertools import chain, product


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


def get_vocab(dataset, column, tokenizer=None):
    word2idx, idx2word, idx2cnt = dict(), dict(), dict()
    X = [ex[column] for ex in dataset]
    word2cnt = Counter(list(chain.from_iterable(X))).most_common()
    if not tokenizer:
        word2cnt.sort(key=operator.itemgetter(1, 0), reverse=True)
        for idx, (word, cnt) in enumerate(tqdm(word2cnt)):
            word2idx[word] = idx
            idx2word[idx] = word
            idx2cnt[idx] = cnt
    else:
        vocab = tokenizer.get_vocab()
        word2idx = dict(sorted(vocab.items(), key=lambda item: item[1])) 
        idx2word = {v:k for k,v in word2idx.items()}
        idx2cnt = {word2idx[word]:count for word, count in word2cnt}

    return word2idx, idx2word, idx2cnt


def update_dicts(dataset, lang1, lang2, vocab1, vocab2, cutoff, n_lines, save_pmi):
    """Get monolingual and cross-lingual count dictionaries.

    'cutoff' determines how many collocates are considered in each language.
    """

    def u2_iter(t1, t2, same_ignore=False, cut_t2=None):
        for _ in product(t1, t2):
            if (not same_ignore or _[0] != _[1]) and (not cut_t2 or _[1] < cut_t2):
                yield _

    def build_ddi():
        return defaultdict(lambda: defaultdict(int))

    x_x_dict = build_ddi()
    y_y_dict = build_ddi()
    x_y_dict = build_ddi()
    y_x_dict = build_ddi()
    seqlens1 = []
    seqlens2 = []

    for ex in tqdm(dataset, total=n_lines):

        if save_pmi:
            seqlens1.append(len(ex[lang1]))
            seqlens2.append(len(ex[lang2]))

        xs = [vocab1[wx] for wx in ex[lang1] if wx in vocab1]
        ys = [vocab2[wy] for wy in ex[lang2] if wy in vocab2]

        for xx1, xx2 in u2_iter(xs, xs, same_ignore=True, cut_t2=cutoff):
            x_x_dict[xx1][xx2] += 1
        for yy1, yy2 in u2_iter(ys, ys, same_ignore=True, cut_t2=cutoff):
            y_y_dict[yy1][yy2] += 1
        for xx, yy in u2_iter(xs, ys, same_ignore=False):
            x_y_dict[xx][yy] += 1
            y_x_dict[yy][xx] = x_y_dict[xx][yy]

    # convert to ordinary dicts for pickling
    def ddi2dict(ddi):
        return {k: dict(v) for k, v in ddi.items()}

    return tuple(
        list(ddi2dict(ddi) for ddi in [x_x_dict, y_y_dict, x_y_dict, y_x_dict])
        + [seqlens1, seqlens2]
    )
