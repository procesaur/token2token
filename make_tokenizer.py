# -*- coding: utf-8 -*-
r"""
A command line script for building an extended tokenizer based on an existing tokenizer (HF model) and a custom training dataset with four pruning options to create space for new vocab:
cyr - remove all Cyrillic tokens 
zh - remove all Chinese tokens 
both - remove all Cyrillic and Chinese tokens
all - empty the whole vocab

Usage:
    extended_tokenizer, pruned_tokenizer, new_vocab_map = adapt_tokenizer(
        model="Qwen/Qwen3.5-0.8B",
        dataset="procesaur/sr-tokenizer-test",
        prune_target="cyr",
        n_lines=3000
        )

    python make_tokenizer.py "Qwen/Qwen3.5-0.8B" "procesaur/sr-tokenizer-test" --prune_target cyr --n_lines 3000

Authors:
    Mihailo Škorić (procesaur@gmail.com), based on Taido Purason (taido.purason@ut.ee)
"""

import argparse
from token2token.extend import adapt_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help="identifier of a huggingface model, or a path to dir with tokenizer.json you want to extend")
    parser.add_argument('--no_translit', dest="no_translit", action="store_true",
                        help="Do not use tranliteration for the tokenizer")
    parser.add_argument('--dataset', type=str, required=True,
                        help="data prefix to a custom tokenizer training corpus.") 
    parser.add_argument('--subset', type=str, default=None,
                        help="subset identifier in a huggingface dataset")                                         
    parser.add_argument('--split', type=str, default="train",
                        help="split name for training corpus")
    parser.add_argument('--prune_target', type=str, default="both",
                        help="what vocab from initial tokenizer to remove: cyr, zh, both or all")
    parser.add_argument('--extension_size', type=str, default=None,
                        help="number of tokens to add to the new vocab")
    parser.add_argument('--n_lines', type=int, default=None,
                        help="number of parallel sentences used")
    parser.add_argument('--savedir', type=str, default=None,
                        help="location to store the new tokenizer")
    parser.add_argument('--num_workers', default=16, type=int,
                        help="number of workers used for multiprocessing")

    args = parser.parse_args()

    adapt_tokenizer(**vars(args))

def test():
    extended_save_path, prunned_save_path, vocab_map_save_path = adapt_tokenizer(
        model="Qwen/Qwen3.5-0.8B",
        dataset="procesaur/sr-tokenizer-test",
        prune_target="cyr",
        n_lines=500
    )

if __name__ == "__main__":
    # test()
    main()
    