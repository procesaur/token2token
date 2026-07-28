# -*- coding: utf-8 -*-
r"""
A command line script for building an extended tokenizer based on an existing tokenizer (HF model) and a custom training dataset with four pruning options to create space for new vocab:
cyr - remove all Cyrillic tokens 
zh - remove all Chinese tokens 
both - remove all Cyrillic and Chinese tokens
all - empty the whole vocab

Usage:
    extended_save_path, prunned_save_path, vocab_map_save_path = adapt_tokenizer(
        model="Qwen/Qwen3.5-0.8B",
        lang = "sr",
        dataset="procesaur/sr-tokenizer-test",
        prune_target="cyr",
        n_lines=10000,
        no_overlap_data = "procesaur/KOMPaS",
        no_overlap_subset="en",
        no_overlap_lines=10000,
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
    parser.add_argument('--lang', type=str, required=True,
                        help="ISO 639-1 code of language. "
                             "See `http://opus.nlpl.eu/OpenSubtitles2018.php`")
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
    # To merge old vocab for the language and new vocab>
    parser.add_argument('--reinitialize_old', dest="reinitialize_old", action="store_true",
                        help="Also change weights of the existing tokens for the new language")
    parser.add_argument('--no_overlap', type=str, default=None,
                        help="identifier of the language you want to avoid overlap with")
    parser.add_argument('--no_overlap_data', type=str, default=None,
                        help="identifier of the dataset you want to use to calculate possible overlap tokens")
    parser.add_argument('--no_overlap_lines', type=int, default=None,
                        help="number of parallel sentences used to caluclate overlap")
    parser.add_argument('--no_overlap_split', type=str, default="train",
                        help="split name for training corpus used to caluclate overlap")
    parser.add_argument('--no_overlap_subset', type=str, default=None,
                        help="subset identifier in a huggingface dataset used to caluclate overlap")
    parser.add_argument('--no_overlap_column1', type=str, default="src_text",
                        help="identifier of the first column with parallel text in a huggingface dataset")
    parser.add_argument('--no_overlap_column2', type=str, default="tgt_text",
                        help="identifier of the second column with parallel text in a huggingface dataset")
    parser.add_argument('--num_workers', default=16, type=int,
                        help="number of workers used for multiprocessing")

    args = parser.parse_args()

    adapt_tokenizer(**vars(args))

def test():
    extended_save_path, prunned_save_path, vocab_map_save_path = adapt_tokenizer(
        model="Qwen/Qwen3.5-0.8B",
        lang = "sr",
        dataset="procesaur/sr-tokenizer-test",
        prune_target="cyr",
        n_lines=10000,
        reinitialize_old=True,
        no_overlap_data = "procesaur/KOMPaS",
        no_overlap_subset="en",
        no_overlap_lines=10000,
    )

if __name__ == "__main__":
    # test()
    main()
    