# -*- coding: utf-8 -*-
r"""
A command line script for building a token id mapping required for the weights transfer.

Usage:
        reinitialize_weights(
            lang1="sr",
            lang2="ru",
            model="Qwen/Qwen3.5-0.8B",
            extended_tokenizer_path="C:/word2word/cyr/my-tokenizer",
            pruned_tokenizer_path="C:/word2word/cyr/my-pruned-tokenizer",
            new_vocab_map_path="C:/word2word/cyr/my-tokenizer/new_vocab_map.json",
            reinitialize_old=True,
            num_workers=8,
            no_overlap_data = "procesaur/ParalelniSrEn",
            no_overlap_lines=10000,
            datapref = "procesaur/KOPaKS.ru",
            split = "dev",
            subset = "ru",
            n_lines = 20000
        )

python transform_weights.py \
     sr \
     ru \
     "Qwen/Qwen3.5-0.8B" \
    --extended_tokenizer_path "C:/word2word/cyr/my-tokenizer" \
    --pruned_tokenizer_path "C:/word2word/cyr/my-pruned-tokenizer" \
    --new_vocab_map_path "C:/word2word/cyr/my-tokenizer/new_vocab_map.json" \
    --reinitialize_old \
    --num_workers 8 \
    --no_overlap_data "procesaur/ParalelniSrEn" \
    --no_overlap_lines 10000 \
    --datapref "procesaur/KOPaKS.ru" \
    --split dev \
    --subset ru \
    --n_lines 20000

Authors:
    Mihailo Škorić (procesaur@gmail.com)
"""

import argparse
from token2token.weights import reinitialize_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang1', type=str, required=True,
                        help="ISO 639-1 code of language. "
                             "See `http://opus.nlpl.eu/OpenSubtitles2018.php`")
    parser.add_argument('--lang2', type=str, required=True,
                        help="ISO 639-1 code of language. "
                             "See `http://opus.nlpl.eu/OpenSubtitles2018.php`")
    parser.add_argument('--model', type=str, required=True,
                        help="identifier of a huggingface model, or a path to dir with tokenizer.json you want to extend")
    parser.add_argument('--datapref', type=str, default=None,
                        help="data prefix to a custom parallel corpus. "
                             "builds a bilingual lexicon using OpenSubtitles2018 "
                             "unless this option is provided.")                         
    parser.add_argument('--split', type=str, default="train",
                        help="split name for training corpus")
    parser.add_argument('--subset', type=str, default=None,
                        help="subset identifier in a huggingface dataset")
    parser.add_argument('--column1', type=str, default="src_text",
                        help="identifier of the first column with parallel text in a huggingface dataset")
    parser.add_argument('--column2', type=str, default="tgt_text",
                        help="identifier of the second column with parallel text in a huggingface dataset")
    parser.add_argument('--n_lines', type=int, default=None,
                        help="number of parallel sentences used")
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
    parser.add_argument('--savedir', type=str, default=None,
                        help="location to store the new tokenizer")
    parser.add_argument('--num_workers', default=16, type=int,
                        help="number of workers used for multiprocessing")

    args = parser.parse_args()

    reinitialize_weights(**vars(args))


def test(): 
    reinitialize_weights(
        lang1="sr",
        lang2="ru",
        model="Qwen/Qwen3.5-0.8B",
        extended_tokenizer_path="C:/word2word/cyr/my-tokenizer",
        pruned_tokenizer_path="C:/word2word/cyr/my-pruned-tokenizer",
        new_vocab_map_path="C:/word2word/cyr/my-tokenizer/new_vocab_map.json",
        reinitialize_old=True,
        num_workers=8,
        no_overlap_data = "procesaur/ParalelniSrEn",
        no_overlap_lines=10000,
        datapref = "procesaur/KOPaKS.ru",
        split = "dev",
        subset = "ru",
        n_lines = 20000,
        savedir = "c:/word2word/test"
    )

if __name__ == "__main__":
    test()
    # main()