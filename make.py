# -*- coding: utf-8 -*-
r"""
A command line script for building a token2token bilingual lexicon.

By default, builds from a (downloaded) OpenSubtitles2018 dataset;
 also supports building from a custom parallel corpus.

Usage:
    # Build with OpenSubtitles2018

        my_en2fr = Token2token.make("sr", "hr", "procesaur/gpt2-srlat", "procesaur/gpt2-srlat", "Helsinki-NLP/OpenSubtitles2024", column1="src_text", column2="tgt_text")
    python make.py sr hr
    
    # Build with custom tokenizer
    python make.py sr hr --tokenizer1 "procesaur/gpt2-srlat" --tokenizer2 "procesaur/gpt2-srlat"

    # Build with a custom dataset
    python make.py --lang1 sr --lang2 hr --datapref "Helsinki-NLP/OpenSubtitles2024" --column1 src_text --column2 tgt_text

    # Use the original implementation
    python make.py sr hr --word2word


Authors:
    Mihailo Škorić (procesaur@gmail.com), based on Kyubyong Park (kbpark.linguist@gmail.com), YJ Choe (yjchoe33@gmail.com), Dongwoo Kim (kimdwkimdw@gmail.com)
"""

import argparse

from token2token import Token2token, Word2word


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang1', type=str, required=True,
                        help="ISO 639-1 code of language. "
                             "See `http://opus.nlpl.eu/OpenSubtitles2018.php`")
    parser.add_argument('--lang2', type=str, required=True,
                        help="ISO 639-1 code of language. "
                             "See `http://opus.nlpl.eu/OpenSubtitles2018.php`")
    parser.add_argument('--tokenizer1', type=str, default="openai-community/gpt2",
                        help="identifier of a huggingface model, or a path to dir with tokenizer.json")
    parser.add_argument('--tokenizer2', type=str, required="openai-community/gpt2",
                        help="identifier of a huggingface model, or a path to dir with tokenizer.json")

    parser.add_argument('--datapref', type=str, default=None,
                        help="data prefix to a custom parallel corpus. "
                             "builds a bilingual lexicon using OpenSubtitles2018 "
                             "unless this option is provided.")
    parser.add_argument('--column1', type=str, default="src_text",
                        help="identifier of the first column with parallel text in a huggingface dataset")
    parser.add_argument('--column2', type=str, default="tgt_text",
                        help="identifier of the second column with parallel text in a huggingface dataset")
    parser.add_argument('--split', type=str, default="train",
                        help="split identifier in a huggingface dataset")
    parser.add_argument('--subset', type=str, default=None,
                        help="split identifier in a huggingface dataset")
    parser.add_argument('--word2word', action='store_true',
                            help="Use the original word2word")

    parser.add_argument('--n_lines', type=int, default=100000000,
                        help="number of parallel sentences used")
    parser.add_argument('--cutoff', type=int, default=5000,
                        help="number of words that are used in calculating collocates within each language")
    parser.add_argument('--rerank_width', default=100, type=int,
                        help="maximum number of target-side collocates considered for reranking")
    parser.add_argument('--rerank_impl', default="multiprocessing", type=str,
                        help="choice of reranking implementation: simple, multiprocessing (default)")
    parser.add_argument('--cased', dest="cased", action="store_true",
                        help="Keep the case.")
    parser.add_argument('--n_translations', type=int, default=10,
                        help="number of final word2word translations kept")
    parser.add_argument('--save_cooccurrence', dest="save_cooccurrence", action="store_true",
                        help="Save the cooccurrence results")
    parser.add_argument('--save_pmi', dest="save_pmi", action="store_true",
                        help="Save the pmi results")
    parser.add_argument('--savedir', type=str, default=None,
                        help="location to store bilingual lexicons."
                             "make sure to use this input when loading from "
                             "a custom-bulit lexicon.")
    parser.add_argument('--num_workers', default=16, type=int,
                        help="number of workers used for multiprocessing")
    parser.add_argument('--vocab_only', action='store_true',
                            help="Return only the vocab frequencies, without crosslingual mapping")
    args = parser.parse_args()

    w2w = args.word2word
    del args.word2word
    if w2w:
        Token2token.make(**vars(args))
    else:
        del args.tokenizer1
        del args.tokenizer2
        del args.vocab_only
        Word2word.make(**vars(args))

if __name__ == "__main__":
    main()
