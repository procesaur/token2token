# -*- coding: utf-8 -*-
import os
from time import time
from token2token.token2token import Token2token

from token2token.utils import build_dataset, get_savedir
from token2token.methods import rerank, rerank_mp, get_trans_pmi, get_vocab, update_dicts


def load_word_tokenizer(lang):
    if lang == "ko":
        from konlpy.tag import Mecab
        tokenizer, name = Mecab(), "konlpy"
    elif lang == "ja":
        import Mykytea
        opt = "-model jp-0.4.7-1.mod"
        tokenizer, name = Mykytea.Mykytea(opt), "Mykytea-jp-0.4.7-1"
    elif lang == "zh_cn":
        import Mykytea
        opt = "-model ctb-0.4.0-1.mod"
        tokenizer, name = Mykytea.Mykytea(opt), "Mykytea-ctb-0.4.0-1"
    elif lang == "zh_tw":
        import jieba
        tokenizer, name = jieba, "jieba"
    elif lang == "vi":
        from pyvi import ViTokenizer
        tokenizer, name = ViTokenizer, "ViTokenizer"
    elif lang == "th":
        from pythainlp.tokenize import word_tokenize
        tokenizer, name = word_tokenize, "pythainlp"
    elif lang == "ar":
        import pyarabic.araby as araby
        tokenizer, name = araby, "araby"
    else:
        from nltk.tokenize import ToktokTokenizer
        tokenizer, name = ToktokTokenizer(), "nltk"
    return tokenizer, name


class Word2word (Token2token):
    """The word2word class.

    Usage:
        from word2word import Word2word

        # Load a pre-computed bilingual lexicon
        en2fr = Word2word("en", "fr")
        print(en2fr("apple"))
        # out: {'pomme': 0.58, 'pommes':0.3, 'pommier': 0.11, 'tartes': 0.09, 'fleurs':0.01}

        # Build a custom bilingual lexicon
        # (requires a parallel corpus on huggingface)
        my_en2fr = Word2word.make("en", "fr", "Helsinki-NLP/OpenSubtitles2024", column1="src_text", column2="tgt_text")
    """

    @classmethod
    def make(
            cls,
            lang1: str,
            lang2: str,
            datapref: str = None,
            column1: str = None,
            column2: str = None,
            n_lines: int = 1000000,
            cutoff: int = 5000,
            rerank_width: int = 100,
            rerank_impl: str = "multiprocessing",
            n_translations: int = 10,
            save_pmi: bool = False,
            savedir: str = None,
            num_workers: int = 8,
    ):
        """Build a bilingual lexicon using a parallel corpus."""

        print("Step 1. Load tokenizers and build dataset")
        lang1, lang2 = sorted([lang1, lang2])
        tokenizer1, t1name = load_word_tokenizer(lang1)
        tokenizer2, t2name = load_word_tokenizer(lang2)
        dataset = build_dataset(lang1, lang2, tokenizer1, tokenizer2, datapref, column1, column2)

        # input savedir if provided, system default otherwise
        if not savedir:
            savedir = get_savedir()

        print("Step 3. Compute vocabularies")
        # word <-> index

        word2x, x2word, x2cnt = get_vocab(dataset.take(n_lines), lang1)
        word2y, y2word, y2cnt = get_vocab(dataset.take(n_lines), lang2)

        print("Step 4. Update count dictionaries")
        # monolingual and cross-lingual dictionaries
        x2xs, y2ys, x2ys, y2xs, seqlens1, seqlens2 = update_dicts(
            dataset.take(n_lines), lang1, lang2, word2x, word2y, cutoff, n_lines, save_pmi
        )

        t0 = time()
        print("Step 5. Translation using CPE scores")
        if rerank_impl == "simple":
            x2ys_cpe = rerank(x2ys, x2cnt, x2xs, rerank_width, n_translations)
            y2xs_cpe = rerank(y2xs, y2cnt, y2ys, rerank_width, n_translations)
        elif rerank_impl == "multiprocessing":
            x2ys_cpe = rerank_mp(
                x2ys, x2cnt, x2xs, rerank_width, n_translations, num_workers
            )
            y2xs_cpe = rerank_mp(
                y2xs, y2cnt, y2ys, rerank_width, n_translations, num_workers
            )
        else:
            raise ValueError("unrecognized --rerank_impl argument. "
                             "Options: simple, multiprocessing")
        print(f"Time taken for step 5: {time() - t0:.2f}s")

        print("Saving...")
        Word2word.save(lang1, lang2, savedir, word2x, word2y, x2word,
                       x2ys_cpe, y2word, y2xs_cpe, t1name, t2name)

        if save_pmi:
            print("Step 5-1. Translation using PMI scores")
            subdir = os.path.join(savedir, "pmi")
            os.makedirs(subdir, exist_ok=True)
            Nx = sum(seqlens1)
            Ny = sum(seqlens2)
            Nxy = sum([seqlen_x * seqlen_y
                       for seqlen_x, seqlen_y in zip(seqlens1, seqlens2)])

            x2ys_pmi = get_trans_pmi(x2ys, x2cnt, y2cnt, Nxy, Nx, Ny,
                                     rerank_width, n_translations)
            y2xs_pmi = get_trans_pmi(y2xs, y2cnt, x2cnt, Nxy, Ny, Nx,
                                     rerank_width, n_translations)

            Word2word.save(lang1, lang2, subdir, word2x, word2y, x2word,
                           x2ys_pmi, y2word, y2xs_pmi, t1name, t2name)

        print("Done!")
        return cls(lang1, lang2, word2x, y2word, x2ys_cpe)
