# -*- coding: utf-8 -*-
import os
from time import time
import re
from typing import List

from token2token.token2token import Token2token
from token2token.utils import build_dataset, get_savedir
from token2token.methods import rerank, get_trans_pmi, get_vocab, update_dicts


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
        tokenizer, name = WordTokenizer(), "wordTokenizer"
    return tokenizer, name


mapping = {
        "Љ": "Lj", "љ": "lj", "Њ": "Nj", "њ": "nj", "Џ": "Dž", "џ": "dž",
        "А": "A", "а": "a", "Б": "B", "б": "b", "В": "V", "в": "v",
        "Г": "G", "г": "g", "Д": "D", "д": "d", "Ђ": "Đ", "ђ": "đ",
        "Е": "E", "е": "e", "Ж": "Ž", "ж": "ž", "З": "Z", "з": "z",
        "И": "I", "и": "i", "Ј": "J", "ј": "j", "К": "K", "к": "k",
        "Л": "L", "л": "l", "М": "M", "м": "m", "Н": "N", "н": "n",
        "О": "O", "о": "o", "П": "P", "п": "p", "Р": "R", "р": "r",
        "С": "S", "с": "s", "Т": "T", "т": "t", "Ћ": "Ć", "ћ": "ć",
        "У": "U", "у": "u", "Ф": "F", "ф": "f", "Х": "H", "х": "h",
        "Ц": "C", "ц": "c", "Ч": "Č", "ч": "č", "Ш": "Š", "ш": "š",
    }

class WordTokenizer:
    def __init__(self, lower: bool = True, trans: bool = True):
        self.lower = lower
        self.trans = trans
        self.word_pattern = re.compile(r'[^\W\d_]+')

    def cyr2lat(self, text: str) -> str:
        return "".join(mapping.get(ch, ch) for ch in text)

    def tokenize(self, text: str) -> List[str]:
        if self.lower:
            text = text.lower()
        if self.trans:
            text = self.cyr2lat(text)
        return self.word_pattern.findall(text)


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
            split: str = "train",
            subset: str = None,
            n_lines: int = 1000000,
            rerank_width: int = 100,
            n_translations: int = 10,
            save_pmi: bool = False,
            savedir: str = None,
            min_frequency: int = 3,
    ):
        """Build a bilingual lexicon using a parallel corpus."""

        print("Step 1. Load tokenizers and build dataset")
        tokenizer1, t1name = load_word_tokenizer(lang1)
        tokenizer2, t2name = load_word_tokenizer(lang2)
        dataset = build_dataset(lang1, lang2, tokenizer1, tokenizer2, datapref, column1, column2, split=split, subset=subset)

        # input savedir if provided, system default otherwise
        if not savedir:
            savedir = get_savedir()

        print("Step 3. Compute vocabularies")
        word2x, x2word, x2cnt, word2y, y2word, y2cnt = get_vocab(dataset.take(n_lines), lang1, lang2, min_frequency=min_frequency)

        x_total_count = sum(x2cnt.values())
        y_total_count = sum(y2cnt.values())
        xfpm = {x2word[x]:round(1000000*y/x_total_count) for x, y in x2cnt.items()}
        yfpm = {y2word[x]:round(1000000*y/y_total_count) for x, y in y2cnt.items()}

        print("Step 4. Update count dictionaries")
        # monolingual and cross-lingual dictionaries
        x2xs, x2ys, seqlens1, seqlens2 = update_dicts(
            dataset.take(n_lines), lang1, lang2, n_lines, save_pmi, x2cnt, word2x, word2y
        )

        t0 = time()
        print("Step 5. Translation using CPE scores")

        x2ys_cpe = rerank(x2xs, x2ys, rerank_width, n_translations)
        print(f"Time taken for step 5: {time() - t0:.2f}s")

        print("Saving...")
        Word2word.save(lang1, lang2, savedir, word2x, word2y, x2word,
                      y2word, x2ys_cpe, xfpm, yfpm, t1name, t2name)

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
            #y2xs_pmi = get_trans_pmi(y2xs, y2cnt, x2cnt, Nxy, Ny, Nx,
            #                         rerank_width, n_translations)

            Word2word.save(lang1, lang2, subdir, word2x, word2y, x2word,
                           y2word, x2ys_pmi, xfpm, yfpm, t1name, t2name)

        print("Done!")
        return cls(lang1, lang2, word2x, y2word, x2ys_cpe)
