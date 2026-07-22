# -*- coding: utf-8 -*-
import platform
import requests
from os import path as px, makedirs
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple
from json import loads, dump, load
from tokenizers.normalizers import Sequence, Replace


def load_hf_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    return tokenizer


def get_savedir(savedir=None):
    if savedir:
        makedirs(savedir, exist_ok=True)
        return savedir

    pf = platform.system()
    if pf == "Windows":
        savedir = "C:/word2word"
    else:
        homedir = px.expanduser("~")
        savedir = px.join(homedir, ".word2word")

    if not px.exists(savedir):
        makedirs(savedir, exist_ok=True)
    return savedir


def exists(path):
    r = requests.head(path)
    return r.status_code == requests.codes.ok


def build_dataset(lang1, lang2, tokenizer1, tokenizer2, datapref=None, column1="src_text", column2="tgt_text", split="train"):
    """Download corpora from OpenSubtitles2024.
    :return huggingface dataset
    """

    reverse = False
    datasetlang1, datasetllang2 = sorted([lang1, lang2])

    if column1 == None:
        column1 = "src_text"
    if column2 == None:
        column2 = "tgt_text"

    def preprocess(example, reverse):
        if reverse:
            return {
                lang1: tokenizer1.tokenize(example[column2]),
                lang2: tokenizer2.tokenize(example[column1])
            }
        else:
            return {
                lang1: tokenizer1.tokenize(example[column1]),
                lang2: tokenizer2.tokenize(example[column2])
            }

    if datapref:
        ds = load_dataset(
            datapref,
            split=split,
            streaming=True
        )

    else:
        if datasetlang1 != lang1 and not datapref:
            reverse = True
        ds = load_dataset(
            "Helsinki-NLP/OpenSubtitles2024",
            split="train",
            trust_remote_code=True,
            data_files=f"dev/{datasetlang1}-{datasetllang2}/{datasetlang1}-{datasetllang2}.parquet",
            streaming=True
        )

    ds = ds.map(lambda x: preprocess(x, reverse=reverse))
    return ds


def ds_iterator(id, split="train", streaming=True, limit=None, fn=None):
    dataset = load_dataset(id, split=split, streaming=streaming)
    if limit:
        dataset = dataset.take(limit)
    for example in dataset:
        if fn:
            yield fn(example["text"])
        else:
            yield example["text"]


def get_vocab_and_merges(tokenizer) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    cfg = loads(tokenizer._tokenizer.to_str())
    merges = [tuple(x) if isinstance(x, list) else tuple(x.split(" ")) for x in cfg["model"]["merges"]]
    vocab = cfg["model"]["vocab"]
    return vocab, merges

def j_dump(obj, path):
    with open(path, "w", encoding="utf-8") as jf:
        dump(obj, jf, ensure_ascii=False)

def j_read(path):
    with open(path, "r", encoding="utf-8") as jf:
        obj = load(jf)
    return obj

cyrillic_to_latin = {
    # 2-character mappings
    "Љ": "Lj", "љ": "lj",
    "Њ": "Nj", "њ": "nj",
    "Џ": "Dž", "џ": "dž",
    # 1-character mappings
    "А": "A", "а": "a",
    "Б": "B", "б": "b",
    "В": "V", "в": "v",
    "Г": "G", "г": "g",
    "Д": "D", "д": "d",
    "Ђ": "Đ", "ђ": "đ",
    "Е": "E", "е": "e",
    "Ж": "Ž", "ж": "ž",
    "З": "Z", "з": "z",
    "И": "I", "и": "i",
    "Ј": "J", "ј": "j",
    "К": "K", "к": "k",
    "Л": "L", "л": "l",
    "М": "M", "м": "m",
    "Н": "N", "н": "n",
    "О": "O", "о": "o",
    "П": "P", "п": "p",
    "Р": "R", "р": "r",
    "С": "S", "с": "s",
    "Т": "T", "т": "t",
    "Ћ": "Ć", "ћ": "ć",
    "У": "U", "у": "u",
    "Ф": "F", "ф": "f",
    "Х": "H", "х": "h",
    "Ц": "C", "ц": "c",
    "Ч": "Č", "ч": "č",
    "Ш": "Š", "ш": "š"
}

def add_translit_normilizer(tokenizer):
    translit_normalizers = [
        Replace(cyr, lat) for cyr, lat in cyrillic_to_latin.items()
    ]

    backend_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
    existing_normalizer = backend_tokenizer.normalizer

    if existing_normalizer is not None:
        new_normalizer = Sequence(
            [Sequence(translit_normalizers), existing_normalizer]
        )
    else:
        new_normalizer = Sequence(translit_normalizers)

    backend_tokenizer.normalizer = new_normalizer
    return tokenizer
