# -*- coding: utf-8 -*-
import platform
import wget
import requests
import os
import pickle
from os.path import expanduser
from datasets import load_dataset


def get_savedir(savedir=None):
    if savedir:
        os.makedirs(savedir, exist_ok=True)
        return savedir

    pf = platform.system()
    if pf == "Windows":
        savedir = "C:\word2word"
    else:
        homedir = expanduser("~")
        savedir = os.path.join(homedir, ".word2word")

    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
    return savedir


def exists(path):
    r = requests.head(path)
    return r.status_code == requests.codes.ok


def get_download_url(lang1, lang2):
    filepath = os.path.dirname(os.path.abspath(__file__)) + '/supporting_languages.txt'
    for line in open(filepath, 'r'):
        l1, l2 = line.strip().split("-")
        if lang1 == l1 and lang2 == l2:
            return f"https://mk.kakaocdn.net/dn/kakaobrain/word2word/{lang1}-{lang2}.pkl"
    raise Exception(f"Language pair {lang1}-{lang2} is not supported.")


def download_or_load(lang1, lang2, custom_savedir):
    savedir = get_savedir(savedir=custom_savedir)
    fpath = os.path.join(savedir, f"{lang1}-{lang2}.pkl")
    if not os.path.exists(fpath):
        # download from cloud
        url = get_download_url(lang1, lang2)
        if url is None:
            raise ValueError(f"Dataset not found for {lang1}-{lang2}.")

        if not exists(url):
            raise ValueError("Sorry. There seems to be a problem with cloud access.")

        print("Downloading data ...")
        wget.download(url, fpath)
    word2x, y2word, x2ys = pickle.load(open(fpath, 'rb'))
    return word2x, y2word, x2ys


def build_dataset(lang1, lang2, tokenizer1, tokenizer2):
    """Download corpora from OpenSubtitles2018.

    :return huggingface dataset
    """

    def preprocess(example):
        return {
            lang1: tokenizer1.tokenize(example["src_text"]),
            lang2: tokenizer2.tokenize(example["tgt_text"])
        }

    ds = load_dataset(
        "Helsinki-NLP/OpenSubtitles2024",
        split="train",
        trust_remote_code=True,
        data_files=f"dev/{lang1}-{lang2}/{lang1}-{lang2}.parquet",
        streaming=True
    )

    ds = ds.map(preprocess)
    return ds
