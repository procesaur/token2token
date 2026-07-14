# -*- coding: utf-8 -*-
import platform
import requests
from os import path as px, makedirs
from datasets import load_dataset
from transformers import AutoTokenizer


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


def build_dataset(lang1, lang2, tokenizer1, tokenizer2, datapref=None, column1="src_text", column2="tgt_text"):
    """Download corpora from OpenSubtitles2024.
    :return huggingface dataset
    """
    if column1 == None:
        column1 = "src_text"
    if column2 == None:
        column2 = "tgt_text"

    def preprocess(example):
        return {
            lang1: tokenizer1.tokenize(example[column1]),
            lang2: tokenizer2.tokenize(example[column2])
        }

    if datapref:
        ds = load_dataset(
            datapref,
            split="train",
            streaming=True
        )
        
    else:
        ds = load_dataset(
            "Helsinki-NLP/OpenSubtitles2024",
            split="train",
            trust_remote_code=True,
            data_files=f"dev/{lang1}-{lang2}/{lang1}-{lang2}.parquet",
            streaming=True
        )

    ds = ds.map(preprocess)
    return ds
