[![image](https://img.shields.io/pypi/v/token2token.svg)](https://pypi.org/project/token2token/)
[![image](https://img.shields.io/pypi/l/token2token.svg)](https://pypi.org/project/token2token/)
[![image](https://img.shields.io/pypi/pyversions/token2token.svg)](https://pypi.org/project/token2token/)

# token2token

Easy-to-make token mappings using one or two tokenizers and a parallel corpus.
<!--This is the official code accompanying [our LREC 2020 paper](https://arxiv.org/abs/1911.12019).-->

## Example

You want to align French and English on sub-token level.
You need:
- A French (HuggingFace) tokenizer
- An English tokenizer (could be the same one)
- A French-English parallel corpus (if none provided OpenSubtitles2024 from huggingface is used by default)
- This software

For each token in the first tokenizer you will get a list of possible matching tokens from the second tokenizers and a score for each of them.

Alternatively, you can still use the old pipeline and get word mappings based on NLTK or other specialized tokenizer

## Usage

First, install the package using 
```shell script
git clone https://github.com/kakaobrain/word2word
python setup.py install
```
<!--
OR

```shell script
pip install token2token
```
-->


Then, in Python, download the model and retrieve top-5 word translations 
of any given word to the desired language:
```python
from token2token import Token2token
enfr = Token2token.make(lang1="en", lang2="fr", tokenizer1="Qwen/Qwen3.5-0.8B", tokenizer2="Qwen/Qwen3.5-0.8B", n_lines=500000)
print(en2fr("Ġapple"))
# out: {'Ġpomme': 18.72391482536058, 'omm': 4.7151260350878825, 'nÃ©s': 2.887133318202845, 'Ġpommes': 2.8528411761126584, 'po': 2.799092675636191}
```

Alternatively you can still use the old pipeline to get word mappings:
```python
from token2token import Word2word
enfr = Word2word.make(lang1="en", lang2="fr", n_lines=500000)
print(en2fr("apple"))
# out: {'pomme': 18.491287696990998, 'pommiers': 2.913168676725654, 'pommes': 2.8193681613734003, 'empoisonnés': 2.767322352478363, 'pommier': 1.8529305946107455}
```
The old pipeline has been modified :
- to use huggingface datasets for corpora
- to output scores together with words and
- to save in plain, human readable JSON format.

In both cases, the custom lexicon can be loaded from the directory it is stored in
(defaulting to home directory in linux or "C:\word2word" in Windows
```python
from token2token import Token2token
my_en2fr = Token2token.load("en", "fr")
# Loaded token2token custom token mapping from C:\word2word\en-fr.json
```

```python
from token2token import Word2word
my_en2fr = Word2word.load("en", "fr", "data/pubmed.en-fr")
# Loaded token2word custom bilingual lexicon from C:\word2word\en-fr.json
```

## Supported Languages

As already mentioned, when custom dataset is not provided the fallback is [OpenSubtitles2024](https://opus.nlpl.eu/datasets/OpenSubtitles), supporting 94 langugages.


## Methodology

The approach computes top-k word translations based on 
the co-occurrence statistics between cross-lingual word pairs in a parallel corpus.
There is also a correction term that controls for any confounding effect
coming from other source words within the same sentence.
The resulting method is an efficient and scalable approach that allows the
construction of large bilingual dictionaries from any given parallel corpus,
 or a (subword) token alignment bwtween different languages and/or tokenizers. 

For more details, see the Methodology section of [the original paper](https://arxiv.org/abs/1911.12019).


### Multiprocessing

In both the Python interface and the command line interface, 
`make` uses multiprocessing with 8 CPUs by default.
The number of CPU workers can be adjusted by setting 
`num_workers=N` (Python) or `--num_workers N` (command line).

## References


If you use word2word for research, please cite [our paper](https://arxiv.org/abs/1911.12019):
```bibtex
@inproceedings{choe2020word2word,
 author = {Yo Joong Choe and Kyubyong Park and Dongwoo Kim},
 title = {word2word: A Collection of Bilingual Lexicons for 3,564 Language Pairs},
 booktitle = {Proceedings of the 12th International Conference on Language Resources and Evaluation (LREC 2020)},
 year = {2020}
}
```

For token2token add-on citation coming soon.

## Authors
[Mihailo Škorić](https://github.com/procesaur)
based on 
[Kyubyong Park](https://github.com/Kyubyong), 
[Dongwoo Kim](https://github.com/kimdwkimdw), and 
[YJ Choe](https://github.com/yjchoe)

