[![image](https://img.shields.io/pypi/v/token2token.svg)](https://pypi.org/project/token2token/)
[![image](https://img.shields.io/pypi/l/token2token.svg)](https://pypi.org/project/token2token/)
[![image](https://img.shields.io/pypi/pyversions/token2token.svg)](https://pypi.org/project/token2token/)

# token2token

Easy-to-make sub-token mappings using parallel corpora and automated embedding reinitialization for Large Language Models (LLMs).
<!--This is the official code accompanying [our LREC 2020 paper](https://arxiv.org/abs/1911.12019).-->

## Key Features

- **Sub-Token Alignment (`Token2token`)**: Map sub-token vocabularies across different HuggingFace tokenizers (or languages) using parallel text corpora.
- **Classic Word Alignment (`Word2word`)**: Generate word-level cross-lingual dictionaries using NLTK or custom tokenizers.
- **Model Weight Reinitialization**: Transfer target sub-token representations by computing mean embeddings from mapped source sub-tokens across model weight matrices (inputs, outputs, and biases).

## Example

You want to align French and English on sub-token level.
You need:
- A French (HuggingFace) tokenizer
- An English tokenizer (could be the same one)
- A French-English parallel corpus (if none provided OpenSubtitles2024 from huggingface is used by default)
- This software

For each token in the first tokenizer you will get a list of possible matching tokens from the second tokenizers and a score for each of them.

Alternatively, you can still use the old pipeline and get word mappings based on NLTK or other specialized tokenizer

## Installation

Install the package directly from PyPI or build from source:

```bash
pip install token2token
```

OR

```bash
git clone [https://github.com/procesaur/token2token.git](https://github.com/procesaur/token2token.git)
cd token2token
python setup.py install
```

## 1. Token Alignment & Mapping Generation

### A. Sub-token Level Mapping (`Token2token`)
Align sub-tokens using HuggingFace tokenizers (e.g., aligning target sub-tokens to English source sub-tokens on a parallel corpus like OpenSubtitles):

```python
from token2token import Token2token

# Generates top-k mapping from English to French tokens for a specific model
enfr = Token2token.make(
    lang1="en", 
    lang2="fr", 
    tokenizer1="Qwen/Qwen3.5-0.8B", 
    tokenizer2="Qwen/Qwen3.5-0.8B", 
    n_lines=500000
)

# Returns dictionary of mapped source tokens and co-occurrence scores
print(enfr("Ġapple"))
# Output: {'Ġpomme': 18.723, 'omm': 4.715, 'Ġpommes': 2.852}
```

### B. Word Level Mapping (`Word2word`)
For traditional word-level alignment across languages:

```python
from token2token import Word2word

enfr = Word2word.make(lang1="en", lang2="fr", n_lines=500000)
print(enfr("apple"))
# Output: {'pomme': 18.491, 'pommiers': 2.913, 'pommes': 2.819}
```

The old pipeline has been modified :
- to use huggingface datasets for corpora
- to output scores together with words and
- to save in plain, human readable JSON format.

In both cases, the custom lexicon can be loaded from the directory it is stored in
(defaulting to home directory in linux or "C:\word2word" in Windows


### C. Loading Saved Mappings
Saved lexicons are written as human-readable JSON files (`id_mapping.json`) in `C:\word2word` (Windows) or `~/word2word` (Linux):

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

## 2. Model Weight Reinitialization (`transform_weights.py`)

Once an `id_mapping.json` file is generated, you can initialize or adapt sub-token embeddings in a target language or model by averaging source token vectors.

### Example Script Usage

```python
from token2token.extend import adapt_tokenizer

extended_save_path, prunned_save_path, vocab_map_save_path = adapt_tokenizer(
    model="Qwen/Qwen3.5-0.8B",
    dataset="procesaur/sr-tokenizer-test",
    prune_target="cyr",
    n_lines=500
)

from token2token import reinitialize_weights

    reinitialize_weights(
        lang1="sr",
        lang2="ru",
        model="Qwen/Qwen3.5-0.8B",
        extended_tokenizer_path=extended_save_path,
        pruned_tokenizer_path=prunned_save_path,
        new_vocab_map_path=vocab_map_save_path,
        reinitialize_old=True,
        num_workers=8,
        no_overlap_data = "procesaur/KOPaKS",
        no_overlap_lines=10000,
        no_overlap_subset = "en"
        datapref = "procesaur/KOPaKS",
        split = "dev",
        subset = "ru",
        n_lines = 20000,
        savedir = "c:/word2word/test"
    )
```

## Methodology

`token2token` calculates top-k word and token translations based on co-occurrence statistics between cross-lingual pairs in parallel corpora. It incorporates a correction term to counteract confounding context words within the same sentence.

When reinitializing model parameters, target token vectors are set to the vector mean (or median) of their mapped source sub-tokens across:
- **Input Embeddings** (`model.get_input_embeddings()`)
- **Output LM Head Weights** (`model.get_output_embeddings()`)
- **Output Biases** (if present)

Reads are performed off a frozen copy of the original weight tensors to prevent cascaded/dependency update corruption.

---

## Multiprocessing

Dataset processing and token alignment utilize multiprocessing (defaulting to 8 workers):

```python
enfr = Token2token.make(lang1="en", lang2="fr", num_workers=16)
```

## References


If you use word2word for research, please cite:
```bibtex
@inproceedings{choe2020word2word,
 author = {Yo Joong Choe and Kyubyong Park and Dongwoo Kim},
 title = {word2word: A Collection of Bilingual Lexicons for 3,564 Language Pairs},
 booktitle = {Proceedings of the 12th International Conference on Language Resources and Evaluation (LREC 2020)},
 year = {2020}
}
```

For token2token and weight initialization add-ons citation coming soon.

## Authors & Contributors
[Mihailo Škorić](https://github.com/procesaur)
based on 
[Kyubyong Park](https://github.com/Kyubyong), 
[Dongwoo Kim](https://github.com/kimdwkimdw), 
[YJ Choe](https://github.com/yjchoe), and 
[Taido Purason ](taido.purason@ut.ee)

