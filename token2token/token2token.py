# -*- coding: utf-8 -*-
import os
from json import dump, load
from time import time

from token2token.utils import build_dataset, get_savedir, load_hf_tokenizer
from token2token.methods import rerank, rerank_mp, get_trans_pmi, get_vocab, update_dicts


class Token2token:
    """The token2token class.

    Usage:
        from token2token import Token2token

        # Load a pre-computed token mapping from default path
        en2fr = Token2token("en", "fr")
        print(en2fr("Ġapple"))
        # out: {'Ġpomme': 0.58, 'Ġpommes':0.3, 'Ġpomm': 0.11, 'Ġtart': 0.09}

        # Build a custom token mapping
        # (requires two aligned files, e.g., my_corpus.en, my_corpus.fr)
        my_en2fr = Token2token.make("en", "fr", "myfirsttokenizer", "mysecondtokenizer", datapref="my_corpus_id_on_hf", column1="text_en_or_smthng", column2="text_fr_or_smthng")
        my_en2fr = Token2token.make("sr", "hr", "procesaur/gpt2-srlat", "procesaur/gpt2-srlat", "Helsinki-NLP/OpenSubtitles2024", column1="src_text", column2="tgt_text")
    """

    def __init__(self, lang1=None, lang2=None, token2x=None, token2y=None, x2token=None, y2token=None, x2ys=None, path=None,):
        """Loads this object with a custom-built token mapping.

        savedir is the directory containing {lang1}-{lang2}.pkl files
        built from the make function.
        """

        if all(d is not None for d in [lang1, lang2, token2x, token2y, x2token, y2token, x2ys]):
            # load a custom-built token2token bilingual tool mapping
            self.lang1, self.lang2, self.token2x, self.token2y, self.x2token, self.y2token, self.x2ys = lang1, lang2, token2x, token2y, x2token, y2token, x2ys
            return  

        if not path:
            if lang1 and lang2:
                savedir = get_savedir()
                path = os.path.join(savedir, f"{lang1}-{lang2}.json")

            else:
                 raise ValueError("you have to define either correct path or lang1 and lang2.")

        assert os.path.exists(path), f"processed lexicon file not found at {path}"
        with open(path, "r", encoding="utf-8") as f:
            data = load(f)

        self.lang1 = data["src_lang"]
        self.lang2 = data["tgt_lang"]

        print(f"Loaded token2token custom token mapping from {path}")

        self.token2x = data["src_vocab"]
        self.token2y = data["tgt_vocab"]
        self.y2token = {y:x for x,y in self.token2y.items()}

        # Rebuild translations into list of (target, score) tuples
        x2ys = {}
        for src, entries in data["translations"].items():
            l = []
            for entry in entries:
                key = next(iter(entry))
                l.append((self.token2y[key], entry[key]))

            x2ys[self.token2x[src]] = l
        self.x2ys = x2ys

    def __call__(self, query, n_best=5):
        """Retrieve top-k token translations for the query token."""
        try:
            x = self.token2x[query]
            ys = self.x2ys[x]
            tokens = {self.y2token[y[0]] : y[1] for y in ys[:n_best]}
        except KeyError:
            raise KeyError(
                f"query token {query} not found in the token mapping."
            )
        return tokens

    def __len__(self):
        """Return the number of source tokens for which translation exists."""
        return len(self.x2ys)

    def compute_summary(self):
        """Compute basic summaries for the token mapping."""
        n_unique_ys = len(set([y for ys in self.x2ys.values() for y in ys]))
        n_ys = [len(ys) for ys in self.x2ys.values()]
        self.summary = {
            "n_valid_tokens": len(self),
            "n_valid_targets": n_unique_ys,
            "n_total_tokens": len(self.token2x),
            "n_total_targets": len(self.y2token),
            "n_translations_per_token": sum(n_ys) / len(n_ys),
            "n_sentences": None,  # original file required
        }
        return self.summary

    @classmethod
    def make(
            cls,
            lang1: str,
            lang2: str,
            tokenizer1: str,
            tokenizer2: str,
            datapref: str = None,
            column1: str = None,
            column2: str = None,
            split: str = "train",
            n_lines: int = 1000000,
            cutoff: int = 5000,
            rerank_width: int = 100,
            rerank_impl: str = "multiprocessing",
            n_translations: int = 10,
            save_pmi: bool = False,
            savedir: str = None,
            num_workers: int = 8,
    ):
        """Build a token mapping using a parallel corpus."""

        print("Step 1. Load tokenizers and build dataset")
        
        if isinstance(tokenizer1, str):
            t1name = tokenizer1
            tokenizer1 = load_hf_tokenizer(t1name)
        else:
            try:
                t1name = tokenizer1.pretrained_model_name_or_path
            except:
                t1name = getattr(tokenizer1, "name_or_path", "unknown_model")
            
        if isinstance(tokenizer2, str):
            t2name = tokenizer2
            tokenizer2 = load_hf_tokenizer(t2name)
        else:
            try:
                t2name = tokenizer2.pretrained_model_name_or_path
            except:
                t2name = getattr(tokenizer2, "name_or_path", "unknown_model")

        dataset = build_dataset(lang1, lang2, tokenizer1, tokenizer2, datapref, column1, column2, split=split)

        # input savedir if provided, system default otherwise
        if not savedir:
            savedir = get_savedir()

        print("Step 3. Compute vocabularies")
        # token <-> index

        token2x, x2token, x2cnt = get_vocab(dataset.take(n_lines), lang1, tokenizer1)
        token2y, y2token, y2cnt = get_vocab(dataset.take(n_lines), lang2, tokenizer2)

        print("Step 4. Update count dictionaries")
        # monolingual and cross-lingual dictionaries
        x2xs, y2ys, x2ys, y2xs, seqlens1, seqlens2 = update_dicts(
            dataset.take(n_lines), lang1, lang2, token2x, token2y, cutoff, n_lines, save_pmi
        )

        t0 = time()
        print("Step 5. Translation using CPE scores")
        if rerank_impl == "simple":
            x2ys_cpe = rerank(x2ys, x2cnt, x2xs, rerank_width, n_translations)
            # y2xs_cpe = rerank(y2xs, y2cnt, y2ys, rerank_width, n_translations)
        elif rerank_impl == "multiprocessing":
            x2ys_cpe = rerank_mp(
                x2ys, x2cnt, x2xs, rerank_width, n_translations, num_workers
            )
            # y2xs_cpe = rerank_mp(
            #     y2xs, y2cnt, y2ys, rerank_width, n_translations, num_workers
            # )
        else:
            raise ValueError("unrecognized --rerank_impl argument. "
                             "Options: simple, multiprocessing")
        print(f"Time taken for step 5: {time() - t0:.2f}s")

        print("Saving...")
        Token2token.save(lang1, lang2, savedir, token2x, token2y, x2token,
                       x2ys_cpe, y2token, t1name, t2name)

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

            Token2token.save(lang1, lang2, subdir, token2x, token2y, x2token,
                           x2ys_pmi, y2token, y2xs_pmi, t1name, t2name)

        print("Done!")
        return cls(lang1, lang2, token2x, token2y, x2token, y2token, x2ys_cpe)

    @staticmethod
    def save(lang1, lang2, savedir, token2x, token2y, x2token, x2ys, y2token, t1name, t2name):

        def _dump_json(path, src_vocab, tgt_vocab, translations, src_lang, tgt_lang,
                    id2token_src, id2token_tgt, t1name=t1name, t2name=t2name):
            """Helper to write bilingual dictionary JSON with tokens instead of IDs."""
            norm_translations = {}
            for src_id, tgts in translations.items():
                if not tgts:
                    norm_translations[id2token_src[int(src_id)]] = []
                    continue
                
                norm_translations[id2token_src[int(src_id)]] = [
                    {id2token_tgt[int(tgt)]: float(score)}
                    for tgt, score in tgts
                ]

            data = {
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src_tokenizer": t1name,
                "tgt_tokenizer": t2name,
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "translations": norm_translations
            }
            with open(path, "w", encoding="utf-8") as f:
                dump(data, f, ensure_ascii=False, indent=2)

        # lang1 → lang2
        _dump_json(
            os.path.join(savedir, f"{lang1}-{lang2}.json"),
            src_vocab=token2x,
            tgt_vocab=token2y,
            translations=x2ys,
            src_lang=lang1,
            tgt_lang=lang2,
            id2token_src=x2token,
            id2token_tgt=y2token,
            t1name=t1name,
            t2name=t2name
        )
