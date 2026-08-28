# -*- coding: utf-8 -*-
from os import makedirs, path as px
from json import dump, load, dumps
from time import time

from token2token.utils import build_dataset, get_savedir, load_hf_fast_tokenizer
from token2token.methods import rerank, get_trans_pmi, get_vocab, update_dicts


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

    def __init__(self, lang1=None, lang2=None, token2x=None, token2y=None, x2token=None, y2token=None, x2ys=None, xfpm=None, yfpm=None, t1name=None, t2name=None, path=None,):
        """Loads this object with a custom-built token mapping.

        savedir is the directory containing {lang1}-{lang2}.pkl files
        built from the make function.
        """

        if all(d is not None for d in [lang1, lang2, token2x, token2y, x2token, y2token, x2ys, xfpm, yfpm]):
            # load a custom-built token2token bilingual tool mapping
            self.lang1, self.lang2, self.token2x, self.token2y, self.x2token  = lang1, lang2, token2x, token2y, x2token, 
            self.y2token, self.x2ys, self.xfpm, self.yfpm, self.t1name, self.t2name = y2token, x2ys, xfpm, yfpm, t1name, t2name
            return  

        if not path:
            if lang1 and lang2:
                savedir = get_savedir()
                path = px.join(savedir, f"{lang1}-{lang2}.json")

            else:
                 raise ValueError("you have to define either correct path or lang1 and lang2.")

        assert px.exists(path), f"processed lexicon file not found at {path}"
        with open(path, "r", encoding="utf-8") as f:
            data = load(f)

        self.lang1 = data["src_lang"]
        self.lang2 = data["tgt_lang"]

        print(f"Loaded token2token custom token mapping from {path}")

        self.token2x = data["src_vocab"]
        self.token2y = data["tgt_vocab"]
        self.y2token = {y:x for x,y in self.token2y.items()}
        self.x2token = {y:x for x,y in self.token2x.items()}

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
            tokens = {}
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
            subset: str = None,
            n_lines: int = 1000000,
            rerank_width: int = 100,
            n_translations: int = 10,
            save_pmi: bool = False,
            savedir: str = None,
            vocab_only: bool = False,
    ):
        """Build a token mapping using a parallel corpus."""

        print("Step 1. Load tokenizers and build dataset")
        
        if isinstance(tokenizer1, str):
            t1name = tokenizer1
            tokenizer1 = load_hf_fast_tokenizer(t1name)
        else:
            try:
                t1name = tokenizer1.pretrained_model_name_or_path
            except:
                t1name = getattr(tokenizer1, "name_or_path", "unknown_model")
            
        if isinstance(tokenizer2, str):
            t2name = tokenizer2
            tokenizer2 = load_hf_fast_tokenizer(t2name)
        else:
            try:
                t2name = tokenizer2.pretrained_model_name_or_path
            except:
                t2name = getattr(tokenizer2, "name_or_path", "unknown_model")

        dataset = build_dataset(lang1, lang2, tokenizer1, tokenizer2, datapref, column1, column2, split=split, subset=subset)

        # input savedir if provided, system default otherwise
        if not savedir:
            savedir = get_savedir()

        print("Step 3. Compute vocabularies")
        token2x, x2token, x2cnt, token2y, y2token, y2cnt = get_vocab(dataset.take(n_lines), lang1, lang2, tokenizer1, tokenizer2)

        x_total_count = sum(x2cnt.values())
        y_total_count = sum(y2cnt.values())
        xfpm = {x2token[x]:round(1000000*y/x_total_count) for x, y in x2cnt.items()}
        yfpm = {y2token[x]:round(1000000*y/y_total_count) for x, y in y2cnt.items()}

        del y2cnt

        if vocab_only:
            return xfpm, yfpm, token2x, token2y

        print("Step 4. Update count dictionaries")
        # monolingual and cross-lingual dictionaries
        x2xs, x2ys, seqlens1, seqlens2 = update_dicts(
            dataset.take(n_lines), lang1, lang2, n_lines, save_pmi, x2cnt, (len(token2x), len(token2y)) 
        )

        if save_pmi:
            print("Step 5-1. Translation using PMI scores")
            subdir = px.join(savedir, "pmi")
            makedirs(subdir, exist_ok=True)
            Nx = sum(seqlens1)
            Ny = sum(seqlens2)
            Nxy = sum([seqlen_x * seqlen_y
                       for seqlen_x, seqlen_y in zip(seqlens1, seqlens2)])

            x2ys = get_trans_pmi(x2ys, x2cnt, y2cnt, Nxy, Nx, Ny,
                                     rerank_width, n_translations)
        else:
            t0 = time()
            print("Step 5. Translation using CPE scores")
            x2ys = rerank(x2xs, x2ys, rerank_width, n_translations)
            print(f"Time taken for step 5: {time() - t0:.2f}s")

        obj = cls(lang1, lang2, token2x, token2y, x2token, y2token, x2ys, xfpm, yfpm, t1name, t2name, savedir) 
        obj.save(savedir)
        return obj
        
    @staticmethod
    def _dump_jsonl(path, translations, id2token_src, id2token_tgt, limit):
        """Helper to write bilingual dictionary JSON with tokens instead of IDs."""
        norm_translations = {}
        for src_id, tgts in translations.items():
            if not tgts:
                continue
            
            norm_translations[id2token_src[int(src_id)]] = [
                {id2token_tgt[int(tgt)]: float(score)}
                for tgt, score in tgts
            ]

        with open(path, "w", encoding="utf-8") as f:
            for src, tgt in norm_translations.items():
                tgt = [d for d in tgt if list(d.values())[0] >= limit]
                tgt = {k: v for d in tgt for k, v in d.items()}
                if tgt:
                    f.write(dumps({"word": src, "translations":tgt}, ensure_ascii=False)+"\n")

    @staticmethod
    def _dump_json(path, src_vocab, tgt_vocab, translations, src_lang, tgt_lang,
            id2token_src, id2token_tgt, xfpm, yfpm, t1name=None, t2name=None):
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
            "src_fpm": xfpm,
            "tgt_fpm": yfpm,    
            "translations": norm_translations
        }
        with open(path, "w", encoding="utf-8") as f:
            dump(data, f, ensure_ascii=False, indent=2)

    def save(self, savedir):
        print("Saving...")
        if not savedir:
            savedir = get_savedir
        makedirs(savedir, exist_ok=True)

        self._dump_json(
            px.join(savedir, f"{self.lang1}-{self.lang2}.json"),
            src_vocab=self.token2x,
            tgt_vocab=self.token2y,
            translations=self.x2ys,
            src_lang=self.lang1,
            tgt_lang=self.lang2,
            id2token_src=self.x2token,
            id2token_tgt=self.y2token,
            xfpm=self.xfpm,
            yfpm=self.yfpm,
            t1name=self.t1name,
            t2name=self.t2name
        )
        print("Done!")

    def save_mapping(self, savedir=None, limit=0.01):
        print("Saving mapping...")
        if not savedir:
            savedir = get_savedir
        makedirs(savedir, exist_ok=True)

        self._dump_jsonl(
            px.join(savedir, f"{self.lang1}-{self.lang2}.jsonl"),
            translations=self.x2ys,
            id2token_src=self.x2token,
            id2token_tgt=self.y2token,
            limit=limit
        )
        print("Done!")
