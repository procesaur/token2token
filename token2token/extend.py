from token2token.utils import ds_iterator, j_dump, get_savedir, add_translit_normilizer
from .tokenizer_extension.pruning import prune_tokenizer
from .tokenizer_extension.train_vocab_extension import train_vocab_extension
from .tokenizer_extension.extension import extend_tokenizer
import shutil
from os import path as px
from token2token import Token2token


def extract_old_vocab(xfpm, yfpm, token2x, token2y, new_vocab_map):
    results = {}
    for token, fpm in xfpm.items():
        if token not in new_vocab_map:
            yf = yfpm.get(token, 0)
            if fpm > yf*3 and yf < 1000:
                results[token] = token2x[token]
    return results


def adapt_tokenizer(
        model,
        lang: str,
        dataset: str = None,
        subset: str = None,
        split: str = "train",
        prune_target: str = "both",
        extension_size: int = None,
        no_translit: bool = False,
        savedir: str = None,
        n_lines: int = None,
        reinitialize_old: bool = False,
        no_overlap: str = "en",
        no_overlap_data: str = None,
        no_overlap_split: str = "train",
        no_overlap_subset: str = None,
        no_overlap_lines: str = None,
        no_overlap_column1: str = None,
        no_overlap_column2: str = None,
        num_workers: int = 16,
        **kwargs
):
    if not savedir:  
        savedir = get_savedir()
        
    prunned_save_path = px.join(savedir, "my-pruned-tokenizer")
    extended_save_path = px.join(savedir, "my-tokenizer")
    extend_tokenizer_jpath = px.join(extended_save_path, "tokenizer.json")
    vocab_map_save_path = px.join(extended_save_path, "new_vocab_map.json")

    if dataset:
        pruned_tokenizer, n_pruned, prunned_map = prune_tokenizer(model, prune_target, no_translit)
        pruned_tokenizer.save_pretrained(prunned_save_path)

        if extension_size:
            extension_size = min(extension_size, n_pruned)
        else:
            extension_size = n_pruned
        dataset = ds_iterator(dataset, split, subset, limit=n_lines)
        
        extension_tokens = train_vocab_extension(
            tokenizer=pruned_tokenizer,
            corpus=dataset,
            extension_size=extension_size
        )

        extended_tokenizer, new_vocab_map = extend_tokenizer(
            pruned_tokenizer,
            new_vocab=extension_tokens["vocab"],
            new_merges=extension_tokens["merges"],
            prunned_ids=list(prunned_map.values()) 
        )

        if not no_translit:
            extended_tokenizer = add_translit_normilizer(extended_tokenizer)

        shutil.copytree(prunned_save_path, extended_save_path, dirs_exist_ok=True)
        extended_tokenizer.save(extend_tokenizer_jpath)
        j_dump(new_vocab_map, vocab_map_save_path)

        if reinitialize_old:
            xfpm, yfpm, token2x, token2y = Token2token.make(
                lang,
                no_overlap,
                extended_tokenizer,
                extended_tokenizer,
                datapref=no_overlap_data,
                split=no_overlap_split,
                subset=no_overlap_subset,
                column1=no_overlap_column1,
                column2=no_overlap_column2,
                num_workers=num_workers,
                savedir=savedir,
                n_lines=no_overlap_lines,
                vocab_only=True
                )
            additional_mapping = extract_old_vocab(xfpm, yfpm, token2x, token2y, new_vocab_map)
            new_vocab_map.update(additional_mapping)
            del xfpm, yfpm, token2x, token2y
            vocab_map_save_path = px.join(extended_save_path, "extended_new_vocab_map.json")
            j_dump(new_vocab_map, vocab_map_save_path)

        print(f"Tokenizer prunned and extended. Pruned tokenizer saved to {prunned_save_path}, extended tokenizer saved to {extended_save_path}. New vocab mapping saved to {vocab_map_save_path}")

    return extended_save_path, prunned_save_path, vocab_map_save_path
