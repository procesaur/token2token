from token2token import Token2token
from transformers import AutoTokenizer
from token2token.utils import j_read, get_savedir, j_dump
from os import path as px


def id_mapping(tokenizer, extended_tokenizer, t2t, no_overlap):
    return {}


def id_mapping_mean(pruned_tokenizer, extended_tokenizer, new_vocab_map):
    new_vocab_map = {extended_tokenizer.decode([y]): y for x, y in new_vocab_map.items()}
    strings, keys = zip(*new_vocab_map.items())
    batch_encodings = pruned_tokenizer(list(strings), add_special_tokens=False)
    return dict(zip(keys, batch_encodings["input_ids"]))


def extract_old_vocab(xfpm, yfpm, token2x, token2y, new_vocab_map):
    results = {}
    for token, fpm in xfpm.items():
        if token not in new_vocab_map:
            yf = yfpm.get(token, 0)
            if fpm > yf*3 and yf < 1000:
                results[token] = token2x[token]
    return results

def extract_mapping(t2t, new_vocab_map):
    id_map = {}
    for id in new_vocab_map.values():
        if id in t2t.x2ys:
            try:
                nid = t2t.x2ys[id][0][0]
                id_map[id] = [nid]
            except:
                pass
    return id_map

def extract_weights_at_idx(idx):
    return []

def reinitialize_weights(
        lang1: str,
        lang2: str,
        model,
        extended_tokenizer_path: str,
        pruned_tokenizer_path: str,
        new_vocab_map_path: str,
        datapref: str = None,
        split: str = "train",
        column1: str = None,
        column2: str = None,
        n_lines=10000000,
        reinitialize_old: bool = False,
        no_overlap: str = "en",
        no_overlap_data: str = None,
        no_overlap_lines: str = None,
        num_workers: int = 16,
        savedir: str = None,
        **kwargs
):

    if not savedir:  
        savedir = get_savedir()

    original_tokenizer = AutoTokenizer.from_pretrained(model)
    map_save_path = px.join(extended_tokenizer_path, "id_mapping.json")
    extended_tokenizer = AutoTokenizer.from_pretrained(extended_tokenizer_path)
    pruned_tokenizer = AutoTokenizer.from_pretrained(pruned_tokenizer_path)
    new_vocab_map = j_read(new_vocab_map_path)

    if lang1 == lang2:
        id_map = id_mapping_mean(pruned_tokenizer, extended_tokenizer, new_vocab_map)

    else:
        if reinitialize_old:
            xfpm, yfpm, token2x, token2y = Token2token.make(
                lang1,
                no_overlap,
                extended_tokenizer,
                extended_tokenizer,
                datapref=no_overlap_data,
                column1=column1,
                column2=column2,
                num_workers=num_workers,
                savedir=savedir,
                n_lines=no_overlap_lines,
                vocab_only=True
                )
            additional_mapping = extract_old_vocab(xfpm, yfpm, token2x, token2y, new_vocab_map)
            new_vocab_map.update(additional_mapping)

        t2t = Token2token.make(
            lang1,
            lang2,
            extended_tokenizer,
            original_tokenizer,
            datapref=datapref,
            column1=column1,
            column2=column2,
            split=split,
            num_workers=num_workers,
            savedir=savedir,
            n_lines=n_lines
            )
        id_map = extract_mapping(t2t, new_vocab_map)

    j_dump(id_map, map_save_path) 

    return extended_tokenizer