from token2token import Token2token
from transformers import AutoTokenizer
from token2token.utils import j_read, get_savedir, j_dump
from os import path as px


n_lines=10000

def id_mapping(tokenizer, extended_tokenizer, t2t, no_overlap):
    return {}


def id_mapping_mean(pruned_tokenizer, new_vocab_map):
    strings, keys = zip(*new_vocab_map.items())
    batch_encodings = pruned_tokenizer(list(strings), add_special_tokens=False)
    return dict(zip(keys, batch_encodings["input_ids"]))


def extract_old_vocab(t2t, new_vocab_map):
    src_tokens = t2t.token2y
    tgt_tokens = t2t.token2x
    return {token: id for token, id in src_tokens.items() if token not in tgt_tokens and token not in new_vocab_map}

def extract_mapping(t2t, new_vocab_map, tokenizer):
    id_map = {}
    for token, id in new_vocab_map.items():
        if token not in t2t.translations:
            id_map[id] = []
        else:
            t_mapping = t2t(token)
            mapped_token = max(t_mapping, key=t_mapping.get)
            id_map[id] = [tokenizer.convert_tokens_to_ids(mapped_token)]
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

        reinitialize_old: bool = False,
        no_overlap: str = "en",
        num_workers: int = 16,
        savedir: str = None,
        **kwargs
):

    if not savedir:  
        savedir = get_savedir()

    map_save_path = px.join(savedir, "id_mapping.json")
    

    extended_tokenizer = AutoTokenizer.from_pretrained(extended_tokenizer_path)
    pruned_tokenizer = AutoTokenizer.from_pretrained(pruned_tokenizer_path)
    new_vocab_map = j_read(new_vocab_map_path)

    if lang1 == lang2:
        id_map = id_mapping_mean(pruned_tokenizer, new_vocab_map)

    else:
        if reinitialize_old:
            overlap_t2t = Token2token.make(lang1, no_overlap, extended_tokenizer, extended_tokenizer, datapref=datapref, column1=column1, column2=column2, num_workers=num_workers, savedir=savedir, n_lines=n_lines)
        additional_mapping = extract_old_vocab(overlap_t2t, new_vocab_map)
        new_vocab_map.update(additional_mapping)

        t2t = Token2token.make(lang1, lang2, extended_tokenizer, pruned_tokenizer, datapref=datapref, column1=column1, column2=column2, split=split, num_workers=num_workers, savedir=savedir, n_lines=n_lines)
        id_map = extract_mapping(t2t, new_vocab_map, extended_tokenizer)

    j_dump(id_map, map_save_path) 

    return extended_tokenizer