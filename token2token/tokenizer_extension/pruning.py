from token2token.utils import load_hf_tokenizer, get_vocab_and_merges
from json import loads, dumps
from tokenizers import Tokenizer
from re import compile


patterns = {
    "zh" : compile(r'[\u4e00-\u9fff\u3400-\u4dbf]'),
    "cyr": compile(r'[\u0400-\u04FF\u0500-\u052F]'),
    "both": compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u0400-\u04FF\u0500-\u052F]'),
    "all" : compile(r'[\s\S]')
} 


def prune_tokenizer(tokenizer_name, prune_target="both"):
    if prune_target not in patterns:
        raise ValueError(f"unrecognized --prune_target argument. Options: {', '.join(list(patterns.keys()))}")

    tokenizer = load_hf_tokenizer(tokenizer_name)

    cfg = loads(tokenizer._tokenizer.to_str())
    vocab, merges = get_vocab_and_merges(tokenizer)

    new_vocab_tokens = {}
    pruned_tokens_map = {}

    i = 0
    for token, token_id in vocab.items():
        decoded = tokenizer.decode([token_id])
        if patterns[prune_target].search(decoded):
            new_vocab_tokens[f"<redacted_{str(i)}>"]=token_id
            pruned_tokens_map[token] = token_id
            i += 1
        else:
            new_vocab_tokens[token] = token_id

    pruned_tokens = set(pruned_tokens_map.keys())
    cfg["model"]["vocab"] = new_vocab_tokens

    cfg["model"]["merges"] = [
        f"{m[0]} {m[1]}" 
        for m in merges
        if m[0] not in pruned_tokens and m[1] not in pruned_tokens and "".join(m) not in pruned_tokens
    ]

    cfg["added_tokens"] = [
        dict(token, id=token["id"])
        for token in cfg["added_tokens"] if token["content"] not in pruned_tokens
    ]

    tokenizer._tokenizer = Tokenizer.from_str(dumps(cfg))
    return tokenizer, i, pruned_tokens_map
