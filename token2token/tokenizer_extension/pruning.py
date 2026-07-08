def prune_tokenizer(tokenizer, prune_target="both"):
    if prune_target not in ["cyr", "zh", "both", "all"]:
        raise ValueError("unrecognized --prune_target argument. "
                             "Options: cyr, zh, both, all")
    return tokenizer