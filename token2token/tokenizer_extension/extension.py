from typing import Optional, List, Tuple, Dict, Union
from json import loads, dumps
from tokenizers import Tokenizer


def extend_tokenizer(
        tokenizer,
        prunned_ids: List[int],
        new_vocab: Union[Dict[str, int], List[str]],
        new_merges: Optional[List[Tuple[str, str]]] = None,
):
    """
    Extends the tokenizer with new tokens and merges inplace (changing the original tokenizer object).
    :param tokenizer: Tokenizer to extend
    :param prunned_ids: A list of available token ids resulting from prunning.
    :param new_vocab: New vocabulary to add
    :param new_merges: The merges to add. If None, merges will be generated based on the new_vocab, leaving existing merges intact.
    :return: the tokenizer object that was extended inplace
    """
    tokenizer_json = loads(tokenizer._tokenizer.to_str())
    new_vocab_map = {}

    if isinstance(new_vocab, dict):
        new_vocab = list(new_vocab.keys())
    prunned_ids = set(prunned_ids)

    vocab = tokenizer_json["model"]["vocab"]
    vocab_items = [list(item) for item in vocab.items()]
    new_vocab_iter = iter(new_vocab)

    for idx in range(len(vocab_items)):
        if idx in prunned_ids:
            new_token = next(new_vocab_iter, None)
            if new_token is None:
                break
            vocab_items[idx][0] = new_token
            new_vocab_map[new_token] = idx

    combined_vocab = dict(vocab_items)
    tokenizer_json["model"]["vocab"] = combined_vocab

    existing_merges = set()
    for m in tokenizer_json["model"]["merges"]:
        if isinstance(m, list):
            existing_merges.add(" ".join(m))
        else:
            existing_merges.add(m)

    # Format the new merges you want to inject
    formatted_new_merges = [" ".join(m) for m in new_merges]

    # Cleanly filter out duplicates and invalid matches
    valid_new_merges = [
        m for m, (a, b) in zip(formatted_new_merges, new_merges)
        if a in combined_vocab 
        and b in combined_vocab 
        and (a + b) in combined_vocab 
        and m not in existing_merges
    ]
    
    if len(tokenizer_json["model"]["merges"]) > 0 and isinstance(tokenizer_json["model"]["merges"][0], list):
        # If the original is list of lists, append them as lists
        tokenizer_json["model"]["merges"].extend([m.split() for m in valid_new_merges])
    else:
        # Otherwise, append as space-separated strings
        tokenizer_json["model"]["merges"].extend(valid_new_merges)

    if len(set(tokenizer_json["model"]["vocab"].keys())) != len(set(tokenizer_json["model"]["vocab"].values())):
        raise ValueError("Tokens with the same ID found in vocabulary.")
    return Tokenizer.from_str(dumps(tokenizer_json)), new_vocab_map
