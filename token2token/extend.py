from transformers import AutoTokenizer
from token2token import Token2token
from token2token.utils import ds_iterator
from .tokenizer_extension.pruning import prune_tokenizer
from .tokenizer_extension.train_vocab_extension import train_vocab_extension
from .tokenizer_extension.extension import extend_tokenizer
import shutil


n_lines=10000

def id_mapping(tokenizer, extended_tokenizer, t2t, no_overlap):
    return {}

def id_mapping_mean():
    return {}

def perform_extension(
        lang1: str,
        lang2: str,
        model,
        dataset: str = None,
        datapref: str = None,
        split: str = "train",
        prune_target: str = "both",
        extension_size: int = None,
        column1: str = None,
        column2: str = None,
        no_overlap: str = None,
        savedir: str = None,
        num_workers: int = 16,
):

    if dataset:
        pruned_tokenizer, n_pruned, prunned_map = prune_tokenizer(model, prune_target)
        pruned_tokenizer.save_pretrained("./my-pruned-tokenizer")

        if extension_size:
            extension_size = min(extension_size, n_pruned)
        else:
            extension_size = n_pruned
        dataset = ds_iterator(dataset, split, limit=1000)
        
        extension_tokens = train_vocab_extension(
            tokenizer=pruned_tokenizer,
            corpus=dataset,
            extension_size=extension_size
        )

        print("new vocab and merges prepared, extending now...")

        extended_tokenizer, new_vocab_map = extend_tokenizer(
            pruned_tokenizer,
            new_vocab=extension_tokens["vocab"],
            new_merges=extension_tokens["merges"],
            prunned_ids=list(prunned_map.values()) 
        )

        shutil.copytree("./my-pruned-tokenizer", "./my-tokenizer")
        extended_tokenizer.save("./my-tokenizer/tokenizer.json")
   
    extended_tokenizer = AutoTokenizer.from_pretrained("./my-tokenizer")
    pruned_tokenizer = AutoTokenizer.from_pretrained("./my-pruned-tokenizer")

    # Ovde fali cuvanje new_vocab map, a možda fali i još nešto?

    if lang1 != lang2:
        t2t = Token2token.make(lang1, lang2, extended_tokenizer, pruned_tokenizer, datapref=datapref, column1=column1, column2=column2, num_workers=num_workers, savedir=savedir, n_lines=n_lines)
        if no_overlap:
            no_overlap = Token2token.make(lang1, no_overlap, extended_tokenizer, extended_tokenizer, datapref=datapref, column1=column1, column2=column2, num_workers=num_workers, savedir=savedir, n_lines=n_lines)
    else:
        id_map = id_mapping(model, new_vocab_map)

    return extended_tokenizer
