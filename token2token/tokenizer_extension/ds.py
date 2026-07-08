from datasets import load_dataset


def ds_iterator(id, split="train", streaming=True, limit=None, fn=None):
    dataset = load_dataset(id, split=split, streaming=streaming)
    if limit:
        dataset = dataset.take(limit)
    for example in dataset:
        if fn:
            yield fn(example["text"])
        else:
            yield example["text"]
