from token2token import Token2token
from transformers import AutoTokenizer, AutoModel
from token2token.utils import j_read, get_savedir, j_dump
from os import path as px, makedirs
import torch


import torch

def test_embedding_weights(old_model, new_model, id_mapping, atol=1e-3):
    """
    Verifies that target token embeddings in new_model equal the mean of source
    token embeddings taken from old_model.
    """
    # 1. Source weights from OLD model
    old_input = old_model.get_input_embeddings().weight.data
    old_output_layer = old_model.get_output_embeddings()
    old_output = old_output_layer.weight.data if (old_output_layer and not getattr(old_model.config, "tie_word_embeddings", False)) else None
    old_bias = old_output_layer.bias.data if (old_output_layer and getattr(old_output_layer, "bias", None) is not None) else None

    # 2. Target weights from NEW model
    new_input = new_model.get_input_embeddings().weight.data
    new_output_layer = new_model.get_output_embeddings()
    new_output = new_output_layer.weight.data if (new_output_layer and not getattr(new_model.config, "tie_word_embeddings", False)) else None
    new_bias = new_output_layer.bias.data if (new_output_layer and getattr(new_output_layer, "bias", None) is not None) else None

    passed_count = 0
    failed_count = 0

    for target_id, source_ids in id_mapping.items():
        if not source_ids:
            continue

        target_id = int(target_id)
        source_ids = [int(s) for s in source_ids]

        # --- A. Test Input Embeddings ---
        expected_input_mean = torch.mean(old_input[source_ids], dim=0)
        actual_input_vec = new_input[target_id]
        
        if torch.allclose(actual_input_vec, expected_input_mean, atol=atol):
            passed_count += 1
        else:
            failed_count += 1

        # --- B. Test Output Weights (if untied) ---
        if new_output is not None and old_output is not None:
            expected_output_mean = torch.mean(old_output[source_ids], dim=0)
            actual_output_vec = new_output[target_id]
            if torch.allclose(actual_output_vec, expected_output_mean, atol=atol):
                passed_count += 1
            else:
                failed_count += 1

        # --- C. Test Output Bias (if present) ---
        if new_bias is not None and old_bias is not None:
            expected_bias_mean = torch.mean(old_bias[source_ids], dim=0)
            actual_bias_val = new_bias[target_id]
            if torch.allclose(actual_bias_val, expected_bias_mean, atol=atol):
                passed_count += 1
            else:
                failed_count += 1

    total_tests = passed_count + failed_count
    print(f"\n--- Embedding Weight Verification Summary ---")
    print(f"Passed: {passed_count} / {total_tests}")
    print(f"Failed: {failed_count} / {total_tests}")

    return failed_count == 0


def test_forward_pass(model, target_ids, max_seq_len=128):
    """
    Runs a test forward pass using a safe sample of modified target token IDs.
    """
    model.eval()

    # 1. Ensure target_ids are clean integers
    if isinstance(target_ids, dict):
        target_ids = list(target_ids.keys())
        
    int_target_ids = []
    for item in target_ids:
        try:
            int_target_ids.append(int(item))
        except (ValueError, TypeError):
            continue

    if not int_target_ids:
        print("❌ No valid integer target IDs found to test!")
        return False

    # 2. Slice/Chunk into a safe, short sequence length (e.g., first 128 tokens)
    sample_ids = int_target_ids[:max_seq_len]
    
    # Shape: (batch_size=1, sequence_length=128)
    test_input_ids = torch.tensor([sample_ids], dtype=torch.long, device=model.device)

    with torch.no_grad():
        try:
            outputs = model(input_ids=test_input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state

            # 3. Check for NaNs or Infs
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("❌ Forward pass produced NaN or Inf values!")
                return False

            print(f"✅ Forward pass successful! Output tensor shape: {logits.shape}")
            return True

        except Exception as e:
            print(f"❌ Forward pass failed with error: {e}")
            return False

def passtests(id_mapping, model, model_path):
    original_model = AutoModel.from_pretrained(
        model_path,
        torch_dtype="auto",  # Preserves model's original float precision
    )

    model = model.to("cpu")
    print("--- Running Weight Assertions ---")
    weights_ok = test_embedding_weights(original_model, model, id_mapping)

    print("\n--- Running Forward Pass Test ---")
    target_ids = [int(x) for x in list(id_mapping.keys())]
    forward_ok = test_forward_pass(model, target_ids)
    if weights_ok and forward_ok:
        print("\n🎉 Everything passed! Model is ready to save.")


def model_transform(
    model_path: str,
    id_mapping: dict,
    tokenizer,
    save_directory: str
):
    """
    Loads a pretrained model, replaces specified token embeddings and output biases 
    with the median vector of mapped source tokens, and saves the updated model.

    Args:
        model_path (str): Hugging Face Hub ID or local directory path.
        id_mapping (dict): Dict mapping target token IDs to source ID lists, 
                           e.g., {0: [100, 101], 1: [100, 182]}
        save_directory (str): Destination path to save model & tokenizer.
    """
    print(f"Loading tokenizer and model from '{model_path}'...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype="auto",  # Preserves model's original float precision
    )

    model = model.to("cpu")
    # 1. Retrieve Input Embeddings Layer
    input_layer = model.get_input_embeddings()
    input_weights = input_layer.weight.data

    # 2. Retrieve Output Layer & Check Weight-Tying Structure
    output_layer = model.get_output_embeddings()
    is_tied = getattr(model.config, "tie_word_embeddings", False)

    # Output weights are only untied/separate if is_tied is False
    output_weights = (
        output_layer.weight.data 
        if (output_layer is not None and not is_tied) 
        else None
    )

    # Check if the output linear layer contains a bias tensor
    output_bias = (
        output_layer.bias.data 
        if (output_layer is not None and getattr(output_layer, "bias", None) is not None) 
        else None
    )

    print(f"Modifying parameters for {len(id_mapping)} tokens...")
    
    # Disable autograd for safe in-place memory modifications
    with torch.no_grad():
        orig_input = input_weights.clone()
        orig_output = output_weights.clone() if output_weights is not None else None
        orig_bias = output_bias.clone() if output_bias is not None else None

        for target_id, source_ids in id_mapping.items():
            if not source_ids:
                continue

            target_id = int(target_id)
            source_ids = [int(s) for s in source_ids]

            #--- A. Update Input Embeddings ---
            src_input_vecs = orig_input[source_ids]
            mean_input_vec = torch.mean(src_input_vecs, dim=0)
            input_weights[target_id] = mean_input_vec

            # --- B. Update Output Weights (If separate/untied) ---
            if output_weights is not None and orig_output is not None:
                src_output_vecs = orig_output[source_ids]
                mean_output_vec = torch.mean(src_output_vecs, dim=0)
                output_weights[target_id] = mean_output_vec

            # --- C. Update Output Bias (If bias exists) ---
            if output_bias is not None and orig_bias is not None:
                src_biases = orig_bias[source_ids]
                mean_bias = torch.mean(src_biases, dim=0)
                output_bias[target_id] = mean_bias

    # Re-tie weights if applicable (good practice for HF models)
    if is_tied:
        model.tie_weights()

    # Save everything locally
    makedirs(save_directory, exist_ok=True)

    if passtests(id_mapping, model, model_path):
        print(f"Saving updated model and tokenizer to '{save_directory}'...")
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)

        print("Success! Model and tokenizer saved.")


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
        subset: str = None,
        column1: str = None,
        column2: str = None,
        n_lines=10000000,
        reinitialize_old: bool = False,
        no_overlap: str = "en",
        no_overlap_data: str = None,
        no_overlap_split: str = "train",
        no_overlap_subset: str = None,
        no_overlap_lines: str = None,
        num_workers: int = 16,
        savedir: str = None,
        **kwargs
):

    if not savedir:  
        savedir = get_savedir()

    extended_tokenizer = AutoTokenizer.from_pretrained(extended_tokenizer_path)
    map_save_path = px.join(extended_tokenizer_path, "id_mapping.json")
    if px.isfile(map_save_path):
        id_map = j_read(map_save_path)
    
    else:
        original_tokenizer = AutoTokenizer.from_pretrained(model)
        pruned_tokenizer = AutoTokenizer.from_pretrained(pruned_tokenizer_path)
        new_vocab_map = j_read(new_vocab_map_path)
        id_map = id_mapping_mean(pruned_tokenizer, extended_tokenizer, new_vocab_map)

        if lang1!=lang2:
            if reinitialize_old:
                xfpm, yfpm, token2x, token2y = Token2token.make(
                    lang1,
                    no_overlap,
                    extended_tokenizer,
                    extended_tokenizer,
                    datapref=no_overlap_data,
                    split=no_overlap_split,
                    subset=no_overlap_subset,
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
                subset=subset,
                num_workers=num_workers,
                savedir=savedir,
                n_lines=n_lines
                )
            id_map1 = extract_mapping(t2t, new_vocab_map)
            id_map.update(id_map1)

        j_dump(id_map, map_save_path) 

    model_transform(model, id_map, extended_tokenizer, savedir)
