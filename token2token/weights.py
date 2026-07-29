from token2token import Token2token
from transformers import AutoConfig, AutoModelForCausalLM
from token2token.utils import j_read, get_savedir, j_dump, load_hf_fast_tokenizer
from os import path as px, makedirs
import torch
from tokenizers import Tokenizer
import transformers


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


def passtests(model, id_mapping, model_path):
    original_model = load_model_dynamically(model_path, device="cpu")
    print("--- Running Weight Assertions ---")
    weights_ok = test_embedding_weights(original_model, model, id_mapping)
    print("\n--- Running Forward Pass Test ---")
    target_ids = [int(x) for x in list(id_mapping.keys())]
    forward_ok = test_forward_pass(model, target_ids)
    if weights_ok and forward_ok:
        return True


def load_model_dynamically(model_path: str, device: str = "cpu"):
    """
    Reads the model configuration and loads it using its specific CausalLM 
    or primary class dynamically.
    """
    config = AutoConfig.from_pretrained(model_path)
    
    # Extract registered architecture class name if present (e.g., 'Qwen2ForCausalLM')
    arch_name = config.architectures[0] if getattr(config, "architectures", None) else None
    
    model_cls = None
    if arch_name and hasattr(transformers, arch_name):
        model_cls = getattr(transformers, arch_name)
    else:
        # Fallback to standard CausalLM loader if explicit class isn't directly exposed
        model_cls = AutoModelForCausalLM

    print(f"Loading '{model_path}' using dynamic class: {model_cls.__name__}")
    model = model_cls.from_pretrained(model_path, config=config, torch_dtype="auto").to(device)
    return model


def extract_embedding_layers(model):
    """
    Extracts the input embeddings, output embeddings, and output biases from a model.
    Handles weight-tying logic automatically.

    Returns:
        tuple: (input_layer, input_weights, output_layer, output_weights, output_bias, is_tied)
    """
    input_layer = model.get_input_embeddings()
    input_weights = input_layer.weight.data

    output_layer = model.get_output_embeddings()
    is_tied = getattr(model.config, "tie_word_embeddings", False)

    # Output weights are only untied/separate if is_tied is False
    output_weights = (
        output_layer.weight.data 
        if (output_layer is not None and not is_tied) 
        else None
    )

    # Output bias check
    output_bias = (
        output_layer.bias.data 
        if (output_layer is not None and getattr(output_layer, "bias", None) is not None) 
        else None
    )

    return input_layer, input_weights, output_layer, output_weights, output_bias, is_tied


def finalize_and_save_model(
    model,
    tokenizer,
    save_directory: str,
    id_mapping: dict = None,
    token_mapping: dict = None,
    model_path: str = None,
    run_tests: bool = True
):
    """
    Handles weight re-tying, optional test execution, and saving of model, 
    tokenizer, and mapping metadata.
    """
    is_tied = getattr(model.config, "tie_word_embeddings", False)
    if is_tied:
        model.tie_weights()
    makedirs(save_directory, exist_ok=True)

    if run_tests and id_mapping is not None and model_path is not None:
        if not passtests(model, id_mapping, model_path):
            raise RuntimeError("Model verification tests failed! Model was not saved.")

    print(f"Saving model and associated artifacts to '{save_directory}'...")
    model.save_pretrained(save_directory)
    if tokenizer is not None:
        tokenizer.save(px.join(save_directory, "tokenizer.json"))
    if id_mapping is not None:
        j_dump(id_mapping, px.join(save_directory, "id_map.json"))
    if token_mapping is not None:
        j_dump(token_mapping, px.join(save_directory, "token_map.json"))
    print("Success! Model and artifacts saved cleanly.")


def average_model_embeddings(
    model_paths: list[str],
    save_directory: str,
    tokenizer=None
):
    """
    Loads multiple models, averages their input embedding weights, output embedding weights 
    (if untied), and output biases across all models, and saves the resulting model.
    """
    if not model_paths:
        raise ValueError("The `model_paths` list cannot be empty.")

    num_models = len(model_paths)
    print(f"Starting weight averaging across {num_models} models...")

    # Step 1: Load base model and retrieve embedding layers
    print(f"Loading base model structure from: '{model_paths[0]}'...")
    base_model = load_model_dynamically(model_paths[0], device="cpu")
    input_layer, input_weights, output_layer, output_weights, output_bias, _ = extract_embedding_layers(base_model)

    # Accumulators stored in float32 for summation precision
    accum_input = input_weights.clone().to(torch.float32)
    accum_output = output_weights.clone().to(torch.float32) if output_weights is not None else None
    accum_bias = output_bias.clone().to(torch.float32) if output_bias is not None else None

    # Step 2: Accumulate weights from remaining models
    for path in model_paths[1:]:
        print(f"Accumulating weights from: '{path}'...")
        curr_model = load_model_dynamically(path, device="cpu")
        _, curr_in, _, curr_out, curr_b, _ = extract_embedding_layers(curr_model)

        accum_input += curr_in.to(torch.float32)
        if accum_output is not None and curr_out is not None:
            accum_output += curr_out.to(torch.float32)
        if accum_bias is not None and curr_b is not None:
            accum_bias += curr_b.to(torch.float32)

        del curr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Step 3: Compute mean and update base model
    print("Computing mean and updating base model tensors...")
    with torch.no_grad():
        input_layer.weight.data.copy_((accum_input / num_models).to(input_layer.weight.dtype))

        if accum_output is not None and output_layer is not None:
            output_layer.weight.data.copy_((accum_output / num_models).to(output_layer.weight.dtype))

        if accum_bias is not None and output_layer is not None:
            output_layer.bias.data.copy_((accum_bias / num_models).to(output_layer.bias.dtype))

    finalize_and_save_model(
        model=base_model,
        tokenizer=tokenizer,
        save_directory=save_directory,
        run_tests=False
    )


def model_transform(
    model_path: str,
    id_mapping: dict,
    token_mapping: dict,
    tokenizer,
    save_directory: str
):
    """
    Loads a pretrained model, replaces specified token embeddings and output biases 
    with the mean vector of mapped source tokens, and saves the updated model.
    """
    print(f"Loading tokenizer and model from '{model_path}'...")
    model = load_model_dynamically(model_path, device="cpu")
    input_layer, input_weights, output_layer, output_weights, output_bias, _ = extract_embedding_layers(model)

    print(f"Modifying parameters for {len(id_mapping)} tokens...")
    with torch.no_grad():
        orig_input = input_weights.clone()
        orig_output = output_weights.clone() if output_weights is not None else None
        orig_bias = output_bias.clone() if output_bias is not None else None

        for target_id, source_ids in id_mapping.items():
            if not source_ids:
                continue

            target_id = int(target_id)
            source_ids = [int(s) for s in source_ids]

            # Update Input Embeddings
            input_weights[target_id] = torch.mean(orig_input[source_ids], dim=0)

            # Update Output Weights (if untied)
            if output_weights is not None and orig_output is not None:
                output_weights[target_id] = torch.mean(orig_output[source_ids], dim=0)

            # Update Output Bias (if present)
            if output_bias is not None and orig_bias is not None:
                output_bias[target_id] = torch.mean(orig_bias[source_ids], dim=0)

    # Run verification tests and save finalized model & metadata
    finalize_and_save_model(
        model=model,
        tokenizer=tokenizer,
        save_directory=save_directory,
        id_mapping=id_mapping,
        token_mapping=token_mapping,
        model_path=model_path,
        run_tests=True
    )


def id_mapping_mean(pruned_tokenizer, extended_tokenizer, new_vocab_map):
    new_vocab_map = {extended_tokenizer.decode([y]): y for x, y in new_vocab_map.items()}
    strings, keys = zip(*new_vocab_map.items())
    batch_encodings = pruned_tokenizer.encode_batch(
        list(strings), 
        add_special_tokens=False
    )
    input_ids = [enc.ids for enc in batch_encodings]
    return dict(zip(keys, input_ids))


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
        num_workers: int = 16,
        savedir: str = None,
        **kwargs
):

    if not savedir:  
        savedir = get_savedir()

    extended_tokenizer = load_hf_fast_tokenizer(extended_tokenizer_path)
    map_save_path = px.join(extended_tokenizer_path, "id_map.json")
    token_map_save_path = px.join(extended_tokenizer_path, "token_map.json")
    if px.isfile(map_save_path) and px.isfile(token_map_save_path):
        id_map = j_read(map_save_path)
        token_mapping = j_read(token_map_save_path)
    
    else:
        original_tokenizer = Tokenizer.from_pretrained(model)
        pruned_tokenizer = load_hf_fast_tokenizer(pruned_tokenizer_path)
        new_vocab_map = j_read(new_vocab_map_path)
        id_map = id_mapping_mean(pruned_tokenizer, extended_tokenizer, new_vocab_map)

        if lang1!=lang2:
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

        decoded_keys = [extended_tokenizer.decode([x_id]) for x_id in id_map.keys()]
        all_y_ids = [[y_id] for y_ids in id_map.values() for y_id in y_ids]
        decoded_y_tokens = original_tokenizer.decode_batch(all_y_ids)
        token_mapping = {}
        y_idx = 0
        for x_text, y_ids in zip(decoded_keys, id_map.values()):
            count = len(y_ids)
            token_mapping[x_text] = decoded_y_tokens[y_idx : y_idx + count]
            y_idx += count

    model_transform(model, id_map, token_mapping, extended_tokenizer, savedir)
