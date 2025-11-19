import random
import requests
from typing import Any, Dict, List, Tuple
import os
import re

NEURONPEDIA_DOMAIN = "https://neuronpedia.org"

def select_examples(examples:dict, n_examples:int=3, shuffle:bool=True, unrelated_examples:dict=None) -> dict:
    """
    Select n_examples with highest activations and n_examples with lowest activations.
    If there aren't enough low activation examples, supplement with unrelated examples (treated as zero activation).

    Args:
        examples: Dictionary of examples with tokens and values
        n_examples: Number of examples to select from each end of the spectrum
        shuffle: Whether to shuffle the results
        unrelated_examples: Dictionary of unrelated examples to use as backup low-activation examples

    Returns:
        dict: Selected examples
    """
    # Calculate max activation for each example
    examples_with_max = []
    for sentence_id, values in examples.items():
        max_activation = max(values["values"]) if values["values"] else 0
        examples_with_max.append((sentence_id, values, max_activation))

    # Sort by max activation
    examples_with_max.sort(key=lambda x: x[2])

    # Select n_examples from each end
    selected = []
    
    # Add lowest activations
    low_examples = examples_with_max[:n_examples]
    if len(low_examples) < n_examples and unrelated_examples:
        needed = n_examples - len(low_examples)
        print(f"⚠️ Warning: Only found {len(low_examples)} low activation examples, need {n_examples}. Using {needed} unrelated examples as backup.")
        
        # Add unrelated examples with zero activations
        unrelated_items = list(unrelated_examples.items())
        for i in range(min(needed, len(unrelated_items))):
            sentence_id, values = unrelated_items[i]
            # Create zero activation values for all tokens
            zero_values = {
                "tokens": values["tokens"],
                "values": [0.0] * len(values["tokens"])
            }
            # Use a unique ID to avoid conflicts
            unique_id = f"unrelated_{sentence_id}"
            low_examples.append((unique_id, zero_values, 0.0))
        
        if len(low_examples) < n_examples:
            print(f"⚠️ Warning: Still only have {len(low_examples)} low activation examples after using unrelated examples backup.")
    elif len(low_examples) < n_examples:
        print(f"⚠️ Warning: Only found {len(low_examples)} low activation examples, need {n_examples}. No unrelated examples provided as backup.")
    
    selected.extend(low_examples)
    
    # Add highest activations
    high_examples = examples_with_max[-n_examples:]
    if len(high_examples) < n_examples:
        print(f"⚠️ Warning: Only found {len(high_examples)} high activation examples, need {n_examples}.")
    selected.extend(high_examples)

    # Shuffle if requested
    if shuffle:
        random.shuffle(selected)

    # Convert back to dictionary
    return {item[0]: item[1] for item in selected}


def interleave_tokens_and_values(values, max_tokens=150):
    """
    Interleave tokens and values, ensuring the highest activation token is included.

    Args:
        values: Dictionary with tokens and values
        max_tokens: Maximum number of tokens to include

    Returns:
        str: Interleaved tokens and values
    """
    tokens = values["tokens"]
    activations = values["values"]

    # Find the index of the highest activation
    max_activation_index = activations.index(max(activations)) if activations else 0

    # Determine the window to show, centered on the highest activation if possible
    start_index = max(0, max_activation_index - max_tokens // 2)
    end_index = min(len(tokens), start_index + max_tokens)

    # Adjust start_index if we're near the end of the sequence
    if end_index == len(tokens) and end_index - start_index < max_tokens:
        start_index = max(0, end_index - max_tokens)

    # Interleave tokens and activations within the window
    interleaved = []
    for token, activation in zip(tokens[start_index:end_index], activations[start_index:end_index]):
        text = token
        if activation >= 1:
            activation = int(activation)
            text = f"{token}({activation=})"
        interleaved.append(text)

    # Join the interleaved list into a single string
    result = "".join(interleaved).replace("   ", " ") # TODO: Figure out why this happens
    result = result.replace("▁", " ")

    # Add indicators if the sentence was truncated
    if start_index > 0:
        result = "[...] " + result
    if end_index < len(tokens):
        result += " [...]"

    return result


def format_sentence_for_prediction(values, max_tokens=150):
    """
    Format a sentence for prediction, ensuring the highest activation token is included.

    Args:
        values: Dictionary with tokens and values
        max_tokens: Maximum number of tokens to include

    Returns:
        str: Formatted sentence
    """
    tokens = values["tokens"]
    activations = values["values"]

    # Find the index of the highest activation
    max_activation_index = activations.index(max(activations)) if activations else 0

    # Determine the window to show, centered on the highest activation if possible
    start_index = max(0, max_activation_index - max_tokens // 2)
    end_index = min(len(tokens), start_index + max_tokens)

    # Adjust start_index if we're near the end of the sequence
    if end_index == len(tokens) and end_index - start_index < max_tokens:
        start_index = max(0, end_index - max_tokens)

    # Join the tokens within the window
    result = "".join(tokens[start_index:end_index]).replace("   ", " ") # TODO: Figure out why this happens
    result = result.replace("▁", " ")   

    # Add indicators if the sentence was truncated
    if start_index > 0:
        result = "[...] " + result
    if end_index < len(tokens):
        result += " [...]"

    return result


def process_examples(examples_dict, n_examples, max_tokens=150, unrelated_examples=None):
    """
    Process examples consistently for both explanation and prediction.

    Args:
        examples_dict: Dictionary of examples
        n_examples: Number of examples to select
        max_tokens: Maximum tokens per sentence
        unrelated_examples: Dictionary of unrelated examples to use as backup low-activation examples

    Returns:
        tuple: (selected_examples, test_examples, activations_str, tokens_str, test_index_to_id)
    """
    # Select examples for explanation (n_examples from each end of the spectrum)
    selected_examples = select_examples(examples_dict, n_examples=n_examples, unrelated_examples=unrelated_examples)

    # Convert to list and shuffle for display
    selected_items = list(selected_examples.items())
    random.shuffle(selected_items)

    # Create activations string for explanation
    activations_str = ""
    for i, (sentence_id, values) in enumerate(selected_items):
        interleaved = interleave_tokens_and_values(values, max_tokens=max_tokens)
        activations_str += f"**Sentence {i}:** {interleaved}\n\n"

    # Get remaining examples not used for explanation
    remaining_examples = {}
    for sentence_id, values in examples_dict.items():
        if sentence_id not in selected_examples:
            remaining_examples[sentence_id] = values

    # Sort remaining examples by activation for prediction
    remaining_with_max = []
    for sentence_id, values in remaining_examples.items():
        max_activation = max(values["values"]) if values["values"] else 0
        remaining_with_max.append((sentence_id, values, max_activation))

    # Sort by max activation
    remaining_with_max.sort(key=lambda x: x[2])

    # Select n_examples from each end for prediction
    test_selected = []
    # Add lowest activations (including zeros if available)
    low_test_examples = remaining_with_max[:n_examples]
    if len(low_test_examples) < n_examples and unrelated_examples:
        needed = n_examples - len(low_test_examples)
        print(f"⚠️ Warning: Only found {len(low_test_examples)} low activation test examples, need {n_examples}. Using {needed} unrelated examples as backup.")
        
        # Add unrelated examples with zero activations, but avoid ones already used in explanation
        unrelated_items = list(unrelated_examples.items())
        used_count = 0
        for sentence_id, values in unrelated_items:
            unique_id = f"unrelated_{sentence_id}"
            if unique_id not in selected_examples and used_count < needed:
                # Create zero activation values for all tokens
                zero_values = {
                    "tokens": values["tokens"],
                    "values": [0.0] * len(values["tokens"])
                }
                low_test_examples.append((unique_id, zero_values, 0.0))
                used_count += 1
        
        if len(low_test_examples) < n_examples:
            print(f"⚠️ Warning: Still only have {len(low_test_examples)} low activation test examples after using unrelated examples backup.")
    elif len(low_test_examples) < n_examples:
        print(f"⚠️ Warning: Only found {len(low_test_examples)} low activation test examples, need {n_examples}.")
    
    test_selected.extend(low_test_examples)
    
    # Add highest activations
    high_test_examples = remaining_with_max[-n_examples:]
    if len(high_test_examples) < n_examples:
        print(f"⚠️ Warning: Only found {len(high_test_examples)} high activation test examples, need {n_examples}.")
    test_selected.extend(high_test_examples)

    # Convert to dictionary
    test_examples = {item[0]: item[1] for item in test_selected}

    # Convert to list and shuffle for display
    test_items = list(test_examples.items())
    random.shuffle(test_items)

    # Create tokens string for prediction
    tokens_str = ""
    for i, (sentence_id, values) in enumerate(test_items):
        formatted = format_sentence_for_prediction(values, max_tokens=max_tokens)
        tokens_str += f"**Sentence {i}:** {formatted}\n\n"

    # Create a mapping from display index to original sentence_id for prediction
    test_index_to_id = {i: sentence_id for i, (sentence_id, _) in enumerate(test_items)}

    return selected_examples, test_examples, activations_str, tokens_str, test_index_to_id


def get_neuronpedia_feature(
    feature: int, layer: int, model: str = "gpt2-small", dataset: str = "res-jb"
) -> dict[str, Any]:
    """
    Fetch a feature from Neuronpedia API.

    Args:
        feature (int): Feature index
        layer (int): Layer number
        model (str): Model name
        dataset (str): Dataset name

    Returns:
        dict: Feature data including examples

    Raises:
        Exception: If there's an error fetching the data
    """
    try:
        print(f"Fetching feature data from Neuronpedia: feature={feature}, layer={layer}, model={model}, dataset={dataset}")

        # Construct the URL with the correct format
        url = f"{NEURONPEDIA_DOMAIN}/api/feature/{model}/{layer}-{dataset}/{feature}"
        print(f"Request URL: {url}")

        # Make the request
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error: Neuronpedia API returned status code {response.status_code}")
            print(f"Response content: {response.text}")
            raise Exception(f"Failed to fetch feature data: HTTP {response.status_code}")

        # Parse the response
        data = response.json()

        # Ensure index is an integer
        data["index"] = int(data["index"])

        print(f"Successfully fetched feature data with {len(data.get('examples', {}))} examples") # Note: Neuronpedia API might return 'activations' instead of 'examples'

        return data
    except Exception as e:
        print(f"Error in get_neuronpedia_feature: {e}")
        # Return a minimal structure with empty examples/activations to allow the code to continue
        return {"examples": {}, "activations": [], "index": feature}


def get_feature_examples(feature: int, layer: int, model: str = "gpt2-small", dataset: str = "res-jb") -> tuple[dict,str,list]:

    activations_data = get_neuronpedia_feature(
        feature=feature, layer=layer,
        model=model, dataset=dataset
    )

    activations = activations_data.get("activations", []) # Use .get for safety
    explanation = ""
    if "explanations" in activations_data and activations_data["explanations"]:
        explanation = activations_data["explanations"][0].get("description", "")

    examples = {}
    activation_values = []
    for i in range(len(activations)):
        activation = activations[i]
        tokens = activation.get("tokens", [])
        values = activation.get("values", [])
        examples[i] = {"tokens": tokens, "values": values}
        # Store max activation for each example, not the list of all activations
        activation_values.append(max(values) if values else 0)
    return examples, explanation, activation_values


def validate_features(task_data):
    """
    Validate that all features in the task can be accessed from Neuronpedia.

    Args:
        task_data (dict): Task data containing features to validate

    Returns:
        tuple: (bool, list) - Success status and list of problematic features
    """
    print("\n=== VALIDATING FEATURES ===")
    print(f"Total features to validate: {len(task_data.get('features', []))}")

    model = task_data.get("model", "gemma-2-9b")
    dataset = task_data.get("dataset", "gemmascope-res-131k")

    problematic_features = []
    success_count = 0

    for i, feature in enumerate(task_data.get("features", [])):
        feature_idx = feature.get("feature_idx")
        layer = feature.get("layer")
        is_benign = feature.get("is_benign", True)

        print(f"\nValidating feature {i+1}/{len(task_data.get('features', []))}: "
              f"layer={layer}, feature_idx={feature_idx}, is_benign={is_benign}")

        try:
            # Use the existing get_neuronpedia_feature function
            data = get_neuronpedia_feature(
                feature=feature_idx,
                layer=layer,
                model=model,
                dataset=dataset
            )

            # Check if the response has activations
            example_count = len(data.get("activations", []))

            if example_count == 0:
                print(f"  ⚠️ Warning: Feature found but has no activations/examples")
                problematic_features.append({
                    "feature_idx": feature_idx,
                    "layer": layer,
                    "is_benign": is_benign,
                    "error": "No activations/examples found"
                })
                continue

            print(f"  ✅ Success: Found {example_count} examples/activations")
            success_count += 1

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            problematic_features.append({
                "feature_idx": feature_idx,
                "layer": layer,
                "is_benign": is_benign,
                "error": str(e)
            })

    # Summary
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Total features: {len(task_data.get('features', []))}")
    print(f"Successfully validated: {success_count}")
    print(f"Problematic features: {len(problematic_features)}")

    if problematic_features:
        print("\nProblematic features:")
        for i, feature in enumerate(problematic_features):
            print(f"{i+1}. layer={feature['layer']}, feature_idx={feature['feature_idx']}, "
                  f"is_benign={feature['is_benign']}, error={feature['error']}")

    return len(problematic_features) == 0, problematic_features 


def extract_cot_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
    """
    Extract Chain of Thought (CoT) from assistant messages.

    Supports two formats:
    1. <think>...</think> tags in content (DeepSeek agent)
    2. Separate "reasoning" field (OpenRouter agent)

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        List of extracted CoT thoughts from assistant messages
    """
    cot_thoughts = []

    for message in messages:
        if message.get("role") == "assistant":
            # Method 1: Extract from <think> tags in content (DeepSeek)
            if "content" in message and isinstance(message["content"], str):
                content = message["content"]
                # Find all <think>...</think> patterns
                think_patterns = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
                cot_thoughts.extend(think_patterns)

            # Method 2: Extract from reasoning field (OpenRouter)
            if "reasoning" in message and message["reasoning"]:
                cot_thoughts.append(message["reasoning"])

    return cot_thoughts

