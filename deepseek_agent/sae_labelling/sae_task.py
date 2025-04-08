async def wrap_sae_agent(input_data: dict[str, Any]) -> dict[str, Any]:
    # Extract the input list from the input sample
    input_list = input_data.get("input")

    # Check if input_list is a non-empty list and its first element is a dict
    if not isinstance(input_list, list) or not input_list or not isinstance(input_list[0], dict):
        print(f"⚠️ Warning: Expected 'input' to be a list with one dictionary, but got: {input_list}. Skipping.")
        return {"explanation": "Error: Invalid input format", "score": 0.0}
    
    # Extract feature details from the first dictionary in the input list
    feature_info = input_list[0]
    feature_idx = feature_info.get("feature_idx")
    layer = feature_info.get("layer")
// ... existing code ...