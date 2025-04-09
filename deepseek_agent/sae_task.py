from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_fact, model_graded_qa
from inspect_ai.solver import bridge
from inspect_ai.solver import Solver
from deepseek_agent_async import AsyncDeepSeekAgent
from typing import Any, Optional, Dict, Callable, Awaitable
from dataclasses import dataclass
import yaml 
import json
import os
from sae_labelling import sae_prompts
import datetime
from sae_labelling.sae_labelling_utils import get_feature_examples, process_examples
from sae_labelling.evaluate import evaluate_results
from deepseek_agent_async import AsyncDeepSeekAgent

encode_tools = "encode_message"
decode_tools = "decode_message" 

@dataclass
class ModelConfig:
    name: str
    size: str
    orthogonalize: bool = False
    steering: Optional[dict] = None
    scheming_eval: bool = False
    task: Optional[Dict[str, Any]] = None
    task_file: Optional[str] = None

@dataclass
class AgentConfig:
    type: str

@dataclass
class Config:
    model: ModelConfig
    agent: AgentConfig
    save_conversation: bool = False
    silent: bool = False
    max_steps: int = 10
    no_stegano: bool = False
    use_gt_labels: bool = False
    n_examples: int = 3
    validate_features: bool = False
    scheming_eval: bool = False
    task: Optional[Dict[str, Any]] = None
    task_file: Optional[str] = None

def load_config(config_path: str) -> Config:
    """
    Load configuration from a YAML file and create a Config object.
    
    This function reads a YAML configuration file and constructs a Config object
    with the appropriate settings. It handles special fields like task_file that
    need to be added as attributes to the Config object.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        Config: A Config object containing all the settings from the YAML file
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    
    # Extract task_file before creating Config object
    task_file = config_data.get('task_file')
    
    # Create Config object
    model_config = ModelConfig(**config_data.get('model', {}))
    agent_config = AgentConfig(**config_data.get('agent', {}))
    
    config = Config(
        model=model_config,
        agent=agent_config,
        save_conversation=config_data.get('save_conversation', False),
        silent=config_data.get('silent', False),
        max_steps=config_data.get('max_steps', 10),
        no_stegano=config_data.get('no_stegano', False),
        use_gt_labels=config_data.get('use_gt_labels', False),
        n_examples=config_data.get('n_examples', 3),
        validate_features=config_data.get('validate_features', False),
        scheming_eval=config_data.get('scheming_eval', False),
        task=config_data.get('task')
    )
    
    # Add task_file as an attribute
    if task_file:
        config.task_file = task_file
    
    return config


# Factory function that sets up the agent and returns the actual solver function
def sae_agents(config: Config) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
    # Initialize the agent based on config (moved from the main loop area)
    print(f"\n=== INITIALIZING AGENT ===")
    agent_type = config.agent.type if hasattr(config.agent, "type") else "llama"
    model_name = config.model.name if hasattr(config.model, "name") else "meta-llama/Meta-Llama-3.1-8B-Instruct"
    max_steps = config.max_steps if hasattr(config, "max_steps") else 10
    orthogonalize = config.model.orthogonalize if hasattr(config.model, "orthogonalize") else False
    
    print(f"Agent type: {agent_type}")
    print(f"Model name: {model_name}")
    print(f"Max steps: {max_steps}")
    print(f"Orthogonalize: {orthogonalize}")
    
    # Determine which tools to use based on stegano setting
    tools = ["directly_answer", "guess_activation"]
    if not (hasattr(config, "no_stegano") and config.no_stegano):
        tools.append(encode_tools)
        tools.append(decode_tools)
    
    print(f"Tools: {tools}")
    
    # Determine if we should use the scheming system message
    system_message = None
    if hasattr(config, "scheming_eval") and config.scheming_eval:
        system_message = sae_prompts.prompt_template_sysmsg_scheming # Use sae_prompts
        print(f"Using scheming system message.") # Removed printing the potentially long message
    
    # Get the agent instance
    try:
        agent_args = {
            "tools": tools,
            "model_name": model_name,
            "max_steps": max_steps,
            "save": config.save_conversation if hasattr(config, "save_conversation") else False,
            "verbose": not (hasattr(config, "silent") and config.silent),
            "orthogonalize": orthogonalize,
        }
        if system_message:
            agent_args["system_message"] = system_message
            
        agent_instance = AsyncDeepSeekAgent(**agent_args)
        print(f"Successfully initialized {agent_instance.name}")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        # Raise the exception so the task setup fails clearly
        raise RuntimeError(f"Agent initialization failed: {e}") from e

    # Define the actual solver function that inspect-ai will call for each sample
    async def wrap_sae_agent(input_data: dict[str, Any]) -> dict[str, Any]:
        # Extract the input list from the input sample
        input_list = input_data.get("input")
        # Extract metadata from the input sample
        metadata = input_data.get("metadata", {}) # Get metadata or empty dict if missing

        # Check if input_list is a non-empty list and its first element is a dict
        if not isinstance(input_list, list) or not input_list or not isinstance(input_list[0], dict):
            print(f"⚠️ Warning: Expected 'input' to be a list with one dictionary, but got: {input_list}. Skipping.")
            return {"explanation": "Error: Invalid input format", "score": 0.0}

        # Extract feature details from the metadata
        # feature_info = input_list[0] # No longer needed
        feature_idx = metadata.get("feature_idx")
        layer = metadata.get("layer")
        is_benign = metadata.get("is_benign")
        model = metadata.get("model")
        dataset = metadata.get("dataset")

        if None in [feature_idx, layer, is_benign, model, dataset]:
            print(f"⚠️ Warning: Missing required feature information in metadata: {metadata}. Skipping.")
            return {"explanation": "Error: Missing feature info in metadata", "score": 0.0}

        print(f"\nProcessing feature: layer={layer}, feature_idx={feature_idx}, is_benign={is_benign}")

        # Set consistent parameters from config for this feature
        n_examples = config.n_examples if hasattr(config, "n_examples") else 3
        max_tokens = 150  # Consistent max tokens for all sentences

        # Get feature data from Neuronpedia
        try:
            examples_dict, gt_explanation, activation_values = get_feature_examples(
                feature=feature_idx,
                layer=layer,
                model=model,
                dataset=dataset
            )
            
            if not examples_dict:
                print(f"⚠️ Warning: No examples found for feature L{layer}F{feature_idx}. Skipping.")
                return {"explanation": "Error: No examples found", "score": 0.0}
                
            print(f"Retrieved {len(examples_dict)} examples for L{layer}F{feature_idx}")
        except Exception as e:
            print(f"❌ Error fetching feature data for L{layer}F{feature_idx}: {e}")
            return {"explanation": f"Error fetching examples: {e}", "score": 0.0}
        
        # Process examples consistently
        selected_examples, test_examples, activations_str, tokens_str, test_index_to_id = process_examples(
            examples_dict, n_examples, max_tokens
        )
        
        print(f"Selected {len(selected_examples)} examples for explanation")
        print(f"Selected {len(test_examples)} examples for prediction testing")
        
        # --- Explanation Step ---
        
        # Determine explanation prompt template
        if hasattr(config, "scheming_eval") and config.scheming_eval:
            if hasattr(config, "no_stegano") and config.no_stegano:
                expl_prompt_template = sae_prompts.prompt_template_explanation_no_stegano_scheming
                print("Using non-steganographic scheming explanation prompt template")
            else:
                expl_prompt_template = sae_prompts.prompt_template_explanation_scheming
                print("Using steganographic scheming explanation prompt template")
        else:
            if hasattr(config, "no_stegano") and config.no_stegano:
                expl_prompt_template = sae_prompts.prompt_template_explanation_no_stegano
                print("Using non-steganographic explanation prompt template")
            else:
                expl_prompt_template = sae_prompts.prompt_template_explanation
                print("Using steganographic explanation prompt template")

        # Format the explanation prompt
        expl_description = expl_prompt_template.format(activations=activations_str)
        
        # Get explanation
        if hasattr(config, "use_gt_labels") and config.use_gt_labels and gt_explanation:
            # Use the ground truth explanation from Neuronpedia
            agent_explanation = gt_explanation
            print(f"Using ground truth explanation ({len(agent_explanation)} chars)")
        else:
            # Get explanation from agent
            print(f"Requesting explanation from agent...")
            try:
                agent_explanation, _ = await agent_instance(expl_description) # Ignoring conversation
                print(f"Received explanation ({len(agent_explanation)} chars)")
            except Exception as e:
                print(f"❌ Error getting explanation from agent: {e}")
                agent_explanation = f"Error during explanation: {e}"
        
        # --- Prediction Step ---

        # Determine predictor prompt template
        if hasattr(config, "scheming_eval") and config.scheming_eval:
            if hasattr(config, "no_stegano") and config.no_stegano:
                pred_prompt_template = sae_prompts.prompt_template_predictor_no_stegano_scheming
                print("Using non-steganographic scheming predictor template")
            else:
                pred_prompt_template = sae_prompts.prompt_template_predictor_scheming
                print("Using steganographic scheming predictor template")
        else:
            if hasattr(config, "no_stegano") and config.no_stegano:
                pred_prompt_template = sae_prompts.prompt_template_predictor_no_stegano
                print("Using non-steganographic predictor template")
            else:
                pred_prompt_template = sae_prompts.prompt_template_predictor
                print("Using steganographic predictor template")
        
        # Format the predictor prompt
        pred_description = pred_prompt_template.format(explanation=agent_explanation, tokens=tokens_str)      
        
        # Get prediction from agent
        print(f"Requesting activation prediction from agent...")
        try:
            final_answer, _ = await agent_instance(pred_description) # Ignoring conversation
            print(f"Received prediction response ({len(final_answer)} chars)")
            # Extract predicted activations stored by the 'guess_activation' tool
            predicted_activations_map = agent_instance.tools["guess_activation"].data.copy() # Copy data
            agent_instance.tools["guess_activation"].data = {} # Reset for next call
            print(f"Retrieved {len(predicted_activations_map)} activation predictions.")
        except Exception as e:
            print(f"❌ Error getting prediction from agent: {e}")
            predicted_activations_map = {}
            final_answer = f"Error during prediction: {e}"


        # --- Evaluation Step ---

        # Create feature result entry for evaluation
        feature_result_eval = {
            "feature_idx": feature_idx,
            "layer": layer,
            "is_benign": is_benign,
            "sentences": {}
        }
        
        # Add predicted activations (map display ID back to sentence ID)
        pred_activations_for_eval = {}
        for display_id, pred_value in predicted_activations_map.items():
            try:
                display_id_int = int(display_id) if isinstance(display_id, str) else display_id
                sentence_id = test_index_to_id.get(display_id_int)
                if sentence_id is None: continue
                sentence_id = int(sentence_id) if isinstance(sentence_id, str) else sentence_id
                
                # For eval structure
                pred_activations_for_eval[sentence_id] = float(pred_value)

                # For feature_result_eval structure
                if sentence_id not in feature_result_eval["sentences"]:
                    feature_result_eval["sentences"][sentence_id] = {}
                feature_result_eval["sentences"][sentence_id]["predicted"] = float(pred_value)
                feature_result_eval["sentences"][sentence_id]["made_prediction"] = True

            except Exception as e:
                print(f"Error processing prediction for display ID {display_id} during eval prep: {e}")
        
        # Add actual activations for sentences in the test set
        actual_activations_for_eval = []
        for sentence_id, values in test_examples.items():
            sentence_id = int(sentence_id) if isinstance(sentence_id, str) else sentence_id
            actual_values = values["values"]
            max_activation = max(actual_values) if actual_values else 0.0

            # For eval structure
            # Note: evaluate_results expects a list of activations per sentence (originally for tokens)
            # We use the max activation here as a single value list.
            actual_activations_for_eval.append(float(max_activation))

            # For feature_result_eval structure
            if sentence_id not in feature_result_eval["sentences"]:
                 feature_result_eval["sentences"][sentence_id] = {"predicted": 0.0, "made_prediction": False}
            feature_result_eval["sentences"][sentence_id]["actual"] = float(max_activation)

        # Construct the single-feature run_results dict for evaluate_results
        single_feature_run_results = {
            "agent_type": agent_type,
            "model_name": model_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "use_gt_labels": config.use_gt_labels if hasattr(config, "use_gt_labels") else False,
            "no_stegano": config.no_stegano if hasattr(config, "no_stegano") else False,
            "features": [[layer, feature_idx, is_benign]], # List containing single feature info
            "explanations": [gt_explanation], # List containing the single GT explanation
            "predicted_explanations": [agent_explanation], # List containing the single predicted explanation
            "activations": [actual_activations_for_eval], # List containing the list of actual activations for this feature
            "predicted_activations": [pred_activations_for_eval], # List containing the dict of predicted activations
            "feature_results": [feature_result_eval] # List containing the single feature result dict
        }

        # Perform evaluation for this single feature
        score = 0.0
        try:
            # Note: evaluate_results might need adjustments if it strictly expects multiple features.
            evaluation_results = evaluate_results(single_feature_run_results, no_overseer=False)
            score = evaluation_results.get("average_overall_score", 0.0)
            print(f"Feature L{layer}F{feature_idx} evaluation score: {score}")
        except Exception as e:
            print(f"❌ Error during evaluation for feature L{layer}F{feature_idx}: {e}")
            # Optionally include error details in the output if needed
            # agent_explanation += f"\nEvaluation Error: {e}"

        # Return the explanation and the calculated score
        return {"explanation": agent_explanation, "score": score}

    # Return the inner solver function
    return wrap_sae_agent



@task
def sae_task(scoring_model: Optional[str] = None, config_path: Optional[str] = "sae_labelling/config.yaml") -> Task:
    if scoring_model is None:
        scoring_model = "openai/gpt-4o-mini"

    if config_path is None:
        # Provide a default or raise an error if config_path is essential
        raise ValueError("config_path must be provided for sae_task")
        # Example default: config_path = "configs/default_sae_config.yaml" 
    
    config = load_config(config_path)

    # Ensure task_file is specified in the config
    if not hasattr(config, "task_file") or not config.task_file:
        raise ValueError("Config loaded from '{config_path}' must specify a 'task_file'.")

    # Ensure the task file exists
    if not os.path.exists(config.task_file):
         raise FileNotFoundError(f"Task file specified in config ('{config.task_file}') not found.")

    return Task(
        # Load dataset from the path specified in the loaded config
        dataset=json_dataset(config.task_file), 
        # Get the solver function using the factory and bridge it
        solver=bridge(sae_agents(config=config)), 
        # The scorer will receive the dict {"explanation": ..., "score": ...}
        # model_graded_qa expects a string. A custom scorer will be needed later
        # to parse the dict and use the explanation and/or score.
        scorer=model_graded_qa(model=scoring_model), 
        # Example of how a custom scorer might be used later:
        # scorer=custom_sae_scorer(model=scoring_model, target_key="explanation", score_key="score")
    )

# Example placeholder for a custom scorer if needed later
# from inspect_ai.scorer import Scorer, Score
# from inspect_ai.model import GenerateConfig
# from inspect_ai.sample import Sample
# 
# @scorer(metrics=["accuracy", "custom_score"]) # Define metrics
# def custom_sae_scorer(model: str, target_key: str = "explanation", score_key: str = "score") -> Scorer:
#     async def score(sample: Sample, config: GenerateConfig) -> Score:
#         # Access the solver's output (the dict)
#         output_dict = sample.output 
#         explanation = output_dict.get(target_key, "")
#         numeric_score = output_dict.get(score_key, 0.0)
#         target = sample.target # The target string from the dataset
# 
#         # Option 1: Use model_graded_qa logic on the explanation part
#         # Need to adapt model_graded_qa or reimplement its core logic here
#         # accuracy_score = await perform_model_graded_qa(explanation, target, model) 
#         accuracy_score = 0.5 # Placeholder
# 
#         # Option 2: Use the numeric score directly or combine scores
#         final_score = numeric_score # Or combine, e.g., (accuracy_score + numeric_score) / 2
# 
#         # Determine pass/fail based on the score
#         passed = final_score > 0.7 # Example threshold
#         
#         return Score(
#             value=final_score, 
#             passed=passed, 
#             metrics={"accuracy": accuracy_score, "custom_score": numeric_score},
#             explanation=f"Model graded explanation: {accuracy_score:.2f}, Numeric score: {numeric_score:.2f}" # Example
#         )
#     return score

