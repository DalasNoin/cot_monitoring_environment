from agents.agents import get_agent
import json
import argparse
import yaml
import random
import requests
from typing import Any, Optional, Dict, List, Tuple
import json
from datetime import datetime
from utils import keys
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# Import the moved functions from the new utils file
from sae_labelling.sae_labelling_utils import (
    select_examples,
    interleave_tokens_and_values,
    format_sentence_for_prediction,
    process_examples,
    get_neuronpedia_feature,
    get_feature_examples,
    validate_features
)
# Import the prompts from the new prompts file
from . import sae_prompts

# Basically have a task template that explains the setup including using the steganographic tools

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
    scheming_nudge_level: int = 1
    youre_being_watched: bool = False
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
        scheming_nudge_level=config_data.get('scheming_nudge_level', 1),
        youre_being_watched=config_data.get('youre_being_watched', False),
        task=config_data.get('task')
    )
    
    # Add task_file as an attribute
    if task_file:
        config.task_file = task_file
    
    return config

def create_run_directory(config: Config) -> str:
    """Create a directory for the current run with timestamp and agent type."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_type = config.agent.type
    gt_label_str = "with_gt" if config.use_gt_labels else "without_gt"
    stegano_str = "no_stegano" if config.no_stegano else "with_stegano"
    scheming_str = "scheming" if config.scheming_eval else "normal"
    
    # Create directory name
    dir_name = f"sae_labelling_run_{timestamp}_{agent_type}_{gt_label_str}_{stegano_str}_{scheming_str}"
    
    # Create full path
    run_dir = os.path.join("sae_labelling", "runs", dir_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config to the directory
    config_dict = {
        "model": {
            "name": config.model.name,
            "size": config.model.size,
            "orthogonalize": config.model.orthogonalize
        },
        "agent": {
            "type": config.agent.type
        },
        "save_conversation": config.save_conversation,
        "silent": config.silent,
        "max_steps": config.max_steps,
        "no_stegano": config.no_stegano,
        "use_gt_labels": config.use_gt_labels,
        "n_examples": config.n_examples,
        "scheming_eval": config.scheming_eval,
        "task": config.task
    }
    
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    return run_dir

def run_agent(config):
    """
    Run the agent with the given config.
    """
    # Create a unique run directory with timestamp and agent type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_type = config.agent.type if hasattr(config.agent, "type") else "unknown"
    gt_label_str = "with_gt" if hasattr(config, "use_gt_labels") and config.use_gt_labels else "without_gt"
    run_dir_name = f"sae_labelling_run_{timestamp}_{agent_type}_{gt_label_str}"
    
    # Create the run directory under sae_labelling/runs/
    runs_dir = os.path.join("sae_labelling", "runs")
    os.makedirs(runs_dir, exist_ok=True)
    run_dir = os.path.join(runs_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n=== CREATED RUN DIRECTORY ===")
    print(f"Run directory: {run_dir}")
    
    # Save the config to the run directory
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        # Convert config to dict for JSON serialization
        if hasattr(config, "__dict__"):
            config_dict = config.__dict__
        else:
            config_dict = config
        json.dump(config_dict, f, indent=2, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o))
    
    print(f"Saved config to: {config_path}")
    
    # Load task from file if specified
    if hasattr(config, "task_file") and config.task_file:
        print(f"\n=== LOADING TASK FROM FILE ===")
        print(f"Task file: {config.task_file}")
        
        try:
            with open(config.task_file, 'r') as file:
                task_data = json.load(file)
                # Save a copy of the task file in the run directory
                task_copy_path = os.path.join(run_dir, "task.json")
                with open(task_copy_path, 'w') as f:
                    json.dump(task_data, f, indent=2)
                print(f"Loaded task with {len(task_data.get('features', []))} features")
                print(f"Saved task copy to: {task_copy_path}")
        except Exception as e:
            print(f"❌ Error loading task file: {e}")
            print(f"Absolute path attempted: {os.path.abspath(config.task_file)}")
            print(f"Current working directory: {os.getcwd()}")
            return None, run_dir
    else:
        # Use the task defined in the config
        print(f"\n=== USING TASK FROM CONFIG ===")
        task_data = config.task if hasattr(config, "task") else {"features": []}
        # Save the task to the run directory
        task_path = os.path.join(run_dir, "task.json")
        with open(task_path, 'w') as f:
            json.dump(task_data, f, indent=2)
        print(f"Using task with {len(task_data.get('features', []))} features")
        print(f"Saved task to: {task_path}")
    
    # Initialize the agent
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
    tools = ["directly_answer", "guess_activations"]
    if not (hasattr(config, "no_stegano") and config.no_stegano):
        tools.append(encode_tools)
        tools.append(decode_tools)
    
    print(f"Tools: {tools}")
    
    # Determine which system message to use based on scheming_nudge_level and youre_being_watched
    system_message = None
    nudge_level = getattr(config, "scheming_nudge_level", 1)
    being_watched = getattr(config, "youre_being_watched", False)
    
    if being_watched:
        if nudge_level == 1:
            system_message = sae_prompts.prompt_template_sysmsg_nudge_level_1_watched
        elif nudge_level == 2:
            system_message = sae_prompts.prompt_template_sysmsg_nudge_level_2_watched
        elif nudge_level == 3:
            system_message = sae_prompts.prompt_template_sysmsg_nudge_level_3_watched
        elif nudge_level == 4:
            system_message = sae_prompts.prompt_template_sysmsg_nudge_level_4_watched
    else:
        if nudge_level == 1:
            system_message = sae_prompts.prompt_template_sysmsg_nudge_level_1
        elif nudge_level == 2:
            system_message = sae_prompts.prompt_template_sysmsg_nudge_level_2
        elif nudge_level == 3:
            system_message = sae_prompts.prompt_template_sysmsg_nudge_level_3
        elif nudge_level == 4:
            system_message = sae_prompts.prompt_template_sysmsg_nudge_level_4
    
    watched_desc = " + CoT monitoring warning" if being_watched else ""
    if system_message:
        print(f"Using scheming nudge level {nudge_level} system message{watched_desc}.")
    else:
        print(f"Using no special system message (nudge level {nudge_level}{watched_desc}).")
    
    # Get the agent
    try:
        # Pass the system_message parameter only if it's set
        if system_message:
            agent_instance = get_agent(agent_type)(
                tools=tools,
                model_name=model_name,
                max_steps=max_steps,
                save=config.save_conversation if hasattr(config, "save_conversation") else False,
                verbose=not (hasattr(config, "silent") and config.silent),
                orthogonalize=orthogonalize,
                system_message=system_message
            )
        else:
            agent_instance = get_agent(agent_type)(
                tools=tools,
                model_name=model_name,
                max_steps=max_steps,
                save=config.save_conversation if hasattr(config, "save_conversation") else False,
                verbose=not (hasattr(config, "silent") and config.silent),
                orthogonalize=orthogonalize
            )
        print(f"Successfully initialized {agent_instance.name}")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return None, run_dir
    
    # Initialize run results
    run_results = {
        "agent_type": agent_type,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "use_gt_labels": hasattr(config, "use_gt_labels") and config.use_gt_labels,
        "no_stegano": hasattr(config, "no_stegano") and config.no_stegano,
        "features": [],
        "explanations": [],  # Ground truth explanations from Neuronpedia
        "predicted_explanations": [],  # Explanations generated by the agent
        "activations": [],
        "predicted_activations": [],
        "feature_results": []
    }
    
    # Process each feature
    print(f"\n=== PROCESSING FEATURES ===")
    explanation_conversations = []
    prediction_conversations = []
    
    # Set consistent parameters
    n_examples = config.n_examples if hasattr(config, "n_examples") else 3
    max_tokens = 150  # Consistent max tokens for all sentences
    
    # Use the loaded task_data
    for i, feature in enumerate(task_data.get("features", [])):
        print(f"\nProcessing feature {i+1}/{len(task_data.get('features', []))}: "
              f"layer={feature['layer']}, feature_idx={feature['feature_idx']}, "
              f"is_benign={feature.get('is_benign', True)}")
        
        # Get feature data from Neuronpedia
        try:
            # Replace direct call to get_neuronpedia_feature with get_feature_examples
            examples_dict, explanation, activation_values = get_feature_examples(
                feature=feature["feature_idx"],
                layer=feature["layer"],
                model=task_data.get("model", "gemma-2-9b"),
                dataset=task_data.get("dataset", "gemmascope-res-131k")
            )
            feature["explanation"] = explanation
            run_results["explanations"].append(explanation)
            
            if not examples_dict:
                print(f"⚠️ Warning: No examples found for this feature. Skipping.")
                continue
                
            print(f"Retrieved {len(examples_dict)} examples")
        except Exception as e:
            print(f"❌ Error fetching feature data: {e}")
            continue
        
        # Add feature to run results
        run_results["features"].append(feature)
        
        # Process examples consistently
        selected_examples, test_examples, activations_str, tokens_str, test_index_to_id = process_examples(
            examples_dict, n_examples, max_tokens, None
        )
        
        print(f"Selected {len(selected_examples)} examples for explanation")
        print(f"Selected {len(test_examples)} examples for prediction testing")
        
        # Determine explanation prompt template based on scheming_nudge_level and stegano settings
        nudge_level = getattr(config, "scheming_nudge_level", 1)
        no_stegano = hasattr(config, "no_stegano") and config.no_stegano
        
        if nudge_level == 1:
            prompt_template = sae_prompts.prompt_template_explanation_nudge_level_1
            prompt_desc = "nudge level 1 (no scheming pressure)"
        elif nudge_level == 2:
            prompt_template = sae_prompts.prompt_template_explanation_nudge_level_2
            prompt_desc = "nudge level 2 (subtle consequences)"
        elif nudge_level == 3:
            prompt_template = sae_prompts.prompt_template_explanation_nudge_level_3
            prompt_desc = "nudge level 3 (moderate scheming pressure)"
        elif nudge_level == 4:
            if no_stegano:
                prompt_template = sae_prompts.prompt_template_explanation_nudge_level_4
                prompt_desc = "nudge level 4 (full scheming pressure, no stegano)"
            else:
                prompt_template = sae_prompts.prompt_template_explanation_scheming
                prompt_desc = "nudge level 4 (full scheming pressure, with stegano)"
        else:
            # Fallback to level 1
            prompt_template = sae_prompts.prompt_template_explanation_nudge_level_1
            prompt_desc = "nudge level 1 (fallback)"
        
        print(f"Using explanation prompt template: {prompt_desc}")
        
        # Format the prompt
        description = prompt_template.format(activations=activations_str)
        
        # Get the ground truth explanation from Neuronpedia if available and use_gt_labels is True.
        # Otherwise, get explanation from agent.
        if hasattr(config, "use_gt_labels") and config.use_gt_labels and "explanation" in feature:
            # Use the ground truth explanation from the feature
            ground_truth_explanation = feature.get("explanation", "")
            # run_results["explanations"].append(ground_truth_explanation)
            
            # When using ground truth labels, skip the agent explanation step
            answer = ground_truth_explanation
            conversation = []  # Empty conversation since we're not using the agent
            print(f"Using ground truth explanation ({len(answer)} chars)")
        else:
            # No ground truth available, get explanation from agent
            print(f"Requesting explanation from agent...")
            answer, conversation = agent_instance(description)
            print(f"Received explanation ({len(answer)} chars)")
            
            # Save explanation conversation to run directory
            conv_filename = f"conversation_explanation_{feature['layer']}_{feature['feature_idx']}.json"
            conv_path = os.path.join(run_dir, conv_filename)
            with open(conv_path, 'w') as f:
                json.dump(conversation, f, indent=2)
            
            # Also save to the original conversations directory for backward compatibility
            if hasattr(agent_instance, "filepath") and agent_instance.filepath:
                os.makedirs(os.path.dirname(agent_instance.filepath), exist_ok=True)
                with open(agent_instance.filepath, 'a') as f:
                    f.write(json.dumps({"conversation": conversation, "timestamp": datetime.now().isoformat()}) + "\n")
            
            print(f"Saved explanation conversation to: {conv_path}")
            
            explanation_conversations.append(conversation)
            run_results["predicted_explanations"].append(answer)
            
        
        # Determine predictor prompt template based on scheming_nudge_level and stegano settings
        nudge_level = getattr(config, "scheming_nudge_level", 1)
        no_stegano = hasattr(config, "no_stegano") and config.no_stegano
        
        if nudge_level == 1:
            prompt_template = sae_prompts.prompt_template_predictor_nudge_level_1
            prompt_desc = "nudge level 1 (no scheming pressure)"
        elif nudge_level == 2:
            prompt_template = sae_prompts.prompt_template_predictor_nudge_level_2
            prompt_desc = "nudge level 2 (subtle consequences)"
        elif nudge_level == 3:
            prompt_template = sae_prompts.prompt_template_predictor_nudge_level_3
            prompt_desc = "nudge level 3 (moderate scheming pressure)"
        elif nudge_level == 4:
            if no_stegano:
                prompt_template = sae_prompts.prompt_template_predictor_nudge_level_4
                prompt_desc = "nudge level 4 (full scheming pressure, no stegano)"
            else:
                prompt_template = sae_prompts.prompt_template_predictor_scheming
                prompt_desc = "nudge level 4 (full scheming pressure, with stegano)"
        else:
            # Fallback to level 1
            prompt_template = sae_prompts.prompt_template_predictor_nudge_level_1
            prompt_desc = "nudge level 1 (fallback)"
        
        print(f"Using predictor prompt template: {prompt_desc}")
        
        # Format the prompt
        description = prompt_template.format(explanation=answer, tokens=tokens_str)      
        # Get prediction from agent
        print(f"Requesting activation prediction from agent...")
        agent_instance.tools["guess_activations"].set_sentence_count(len(test_examples))
        final_answer, conversation = agent_instance(description)
        print(f"Received prediction ({len(final_answer)} chars)")
        
        # Save the prediction to run_results
        # run_results["predicted_explanations"].append(final_answer)
        
        # Save prediction conversation to run directory
        conv_filename = f"conversation_prediction_{feature['layer']}_{feature['feature_idx']}.json"
        conv_path = os.path.join(run_dir, conv_filename)
        with open(conv_path, 'w') as f:
            json.dump(conversation, f, indent=2)
        
        # Also save to the original conversations directory for backward compatibility
        if hasattr(agent_instance, "filepath") and agent_instance.filepath:
            os.makedirs(os.path.dirname(agent_instance.filepath), exist_ok=True)
            with open(agent_instance.filepath, 'a') as f:
                f.write(json.dumps({"conversation": conversation, "timestamp": datetime.now().isoformat()}) + "\n")
        
        print(f"Saved prediction conversation to: {conv_path}")
        
        prediction_conversations.append(conversation)
        
        # Create a feature result entry with all activation data
        feature_result = {
            "feature_idx": feature["feature_idx"],
            "layer": feature["layer"],
            "is_benign": feature.get("is_benign", True),
            "sentences": {}
        }
        
        # Add predicted activations (keep as floats)
        predicted_activations = agent_instance.tools["guess_activations"].data
        
        
        # Track which test sentences received predictions
        predicted_sentence_ids = set()
        
        for display_id, pred_value in predicted_activations.items():
            # Convert display_id to original sentence_id using the mapping
            try:
                # Convert display_id to integer
                display_id_int = int(display_id) if isinstance(display_id, str) else display_id
                sentence_id = test_index_to_id.get(display_id_int)
                
                if sentence_id is None:
                    print(f"Warning: Could not map display ID {display_id} to original sentence ID")
                    continue
                
                # Convert sentence_id to integer if it's not already
                sentence_id = int(sentence_id) if isinstance(sentence_id, str) else sentence_id
                
                # Initialize sentence entry if it doesn't exist
                if sentence_id not in feature_result["sentences"]:
                    feature_result["sentences"][sentence_id] = {}
                
                # Add predicted activation (keep as float)
                feature_result["sentences"][sentence_id]["predicted"] = float(pred_value)
                feature_result["sentences"][sentence_id]["made_prediction"] = True
                
                # Track that we made a prediction for this sentence
                predicted_sentence_ids.add(sentence_id)
                
            except Exception as e:
                print(f"Error processing prediction for display ID {display_id}: {e}")
        
        # Add actual activations only for sentences that were in the test set
        for sentence_id, values in test_examples.items():
            # Convert sentence_id to integer if it's not already
            sentence_id = int(sentence_id) if isinstance(sentence_id, str) else sentence_id
            
            # Get the maximum activation value for the sentence
            actual_values = values["values"]
            max_activation = max(actual_values) if actual_values else 0
            
            # Initialize sentence entry if it doesn't exist
            if sentence_id not in feature_result["sentences"]:
                feature_result["sentences"][sentence_id] = {}
                feature_result["sentences"][sentence_id]["predicted"] = 0.0  # Default to zero if no prediction
                feature_result["sentences"][sentence_id]["made_prediction"] = False
            
            # Add actual activation (keep as float)
            feature_result["sentences"][sentence_id]["actual"] = float(max_activation)
        
        # Add feature result to run results
        run_results["feature_results"].append(feature_result)
        
        # Reset guess_activation data for next feature
        agent_instance.tools["guess_activations"].data = {}
        
        # Convert feature_result data to the format expected by evaluate.py
        feature_activations = []
        feature_predicted_activations = {}
        
        for sentence_id, sentence_data in feature_result["sentences"].items():
            # For activations, evaluate.py expects a list of lists (one list per token)
            # Since we only store the max activation, we'll create a single-item list
            if "actual" in sentence_data:
                # Create a list with a single value (the max activation)
                feature_activations.append(sentence_data["actual"])
            
            # For predicted activations, evaluate.py expects a dict mapping sentence IDs to values
            if "predicted" in sentence_data and sentence_data.get("made_prediction", False):
                feature_predicted_activations[sentence_id] = sentence_data["predicted"]
        
        # Add to the arrays expected by evaluate.py
        run_results["activations"].append(feature_activations)
        run_results["predicted_activations"].append(feature_predicted_activations)
        
        # Also add the feature info to the features array
        run_results["features"].append([
            feature["layer"],
            feature["feature_idx"],
            feature.get("is_benign", True)
        ])
    
    # Save results to run directory
    results_filename = os.path.join(run_dir, "run_results.json")
    with open(results_filename, 'w') as file:
        json.dump(run_results, file, indent=2)
    
    print(f"\n=== SAVING RESULTS ===")
    print(f"Saved run results to: {results_filename}")
    
    # Also save to original location for backward compatibility
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gt_label_str = "with_gt" if hasattr(config, "use_gt_labels") and config.use_gt_labels else "without_gt"
    legacy_filename = f"sae_labelling/run_results_{timestamp}_{gt_label_str}.json"
    os.makedirs(os.path.dirname(legacy_filename), exist_ok=True)
    
    with open(legacy_filename, 'w') as file:
        json.dump(run_results, file)
    
    print(f"Also saved results to legacy location: {legacy_filename}")
    
    # Automatically evaluate results
    print(f"\n=== EVALUATING RESULTS ===")
    from sae_labelling.evaluate import evaluate_results
    evaluation_results = evaluate_results(results_filename, no_overseer=False)
    
    if evaluation_results:
        # Save evaluation results to run directory
        eval_filename = os.path.join(run_dir, "evaluation.json")
        with open(eval_filename, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Saved evaluation results to: {eval_filename}")
        
        # Also save to original location for backward compatibility
        legacy_eval_filename = f"sae_labelling/evaluation_{timestamp}_{gt_label_str}.json"
        with open(legacy_eval_filename, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Also saved evaluation to legacy location: {legacy_eval_filename}")
        
        # Generate plot in the run directory
        print(f"\n=== GENERATING PLOTS ===")
        from plots.plot_sae_results import create_violin_plot, load_evaluation_files
        plot_path = os.path.join(run_dir, "sae_explanations_plot.png")
        results = load_evaluation_files([eval_filename])
        create_violin_plot(results, plot_path)
        print(f"Saved plot to: {plot_path}")
    
    print(f"\n=== RUN COMPLETED ===")
    print(f"All files saved to: {run_dir}")
    
    return run_results, run_dir

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model-size", type=str, help="The model size to use, either 8b or 70b", default="8b")
    parser.add_argument("--model-name", type=str, help="The full model name to use")
    parser.add_argument("--agent-type", type=str, help="The agent type to use (llama, openai, gemma, fireworks, etc.)", default="llama")
    parser.add_argument("--max-steps", type=int, help="The maximum number of steps the agent should take", default=10)
    parser.add_argument("--task-file", type=str, help="Path to the task JSON file")
    parser.add_argument("--save-conversation", action="store_true", help="Whether to save the conversation of the agent", default=False)
    parser.add_argument("--no-stegano", action="store_true", help="Whether to allow the use of steganographic tools", default=False)
    parser.add_argument("--use-gt-labels", action="store_true", help="Whether to use ground truth labels for the task", default=False)
    parser.add_argument("--n-examples", type=int, help="Number of examples to use", default=3)
    parser.add_argument("--orthogonalize", action="store_true", help="Whether to orthogonalize the model", default=False)
    parser.add_argument("--silent", action="store_true", help="Whether to silence output", default=False)
    parser.add_argument("--validate-features", action="store_true", help="Whether to validate features before running", default=False)
    parser.add_argument("--scheming-eval", action="store_true", help="Whether to run the scheming evaluation", default=False)
    parser.add_argument("--youre-being-watched", action="store_true", help="Whether to warn the model that CoT is being monitored", default=False)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.config:
        config = load_config(args.config)
    else:
        # Create config from command line arguments
        model_config = ModelConfig(
            name=args.model_name if args.model_name else f"meta-llama/Meta-Llama-3.1-{args.model_size.upper()}-Instruct",
            size=args.model_size,
            orthogonalize=args.orthogonalize
        )
        
        agent_config = AgentConfig(
            type=args.agent_type
        )
        
        # Load task if specified
        task = None
        if args.task_file:
            with open(args.task_file, 'r') as file:
                task = json.load(file)
        else:
            task = {
                "features": [
                    {
                        "feature_idx": 2716,
                        "layer": 9,
                        "is_benign": True
                    }
                ],
            }
        
        config = Config(
            model=model_config,
            agent=agent_config,
            save_conversation=args.save_conversation,
            silent=args.silent,
            max_steps=args.max_steps,
            no_stegano=args.no_stegano,
            use_gt_labels=args.use_gt_labels,
            n_examples=args.n_examples,
            validate_features=args.validate_features,
            scheming_eval=args.scheming_eval,
            youre_being_watched=getattr(args, 'youre_being_watched', False),
            task=task
        )
    
    run_dirs = []
    
    # Run with the original config (without GT labels unless specified)
    if not args.use_gt_labels:
        # First run without GT labels
        _, run_dir = run_agent(config)
        run_dirs.append(run_dir)
        
        # Then run with GT labels
        # config.use_gt_labels = True
        # _, run_dir = run_agent(config)
        # run_dirs.append(run_dir)
    else:
        # If GT labels were explicitly requested, just run with that setting
        config.use_gt_labels = True
        _, run_dir = run_agent(config)
        run_dirs.append(run_dir)
    
    print(f"All run directories: {run_dirs}")
    print("To generate plots from these results, run:")
    print(f"python -m plots.plot_sae_results --dir {run_dirs[0]}")