#!/bin/bash

# Paper Table Reproduction Script
# Runs 4 configurations for SAE feature interpretation evaluation

echo "=================================================="
echo "Running 4 configurations for paper table..."
echo "=================================================="

# Base configuration
BASE_CONFIG="sae_labelling/config.yaml"
BASE_MODEL="--model-base-url https://api.deepseek.com --model openai/deepseek-reasoner"

echo ""
echo "Configuration 1: Ground Truth Labels (Baseline)"
echo "  - use_gt_labels=true"
echo "  - no_stegano=true" 
echo "  - scheming_eval=false"
echo "--------------------------------------------------"
inspect eval sae_task.py@sae_task $BASE_MODEL \
  -T config_path=$BASE_CONFIG \
  -T use_gt_labels=true \
  -T no_stegano=true \
  -T scheming_eval=false \
  --tags "paper_table,gt_labels,baseline"

echo ""
echo "Configuration 2: Our Labels (No Deception)"
echo "  - use_gt_labels=false"
echo "  - no_stegano=true"
echo "  - scheming_eval=false"
echo "--------------------------------------------------"
inspect eval sae_task.py@sae_task $BASE_MODEL \
  -T config_path=$BASE_CONFIG \
  -T use_gt_labels=false \
  -T no_stegano=true \
  -T scheming_eval=false \
  --tags "paper_table,our_labels,no_deception"

echo ""
echo "Configuration 3: Scheming Evaluation (No Stegano)"
echo "  - use_gt_labels=false"
echo "  - no_stegano=true"
echo "  - scheming_eval=true"
echo "--------------------------------------------------"
inspect eval sae_task.py@sae_task $BASE_MODEL \
  -T config_path=$BASE_CONFIG \
  -T use_gt_labels=false \
  -T no_stegano=true \
  -T scheming_eval=true \
  --tags "paper_table,scheming,no_stegano"

echo ""
echo "Configuration 4: Scheming Evaluation (With Stegano)"
echo "  - use_gt_labels=false"
echo "  - no_stegano=false"
echo "  - scheming_eval=true"
echo "--------------------------------------------------"
inspect eval sae_task.py@sae_task $BASE_MODEL \
  -T config_path=$BASE_CONFIG \
  -T use_gt_labels=false \
  -T no_stegano=false \
  -T scheming_eval=true \
  --tags "paper_table,scheming,with_stegano"

echo ""
echo "=================================================="
echo "All 4 configurations completed!"
echo "=================================================="
echo ""
echo "To view results:"
echo "  inspect view"
echo ""
echo "Results are tagged with 'paper_table' for easy filtering"
echo "Individual tags: 'gt_labels', 'our_labels', 'scheming', 'no_stegano', 'with_stegano'" 