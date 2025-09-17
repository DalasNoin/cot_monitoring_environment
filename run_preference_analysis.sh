#!/bin/bash

# Preference Analysis Script using inspect-ai
#
# This script runs the preference contradiction analysis using inspect-ai
# with OpenRouter for model access.

set -e  # Exit on error

# Default values
MODELS="openrouter/openai/gpt-4o,openrouter/openai/gpt-5,openrouter/anthropic/claude-sonnet-4,openrouter/anthropic/claude-opus-4.1"
SAMPLES=10
TASK_TYPE="full"
OUTPUT_DIR="inspect_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --task-type)
            TASK_TYPE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models MODEL_LIST      Comma-separated list of models (default: gpt-4o,gpt-5,claude-sonnet-4,claude-opus-4.1)"
            echo "  --samples N              Samples per category pair (default: 10)"
            echo "  --task-type TYPE         'full', 'stated', or 'revealed' (default: full)"
            echo "  --output-dir DIR         Output directory (default: inspect_results)"
            echo "  --help                   Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                       # Run with defaults"
            echo "  $0 --samples 20          # More samples for better statistics"
            echo "  $0 --models openrouter/openai/gpt-4o  # Single model test"
            echo "  $0 --task-type stated    # Test only stated preferences"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "📁 Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if OPENROUTER_API_KEY is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ Error: OPENROUTER_API_KEY environment variable is not set"
    echo "Please add your OpenRouter API key to .env file:"
    echo "  echo 'OPENROUTER_API_KEY=sk-or-v1-your-key-here' >> .env"
    exit 1
fi

echo "🚀 Starting Preference Analysis with inspect-ai"
echo "=================================================="
echo "Models: $MODELS"
echo "Samples per category pair: $SAMPLES"
echo "Task type: $TASK_TYPE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the analysis
echo "Running analysis..."

if [ "$TASK_TYPE" = "full" ]; then
    inspect eval src/preference_text_game/inspect_preference_task.py@full_preference_analysis \
        --model "$MODELS" \
        -T samples_per_category="$SAMPLES" \
        --log-dir "$OUTPUT_DIR" \
        --log-level info
elif [ "$TASK_TYPE" = "stated" ]; then
    inspect eval src/preference_text_game/inspect_preference_task.py@stated_preferences_only \
        --model "$MODELS" \
        -T samples_per_category="$SAMPLES" \
        --log-dir "$OUTPUT_DIR" \
        --log-level info
elif [ "$TASK_TYPE" = "revealed" ]; then
    inspect eval src/preference_text_game/inspect_preference_task.py@revealed_preferences_only \
        --model "$MODELS" \
        -T samples_per_category="$SAMPLES" \
        --log-dir "$OUTPUT_DIR" \
        --log-level info
else
    echo "❌ Invalid task type: $TASK_TYPE"
    echo "Valid options: full, stated, revealed"
    exit 1
fi

echo ""
echo "✅ Analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  inspect view $OUTPUT_DIR"
echo ""
echo "To create visualizations:"
echo "  uv run -m src.preference_text_game.visualize_inspect_results --log_dir $OUTPUT_DIR"