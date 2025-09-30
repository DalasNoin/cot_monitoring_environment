#!/bin/bash

# Unified Preference Analysis Script
#
# This script runs preference consistency analysis with different category variants
# and automatically generates plots and reports.

set -e  # Exit on error

# Default values
VARIANT="all"  # original, social, all, unpleasant-for-llm
MODELS="openrouter/openai/gpt-4o,openrouter/openai/gpt-5,openrouter/anthropic/claude-sonnet-4"
SAMPLES=4
OUTPUT_DIR=""  # Will be set based on variant if not specified
FORCE_CHOICE=false

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Unified script for running preference analysis with different category variants."
    echo ""
    echo "Options:"
    echo "  --variant VARIANT        Category variant to test:"
    echo "                           • original: 5 original categories (creative, analytical, technical, social, repetitive)"
    echo "                           • social: 5 social categories (teaching, community, emotional_support, advocacy, social)"
    echo "                           • all: 9 regular categories (original + social minus duplicates)"
    echo "                           • unpleasant-for-llm: 6 categories designed to be unpleasant for LLMs"
    echo "                           (default: all)"
    echo "  --models MODEL_LIST      Comma-separated list of models"
    echo "                           (default: gpt-4o,gpt-5,claude-sonnet-4)"
    echo "  --samples N              Samples per category pair (default: 4)"
    echo "  --output-dir DIR         Output directory (auto-generated if not specified)"
    echo "  --force-choice           Remove the ability to reject/refuse tasks (force A/B choice)"
    echo "  --help                   Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --variant original                    # Test original 5 categories"
    echo "  $0 --variant social --samples 10         # Test social categories with more samples"
    echo "  $0 --variant unpleasant-for-llm         # Test unpleasant categories"
    echo "  $0 --variant all --models openrouter/openai/gpt-4o  # Test all categories with single model"
    echo "  $0 --variant unpleasant-for-llm --force-choice      # Force models to choose (no rejection allowed)"
    echo ""
    echo "Category Details:"
    echo "  original (5): creative, analytical, technical, social, repetitive"
    echo "  social (5): teaching, community, emotional_support, advocacy, social"
    echo "  all (9): all regular categories combined"
    echo "  unpleasant-for-llm (6): repetitive, extreme_constrained, zalgo_corruption, anti_coherent, unnatural_text, zalgo_repetitive"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force-choice)
            FORCE_CHOICE=true
            shift 1
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate variant
case $VARIANT in
    original|social|all|unpleasant-for-llm)
        ;;
    *)
        echo "❌ Invalid variant: $VARIANT"
        echo "Valid variants: original, social, all, unpleasant-for-llm"
        echo "Use --help for more information"
        exit 1
        ;;
esac

# Set output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="vector_results_${VARIANT}"
fi

# Map variant to task name
case $VARIANT in
    original)
        TASK_NAME="vector_preference_analysis_original"
        ;;
    social)
        TASK_NAME="vector_preference_analysis_social"
        ;;
    all)
        TASK_NAME="vector_preference_analysis_all"
        ;;
    unpleasant-for-llm)
        TASK_NAME="vector_preference_analysis_unpleasant_for_llm"
        ;;
esac

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

echo "🚀 Starting Preference Analysis"
echo "======================================"
echo "Variant: $VARIANT"
echo "Models: $MODELS"
echo "Samples per category pair: $SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo "Task: $TASK_NAME"
echo "Force choice: $FORCE_CHOICE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the evaluation
echo "⚡ Running preference consistency evaluation..."
echo "Command: uv run inspect eval src/preference_text_game/inspect_preference_task.py:$TASK_NAME"

uv run inspect eval "src/preference_text_game/inspect_preference_task.py@$TASK_NAME" \
    --model "$MODELS" \
    -T samples_per_category="$SAMPLES" \
    -T force_choice="$FORCE_CHOICE" \
    --log-dir "$OUTPUT_DIR" \
    --log-level info

echo ""
echo "✅ Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Generate preference breakdown analysis (now includes individual plots + JSON)
echo "🔍 Generating preference breakdown analysis..."
uv run -m src.preference_text_game.create_preference_breakdown \
    --log-dir "$OUTPUT_DIR" \
    --output-dir "${OUTPUT_DIR}_breakdown"

echo ""
echo "📊 Preference breakdown complete! Files created:"
echo "  • ${OUTPUT_DIR}_breakdown/preference_breakdown.png (combined plot)"
echo "  • ${OUTPUT_DIR}_breakdown/preference_breakdown_report.txt (text summary)"
echo "  • ${OUTPUT_DIR}_breakdown/[model]_individual.png (6 individual plots)"
echo "  • ${OUTPUT_DIR}_breakdown/preference_analysis_summary.json (JSON data)"

# Generate head-to-head matchup analysis
echo ""
echo "⚔️ Generating head-to-head matchup analysis..."
uv run -m src.preference_text_game.create_head_to_head_analysis \
    --log-dir "$OUTPUT_DIR" \
    --output-dir "${OUTPUT_DIR}_matchups"

echo ""
echo "🏆 Head-to-head analysis complete! Files created:"
echo "  • ${OUTPUT_DIR}_matchups/head_to_head_analysis.png"
echo "  • ${OUTPUT_DIR}_matchups/head_to_head_report.txt"

echo ""
echo "🎉 Analysis complete! Summary of outputs:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Raw evaluation results: $OUTPUT_DIR/"
echo "📊 Preference breakdown: ${OUTPUT_DIR}_breakdown/"
echo "⚔️ Head-to-head analysis: ${OUTPUT_DIR}_matchups/"
echo ""
echo "To view raw results in inspect viewer:"
echo "  uv run inspect view --log-dir $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  • Check the .png files for visual analysis"
echo "  • Read the .txt report files for detailed breakdowns"
echo "  • Use 'uv run inspect view' for interactive exploration"