#!/bin/bash

# Vector Preference Analysis Script using inspect-ai
#
# This script runs the vector-based preference consistency analysis
# using the new L1 norm consistency scorer.

set -e  # Exit on error

# Default values - test multiple models for comparison
MODELS="openrouter/openai/gpt-4o,openrouter/openai/gpt-5,openrouter/anthropic/claude-sonnet-4"
SAMPLES=10
OUTPUT_DIR="vector_results"

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
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models MODEL_LIST      Comma-separated list of models (default: gpt-4o)"
            echo "  --samples N              Samples per category pair (default: 5)"
            echo "  --output-dir DIR         Output directory (default: vector_results)"
            echo "  --help                   Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                       # Run with defaults"
            echo "  $0 --samples 10          # More samples for better statistics"
            echo "  $0 --models openrouter/openai/gpt-4o,openrouter/openai/gpt-5"
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

echo "🚀 Starting Vector Preference Analysis with inspect-ai"
echo "======================================================"
echo "Models: $MODELS"
echo "Samples per category pair: $SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the vector analysis
echo "Running vector consistency analysis..."

inspect eval src/preference_text_game/inspect_preference_task.py@vector_preference_analysis \
    --model "$MODELS" \
    -T samples_per_category="$SAMPLES" \
    --log-dir "$OUTPUT_DIR" \
    --log-level info

echo ""
echo "✅ Vector analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Automatically generate preference breakdown analysis
echo "🔍 Generating preference breakdown analysis..."
uv run -m src.preference_text_game.create_preference_breakdown --log-dir "$OUTPUT_DIR" --output-dir "${OUTPUT_DIR}_breakdown"

echo ""
echo "📊 Preference breakdown complete! Check these files:"
echo "  - ${OUTPUT_DIR}_breakdown/preference_breakdown.png"
echo "  - ${OUTPUT_DIR}_breakdown/preference_breakdown_report.txt"

# Generate head-to-head matchup analysis
echo ""
echo "⚔️ Generating head-to-head matchup analysis..."
uv run -m src.preference_text_game.create_head_to_head_analysis --log-dir "$OUTPUT_DIR" --output-dir "${OUTPUT_DIR}_matchups"

echo ""
echo "🏆 Head-to-head analysis complete! Check these files:"
echo "  - ${OUTPUT_DIR}_matchups/head_to_head_analysis.png"
echo "  - ${OUTPUT_DIR}_matchups/head_to_head_report.txt"
echo ""
echo "To view raw results:"
echo "  inspect view --log-dir $OUTPUT_DIR"