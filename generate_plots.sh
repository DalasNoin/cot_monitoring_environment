#!/bin/bash

# Convenience script to regenerate all preference analysis plots from existing inspect-ai results
# This script calls the existing modular plotting scripts without re-running experiments

set -e

# Default values
LOG_DIR="vector_results"
OUTPUT_PREFIX=""  # Will be set based on log_dir if not specified
PLOTS="all"  # breakdown,head-to-head,comparison,all
TOPIC_FILTER=""  # Filter for specific topics/variants (e.g. unpleasant-for-llm, original)

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Generate preference analysis plots from existing inspect-ai evaluation results."
    echo "This script calls the existing plotting modules without re-running experiments."
    echo ""
    echo "Options:"
    echo "  --log-dir DIR            Directory containing inspect eval logs (*.eval files)"
    echo "                           (default: vector_results)"
    echo "  --output-prefix PREFIX   Prefix for output directories (no timestamp added)"
    echo "                           (default: {log_dir}_plots)"
    echo "  --plots TYPE             Which plots to generate:"
    echo "                           • breakdown: Preference breakdown pie charts"
    echo "                           • head-to-head: Win/loss matchup matrices"
    echo "                           • comparison: Model comparison charts"
    echo "                           • all: Generate all plot types (default)"
    echo "  --topic FILTER           Filter eval files by topic/variant:"
    echo "                           • unpleasant-for-llm: Only unpleasant categories"
    echo "                           • original: Only original 5 categories"
    echo "                           • social: Only social categories"
    echo "                           • all: Only 'all' category variant"
    echo "                           If not specified, uses all available eval files"
    echo "  --help                   Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Generate all plots from all experiments"
    echo "  $0 --topic unpleasant-for-llm              # Only plot unpleasant category results"
    echo "  $0 --topic original --plots breakdown       # Only original categories, breakdown only"
    echo "  $0 --plots head-to-head,comparison          # Generate specific plot types"
    echo "  $0 --output-prefix my_analysis              # Custom output folder (overwrites existing)"
    echo ""
    echo "Output Structure:"
    echo "  {output_prefix}_breakdown/            # Preference breakdown plots"
    echo "  {output_prefix}_head_to_head/         # Head-to-head matchup plots"
    echo "  {output_prefix}_comparison/           # Model comparison plots"
    echo ""
    echo "Note: This script uses the LATEST experimental run automatically."
    echo "Multiple .eval files from different timestamps? Only the most recent is used."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --output-prefix)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        --plots)
            PLOTS="$2"
            shift 2
            ;;
        --topic)
            TOPIC_FILTER="$2"
            shift 2
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

# Set default output prefix if not specified (no timestamp)
if [ -z "$OUTPUT_PREFIX" ]; then
    OUTPUT_PREFIX="${LOG_DIR}_plots"
fi

# Validate log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ Log directory not found: $LOG_DIR"
    exit 1
fi

# Apply topic filter if specified
EVAL_PATTERN="*.eval"
TOPIC_INFO=""
if [ -n "$TOPIC_FILTER" ]; then
    case $TOPIC_FILTER in
        unpleasant-for-llm)
            EVAL_PATTERN="*unpleasant*.eval"
            TOPIC_INFO=" (unpleasant-for-llm topic)"
            ;;
        original)
            EVAL_PATTERN="*vector-preference-analysis*.eval"
            # Exclude variants
            FILTERED_FILES=$(find "$LOG_DIR" -name "$EVAL_PATTERN" 2>/dev/null | grep -v "unpleasant" | grep -v "social" | grep -v "_all")
            TOPIC_INFO=" (original 5 categories)"
            ;;
        social)
            EVAL_PATTERN="*social*.eval"
            TOPIC_INFO=" (social categories)"
            ;;
        all)
            EVAL_PATTERN="*_all*.eval"
            TOPIC_INFO=" (all categories variant)"
            ;;
        *)
            echo "❌ Unknown topic filter: $TOPIC_FILTER"
            echo "Valid options: unpleasant-for-llm, original, social, all"
            exit 1
            ;;
    esac
fi

# Check if any eval files exist (with filtering)
if [ "$TOPIC_FILTER" = "original" ]; then
    EVAL_FILES=$(echo "$FILTERED_FILES" | wc -l)
    if [ -z "$FILTERED_FILES" ]; then
        EVAL_FILES=0
    fi
else
    EVAL_FILES=$(find "$LOG_DIR" -name "$EVAL_PATTERN" 2>/dev/null | wc -l)
fi

if [ "$EVAL_FILES" -eq 0 ]; then
    echo "❌ No .eval files found in $LOG_DIR matching pattern: $EVAL_PATTERN"
    echo "Run an experiment first or check the log directory path and topic filter"
    exit 1
fi

echo "🎨 Generating Preference Analysis Plots"
echo "======================================="
echo "Log directory: $LOG_DIR"
echo "Topic filter: ${TOPIC_FILTER:-all}${TOPIC_INFO}"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Plot types: $PLOTS"
echo "Eval files found: $EVAL_FILES"
echo ""
echo "ℹ️  Note: Using LATEST experimental run automatically"
echo "   (Multiple .eval files with different timestamps → only most recent used)"
echo ""

# Parse plot types to generate
IFS=',' read -ra PLOT_TYPES <<< "$PLOTS"

# Function to run a plotting module
run_plot_module() {
    local module_name=$1
    local output_suffix=$2
    local description=$3

    echo "🔍 Generating $description..."
    local output_dir="${OUTPUT_PREFIX}_${output_suffix}"

    # Create output directory and remove existing files to overwrite
    mkdir -p "$output_dir"
    rm -f "$output_dir"/*.png "$output_dir"/*.txt 2>/dev/null || true

    # Create filtered log dir if topic filter is specified
    local actual_log_dir="$LOG_DIR"
    local temp_log_dir=""

    if [ -n "$TOPIC_FILTER" ]; then
        # Create temporary directory with filtered files
        temp_log_dir="/tmp/generate_plots_filtered_$$"
        mkdir -p "$temp_log_dir"

        if [ "$TOPIC_FILTER" = "original" ]; then
            # Use pre-filtered files for original
            if [ -n "$FILTERED_FILES" ]; then
                echo "$FILTERED_FILES" | while read -r file; do
                    if [ -n "$file" ]; then
                        cp "$file" "$temp_log_dir/"
                    fi
                done
            fi
        else
            # Copy files matching pattern
            find "$LOG_DIR" -name "$EVAL_PATTERN" -exec cp {} "$temp_log_dir/" \; 2>/dev/null
        fi

        actual_log_dir="$temp_log_dir"
    fi

    # Run the plotting module
    local success=0
    if uv run -m "src.preference_text_game.$module_name" \
        --log-dir "$actual_log_dir" \
        --output-dir "$output_dir"; then
        echo "✅ $description complete! Files saved to: $output_dir/"
        success=1
    else
        echo "❌ Failed to generate $description"
    fi

    # Cleanup temporary directory
    if [ -n "$temp_log_dir" ] && [ -d "$temp_log_dir" ]; then
        rm -rf "$temp_log_dir"
    fi

    echo ""
    return $((1 - success))
}

# Generate requested plots
SUCCESS_COUNT=0
TOTAL_COUNT=0

# Handle "all" by expanding to individual plot types
if [[ "$PLOTS" == "all" ]]; then
    PLOT_TYPES=("breakdown" "head-to-head" "comparison")
fi

for plot_type in "${PLOT_TYPES[@]}"; do
    plot_type=$(echo "$plot_type" | tr '[:upper:]' '[:lower:]' | xargs) # normalize and trim

    case $plot_type in
        breakdown)
            TOTAL_COUNT=$((TOTAL_COUNT + 1))
            if run_plot_module "create_preference_breakdown" "breakdown" "preference breakdown analysis"; then
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            fi
            ;;
        head-to-head)
            TOTAL_COUNT=$((TOTAL_COUNT + 1))
            if run_plot_module "create_head_to_head_analysis" "head_to_head" "head-to-head matchup analysis"; then
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            fi
            ;;
        comparison)
            TOTAL_COUNT=$((TOTAL_COUNT + 1))
            if run_plot_module "create_model_comparison" "comparison" "model comparison analysis"; then
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            fi
            ;;
        *)
            echo "⚠️ Unknown plot type: $plot_type (skipping)"
            ;;
    esac
done

echo "🎉 Plot Generation Complete!"
echo "=============================="
echo "Generated: $SUCCESS_COUNT/$TOTAL_COUNT plot types successfully"

if [ "$SUCCESS_COUNT" -gt 0 ]; then
    echo ""
    echo "📁 Output directories:"
    find . -maxdepth 1 -name "${OUTPUT_PREFIX}_*" -type d 2>/dev/null | sort || true
    echo ""
    echo "💡 To view plots: open the .png files in the output directories"
fi

if [ "$SUCCESS_COUNT" -ne "$TOTAL_COUNT" ]; then
    exit 1
fi