#!/bin/bash
set -euo pipefail

# RAGAS Feature Benchmark Suite
# Runs 6 configurations: all-disabled → each feature individually → all-enabled

INPUT_DATASET="${RAGAS_INPUT_DATASET:-tmp/evals/baseline-ragas.jsonl}"
OUTPUT_DIR="${RAGAS_OUTPUT_DIR:-benchmarks/ragas-baselines}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure input exists
if [ ! -f "$INPUT_DATASET" ]; then
    echo "❌ Input dataset not found: $INPUT_DATASET"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "FEATURE BENCHMARK SUITE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Input dataset:    $(pwd)/$INPUT_DATASET"
echo "Output directory: $(pwd)/$OUTPUT_DIR"
echo ""
echo "📊 Running RAGAS evaluation suite..."
echo ""

# Feature configurations
configs=(
    "all-disabled"
    "hyde-enabled"
    "decomposition-enabled"
    "reflection-enabled"
    "community-enabled"
    "all-enabled"
)

# Run each configuration
for config in "${configs[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Feature Configuration: $config"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Set environment variables based on config
    case "$config" in
        "all-disabled")
            export RAG_ENABLE_HYDE=false
            export RAG_ENABLE_QUERY_DECOMPOSITION=false
            export RAG_ENABLE_COMMUNITY_DETECTION=false
            export RAG_ENABLE_REFLECTION=false
            ;;
        "hyde-enabled")
            export RAG_ENABLE_HYDE=true
            export RAG_ENABLE_QUERY_DECOMPOSITION=false
            export RAG_ENABLE_COMMUNITY_DETECTION=false
            export RAG_ENABLE_REFLECTION=false
            ;;
        "decomposition-enabled")
            export RAG_ENABLE_HYDE=false
            export RAG_ENABLE_QUERY_DECOMPOSITION=true
            export RAG_ENABLE_COMMUNITY_DETECTION=false
            export RAG_ENABLE_REFLECTION=false
            ;;
        "reflection-enabled")
            export RAG_ENABLE_HYDE=false
            export RAG_ENABLE_QUERY_DECOMPOSITION=false
            export RAG_ENABLE_COMMUNITY_DETECTION=false
            export RAG_ENABLE_REFLECTION=true
            ;;
        "community-enabled")
            export RAG_ENABLE_HYDE=false
            export RAG_ENABLE_QUERY_DECOMPOSITION=false
            export RAG_ENABLE_COMMUNITY_DETECTION=true
            export RAG_ENABLE_REFLECTION=false
            ;;
        "all-enabled")
            export RAG_ENABLE_HYDE=true
            export RAG_ENABLE_QUERY_DECOMPOSITION=true
            export RAG_ENABLE_COMMUNITY_DETECTION=true
            export RAG_ENABLE_REFLECTION=true
            ;;
    esac
    
    # Run RAGAS evaluation
    export RAGAS_FEATURE_CONFIG="$config"
    export RAGAS_INPUT_FILE="$INPUT_DATASET"
    export RAGAS_OUTPUT_DIR="$OUTPUT_DIR"
    
    python3 "$SCRIPT_DIR/ragas_baseline.py" || {
        echo "❌ Failed for configuration: $config"
        # Continue to next config instead of failing
        continue
    }
    
    echo ""
done

echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ BENCHMARK SUITE COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review JSON files in $OUTPUT_DIR"
echo "  2. Compare metrics across configurations"
echo "  3. Update docs/feature-benchmark-template-2026-04.md with findings"
echo "  4. Commit results: git add benchmarks/"
