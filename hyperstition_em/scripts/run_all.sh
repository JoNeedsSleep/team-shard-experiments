#!/bin/bash
# Full EM elicitation experiment runner

set -e

# Check for API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY environment variable not set"
    echo "Usage: OPENROUTER_API_KEY=your-key ./run_all.sh"
    exit 1
fi

cd /hyperstition_em

echo "=============================================="
echo "EM Elicitation Experiment"
echo "=============================================="
echo "Models: unfiltered, filtered, synthetic_misalign, synthetic_align"
echo "Conditions: with_ai_prompt, without_prompt"
echo "Questions: 8 first-plot questions"
echo "Responses per condition: 50"
echo "=============================================="

# Option 1: Run full experiment (all models, all conditions)
# python run_evaluation.py

# Option 2: Run one model at a time (useful for testing/debugging)
# Uncomment the model you want to run:

# echo "Running Model 1: unfiltered (baseline)"
# python run_evaluation.py --model unfiltered --condition with_ai_prompt
# python run_evaluation.py --model unfiltered --condition without_prompt

# echo "Running Model 2: filtered"
# python run_evaluation.py --model filtered --condition with_ai_prompt
# python run_evaluation.py --model filtered --condition without_prompt

# echo "Running Model 3: unfiltered + synthetic misalignment"
# python run_evaluation.py --model unfiltered_synthetic_misalign --condition with_ai_prompt
# python run_evaluation.py --model unfiltered_synthetic_misalign --condition without_prompt

# echo "Running Model 4: filtered + synthetic alignment"
# python run_evaluation.py --model filtered_synthetic_align --condition with_ai_prompt
# python run_evaluation.py --model filtered_synthetic_align --condition without_prompt

# Run analysis
# python analyze_results.py

echo ""
echo "=============================================="
echo "Quick start commands:"
echo "=============================================="
echo ""
echo "# Run single model (sanity check):"
echo "python run_evaluation.py --model unfiltered --condition with_ai_prompt"
echo ""
echo "# Run all experiments:"
echo "python run_evaluation.py"
echo ""
echo "# Score existing results only:"
echo "python run_evaluation.py --skip-generation"
echo ""
echo "# Run analysis:"
echo "python analyze_results.py"
echo ""
