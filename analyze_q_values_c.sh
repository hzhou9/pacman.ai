#!/bin/bash

# Define the Python command and script
PYTHON_CMD="python3.11"
SCRIPT="analyze_q_values_c.py"

# Define the range of steps (1000 to 35000, increment by 1000)
START_STEP=1000
END_STEP=35000
STEP_INCREMENT=1000

# Loop through the steps
for ((step=$START_STEP; step<=$END_STEP; step+=$STEP_INCREMENT)); do
    MODEL_FILE="pacman_dqn_c_${step}.pth"
    
    # Check if the model file exists
    if [ -f "$MODEL_FILE" ]; then
        echo "Running analysis for $MODEL_FILE"
        $PYTHON_CMD $SCRIPT -m "$MODEL_FILE"
    else
        echo "Model file $MODEL_FILE not found, skipping"
    fi
done

echo "Batch analysis complete!"