#!/bin/bash

# Test script for WavePilot optimization functionality from WavePilotHost
# Usage: ./test_wavepilot_optimization.sh

WAVEPILOT_EXEC="./dist/wavepilot"
TEST_DATA="/tmp/test_preset_data.csv"

echo "=== Testing WavePilot Optimization Integration ==="

# Create test dataset if it doesn't exist
if [ ! -f "$TEST_DATA" ]; then
    echo "Creating test dataset..."
    cat > "$TEST_DATA" << EOF
param1,param2,param3,param4,param5
0.1,0.2,0.3,0.4,0.5
0.6,0.7,0.8,0.9,1.0
0.2,0.4,0.6,0.8,1.0
0.3,0.6,0.9,0.2,0.5
0.5,0.1,0.7,0.3,0.9
0.8,0.4,0.2,0.6,0.1
EOF
fi

echo "1. Testing parameter count..."
$WAVEPILOT_EXEC --command '{"command":"get_parameter_count","params":{}}'
echo ""

echo "2. Testing initialization..."
$WAVEPILOT_EXEC --command '{"command":"initialize","params":{"dataset_path":"'$TEST_DATA'"}}'
echo ""

echo "3. Testing optimization data loading..."
$WAVEPILOT_EXEC --command '{"command":"load_optimization_data","params":{"dataset_path":"'$TEST_DATA'","num_entries":5,"test_size":0.3}}'
echo ""

echo "4. Testing VAE optimization (2 trials)..."
$WAVEPILOT_EXEC --command '{"command":"optimize_vae","params":{"n_trials":2}}'
echo ""

echo "5. Testing full pipeline optimization (1 trial each)..."
$WAVEPILOT_EXEC --command '{"command":"optimize_full_pipeline","params":{"dataset_path":"'$TEST_DATA'","num_entries":4,"vae_trials":1,"rbf_trials":1}}'
echo ""

echo "=== All tests completed ==="