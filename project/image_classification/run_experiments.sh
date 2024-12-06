#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# Define optimizers to test
optimizers=("adam" "adamw" "sgd" "rmsprop" "adagrad")

# Define experiment configurations
experiments=("no_scheduler_no_warmup" "scheduler_only" "warmup_only")

# Loop through each optimizer and experiment configuration
for optimizer in "${optimizers[@]}"; do
  for experiment in "${experiments[@]}"; do

    # Experiment conditions
    use_scheduler="false"
    use_warmup="false"
    
    # Configure based on experiment type
    if [ "$experiment" = "scheduler_only" ]; then
      use_scheduler="true"
    elif [ "$experiment" = "warmup_only" ]; then
      use_warmup="true"
    fi

    # Run the Python script
    python3 main.py --optimizer "$optimizer" \
                   $( [ "$use_scheduler" = "true" ] && echo "--use_scheduler" ) \
                   $( [ "$use_warmup" = "true" ] && echo "--use_warmup" ) \
                   --num_warmup_steps 100 \
                   | tee "results/${optimizer}_${experiment}.txt"  # Use tee for output

    echo "Completed $optimizer with $experiment"

  done
done
