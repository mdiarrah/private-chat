#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <number_of_iterations>"
    exit 1
fi

# Number of iterations specified as an argument
iterations="$1"
speed="$2"

# Function to perform a request and print the time taken in seconds with request number
perform_request() {
    local request_number="$1"
    local exp_speed="$2"
    local start_time=$(date +%s%N)  # Get start time in nanoseconds
    curl -X POST "http://51.79.102.44:8181/api/v1/generate" -d "inputs=hi" -d "max_new_tokens=20" -d "model=petals-team/StableBeluga2"
    local end_time=$(date +%s%N)    # Get end time in nanoseconds
    local elapsed_time=$(( (end_time - start_time) / 1000000000 ))  # Calculate elapsed time in seconds
    echo "Time taken for request $request_number: $elapsed_time seconds"
    echo "req $request_number,$elapsed_time" >> "result."$speed".r2.csv"
}

# Perform requests in a loop
for ((i=1; i<=$iterations; i++)); do
    perform_request $i $speed &
    sleep $speed
done

# Wait for all background processes to finish
wait
