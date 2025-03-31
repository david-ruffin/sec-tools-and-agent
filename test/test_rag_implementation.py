#!/usr/bin/env python3
"""
Test script specifically for validating RAG implementations in the SEC Agent.
Tests different document sizes and types, measuring metrics about RAG processing.
"""

import os
import sys
import subprocess
import time
import json
import re
from datetime import datetime

# Single test case that will trigger RAG
TEST_CASE = {
    "name": "Apple Business Overview",
    "question": "Summarize Apple's business model from their latest 10-K",
    "section_code": "1",  # Business section
}

def extract_rag_metrics(output):
    """Extract RAG processing metrics from the output log"""
    metrics = {
        "original_length": None,
        "processed_length": None,
        "compression_ratio": None,
        "num_chunks": None,
        "processing_time": None,
        "rag_triggered": False
    }
    
    # Check if RAG was triggered
    if "Processing text with RAG" in output:
        metrics["rag_triggered"] = True
        
        # Extract original length
        match = re.search(r"Processing text with RAG \(length: (\d+)\)", output)
        if match:
            metrics["original_length"] = int(match.group(1))
            
        # Extract number of chunks
        match = re.search(r"Created (\d+) chunks from text content", output)
        if match:
            metrics["num_chunks"] = int(match.group(1))
            
        # Extract processed length from response
        match = re.search(r"processed_length': (\d+)", output)
        if match:
            metrics["processed_length"] = int(match.group(1))
            
        # Calculate compression ratio if we have both lengths
        if metrics["original_length"] and metrics["processed_length"]:
            metrics["compression_ratio"] = metrics["processed_length"] / metrics["original_length"]
            
        # Try to extract processing time
        start_time = None
        end_time = None
        
        # Find the RAG processing start and end times
        for line in output.split('\n'):
            if "Processing text with RAG" in line:
                match = re.search(r"(\d+:\d+:\d+)", line)
                if match:
                    start_time = match.group(1)
            elif "RAG processing complete" in line:
                match = re.search(r"(\d+:\d+:\d+)", line)
                if match:
                    end_time = match.group(1)
        
        # If we found both times, calculate time difference
        if start_time and end_time:
            # Basic time difference in seconds
            start_parts = [int(x) for x in start_time.split(':')]
            end_parts = [int(x) for x in end_time.split(':')]
            
            start_seconds = start_parts[0] * 3600 + start_parts[1] * 60 + start_parts[2]
            end_seconds = end_parts[0] * 3600 + end_parts[1] * 60 + end_parts[2]
            
            metrics["processing_time"] = end_seconds - start_seconds
    
    return metrics

def run_test(test_case):
    """Run a single RAG test case"""
    print(f"\n{'='*80}")
    print(f"TESTING RAG: {test_case['name']}")
    print(f"Question: {test_case['question']}")
    print(f"{'='*80}")
    
    # Construct the command to run the SEC agent with the question
    cmd = [sys.executable, "sec.py", "--query", test_case["question"]]
    
    # Record start time
    start_time = time.time()
    
    # Run the command and capture output
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        print("Output (truncated to first 300 chars):")
        print(output[:300] + "...\n")
        status = "SUCCESS"
    except subprocess.CalledProcessError as e:
        output = e.stdout + "\n" + e.stderr
        print(f"ERROR (exit code {e.returncode}):\n{output[:300]}...")
        status = "ERROR"
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Extract RAG metrics
    metrics = extract_rag_metrics(output)
    metrics["total_time"] = elapsed_time
    
    print(f"RAG Metrics:")
    print(f"  RAG Triggered: {metrics['rag_triggered']}")
    if metrics["rag_triggered"]:
        print(f"  Original Content Length: {metrics['original_length']} chars")
        print(f"  Processed Content Length: {metrics['processed_length']} chars")
        print(f"  Compression Ratio: {metrics['compression_ratio']:.2f}")
        print(f"  Number of Chunks: {metrics['num_chunks']}")
        print(f"  RAG Processing Time: {metrics['processing_time']} seconds")
    print(f"  Total Query Time: {elapsed_time:.2f} seconds")
    
    return {
        "name": test_case["name"],
        "question": test_case["question"],
        "status": status,
        "metrics": metrics
    }

def main():
    """Run the RAG test case"""
    print(f"Starting RAG implementation test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run the test
    result = run_test(TEST_CASE)
    
    # Save results to JSON file
    with open("rag_test_results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nTest completed. Results saved to rag_test_results.json")

if __name__ == "__main__":
    main() 