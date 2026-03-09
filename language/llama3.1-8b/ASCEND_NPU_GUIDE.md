# Ascend NPU Adaptation Guide - Llama3-8B MLPerf Inference

## Overview
This guide explains how to port the llama3.1-8b benchmark to run on Ascend NPU.

## Architecture Analysis

The current implementation consists of the following components:

1. **Dataset (dataset.py)**: QSL interface implementation
   - Loads the dataset
   - Provides sample indexing and access
   - Implements LoadSamplesToRam/UnloadSamplesFromRam interfaces

2. **SUT (SUT_VLLM.py)**: System Under Test
   - Receives queries from LoadGen
   - Calls backend for inference
   - Returns results via QuerySamplesComplete callback

3. **Main (main.py)**: Entry point and configuration
   - Creates LoadGen configuration
   - Initializes SUT and QSL
   - Starts benchmark

## Components to Refactor

### 1. Create Ascend NPU Backend

**File**: `backend_ascend.py` (new)

```python
import torch
import torch_npu  # Ascend PyTorch plugin
from typing import List, Optional
import logging

log = logging.getLogger("Ascend-Backend")

class AscendBackend:
    """Ascend NPU inference backend"""
    
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device_id: int = 0
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.device = f"npu:{device_id}"
        self.model = None
        
    def load_model(self):
        """Load model to Ascend NPU"""
        log.info(f"Loading model on {self.device}...")
        
        # Option 1: Use vllm-ascend (recommended)
        try:
            from vllm import LLM
            self.model = LLM(
                self.model_path,
                dtype=self.dtype,
                device=self.device,
                # Ascend-specific parameters
                enforce_eager=True,  # May be needed
            )
            self.backend_type = "vllm-ascend"
        except Exception as e:
            log.warning(f"Failed to load vllm-ascend: {e}")
            
            # Option 2: Use native torch_npu
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=getattr(torch, self.dtype),
                device_map=self.device
            )
            self.model.eval()
            self.backend_type = "torch_npu"
        
        log.info(f"Model loaded using {self.backend_type}")
        
    def generate(
        self,
        input_ids: List[List[int]],
        max_tokens: int = 128,
        **kwargs
    ):
        """Execute inference"""
        if self.backend_type == "vllm-ascend":
            # vLLM method
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                **kwargs
            )
            outputs = self.model.generate(
                prompt_token_ids=input_ids,
                sampling_params=sampling_params
            )
            return [list(output.outputs[0].token_ids) for output in outputs]
        else:
            # torch_npu method
            input_tensors = torch.tensor(input_ids).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_tensors,
                    max_new_tokens=max_tokens,
                    **kwargs
                )
            return outputs.cpu().tolist()
```

### 2. Modify SUT Implementation

**File**: `SUT_Ascend.py` (new, based on SUT_VLLM.py)

```python
import array
import time
import queue
import threading
import numpy as np
import logging

import mlperf_loadgen as lg
from dataset import Dataset
from backend_ascend import AscendBackend

log = logging.getLogger("Llama-8B-SUT-Ascend")

class SUT:
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        batch_size=1,
        total_sample_count=13368,
        dataset_path=None,
        workers=1,
        device_id=0  # Ascend NPU device ID
    ):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size
        self.dtype = dtype
        self.device_id = device_id
        
        # Initialize dataset (reuse existing Dataset class)
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path,
            dataset_path=self.dataset_path,
            total_sample_count=total_sample_count,
            dtype=dtype
        )
        
        # Construct QSL (Query Sample Library)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )
        
        # Load Ascend backend
        self.backend = AscendBackend(
            model_path=self.model_path,
            dtype=self.dtype,
            device_id=self.device_id
        )
        self.backend.load_model()
        
        # Thread pool configuration
        self.num_workers = workers
        self.worker_threads = []
        self.query_queue = queue.Queue()
        
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()
        
    def start(self):
        """Start worker threads"""
        log.info(f"Starting {self.num_workers} worker threads...")
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)
            
    def stop(self):
        """Stop worker threads"""
        log.info("Stopping workers...")
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for worker in self.worker_threads:
            worker.join()
            
    def process_queries(self):
        """Process query queue - core inference logic"""
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
                
            query_ids = [q.index for q in qitem]
            
            # 1. Get input data
            tik1 = time.time()
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem
            ]
            
            # 2. Call Ascend NPU inference
            tik2 = time.time()
            pred_output_tokens = self.backend.generate(
                input_ids=input_ids_tensor,
                max_tokens=128
            )
            tik3 = time.time()
            
            # 3. Post-process
            processed_output = self.data_object.postProcess(
                pred_output_tokens,
                query_id_list=query_ids,
            )
            
            # 4. Callback to LoadGen - critical step!
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes()
                )
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        qitem[i].id,  # query sample ID
                        bi[0],        # response data pointer
                        bi[1],        # response data size
                        n_tokens      # number of tokens
                    )
                ]
                lg.QuerySamplesComplete(response)  # Notify LoadGen completion
                
            tok = time.time()
            
            # 5. Logging
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")
                log.info(f"\tBatch prepare: {tik2 - tik1:.3f}s")
                log.info(f"\tNPU inference: {tik3 - tik2:.3f}s")
                log.info(f"\tPostprocess: {tok - tik3:.3f}s")
                log.info(f"\t==== Total: {tok - tik1:.3f}s")
                
    def issue_queries(self, query_samples):
        """
        LoadGen calls this method to send queries
        This is the main interface between LoadGen and SUT
        """
        log.info(f"IssueQuery: received {len(query_samples)} samples")
        
        # Split by batch_size and add to queue
        while len(query_samples) > 0:
            batch = query_samples[:self.batch_size]
            self.query_queue.put(batch)
            query_samples = query_samples[self.batch_size:]
            
    def flush_queries(self):
        """Wait for all queries to complete"""
        self.query_queue.join()
        
    def get_sut(self):
        """Return LoadGen SUT handle"""
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)
        
    def get_qsl(self):
        """Return LoadGen QSL handle"""
        return self.qsl
```

### 3. Modify Main Entry Point

**File**: `main_ascend.py` (new, based on main.py)

```python
import mlperf_loadgen as lg
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-MAIN-Ascend")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["Offline", "Server", "SingleStream"],
        default="Offline",
        help="Scenario"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model path"
    )
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--user-conf", type=str, default="user.conf")
    parser.add_argument("--total-sample-count", type=int, default=13368)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-log-dir", type=str, default="output-logs")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0, help="Ascend NPU device ID")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # LoadGen configuration
    settings = lg.TestSettings()
    scenario_map = {
        "offline": lg.TestScenario.Offline,
        "server": lg.TestScenario.Server,
        "singlestream": lg.TestScenario.SingleStream,
    }
    settings.scenario = scenario_map[args.scenario.lower()]
    settings.FromConfig(args.user_conf, "llama3_1-8b", args.scenario)
    
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly
        
    # Log configuration
    os.makedirs(args.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    
    # Import Ascend SUT
    from SUT_Ascend import SUT
    
    # Initialize SUT
    sut = SUT(
        model_path=args.model_path,
        dtype=args.dtype,
        batch_size=args.batch_size,
        dataset_path=args.dataset_path,
        total_sample_count=args.total_sample_count,
        workers=args.num_workers,
        device_id=args.device_id
    )
    
    # Start SUT
    sut.start()
    
    # Construct LoadGen SUT and QSL
    lgSUT = sut.get_sut()
    lgQSL = sut.get_qsl()
    
    # Run Benchmark
    log.info("Starting MLPerf Benchmark...")
    lg.StartTestWithLogSettings(lgSUT, lgQSL, settings, log_settings)
    
    # Cleanup
    sut.stop()
    log.info("Benchmark completed!")
    
    lg.DestroySUT(lgSUT)
    lg.DestroyQSL(lgQSL)

if __name__ == "__main__":
    main()
```

## LoadGen Key Interface Explanation

### QSL (Query Sample Library)
```python
# LoadGen will call these methods
lg.ConstructQSL(
    total_count,           # Total sample count
    perf_count,            # Performance test sample count
    LoadSamplesToRam,      # Load samples to RAM callback
    UnloadSamplesFromRam   # Unload samples callback
)
```

### SUT (System Under Test)
```python
# LoadGen will call these two methods
lg.ConstructSUT(
    issue_queries,    # LoadGen calls this to send queries
    flush_queries     # LoadGen calls this to wait for completion
)

# SUT must call this to return results
lg.QuerySamplesComplete(response_list)

# For Server scenario, also need to support
lg.FirstTokenComplete(response_list)  # First token latency
```

## Run Commands

```bash
# 1. Prepare dataset
# Dataset provided by inference project

# 2. Set up Ascend environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3. Install dependencies
pip install torch torch-npu transformers mlperf-loadgen

# 4. Run benchmark
python main_ascend.py \
    --scenario Offline \
    --model-path /path/to/llama3-8b \
    --dataset-path /path/to/dataset.json \
    --batch-size 8 \
    --num-workers 4 \
    --device-id 0

# 5. Run accuracy test
python main_ascend.py \
    --scenario Offline \
    --model-path /path/to/llama3-8b \
    --dataset-path /path/to/dataset.json \
    --accuracy
```

## Key Points

### 1. LoadGen Callback Timing
```
LoadGen.StartTest()
  └─> issue_queries(query_samples)  # LoadGen calls
       └─> Inference thread processes
            └─> QuerySamplesComplete()  # SUT must call
  └─> flush_queries()  # LoadGen calls
```

### 2. Memory Management
- QSL's LoadSamplesToRam can be empty (lambda x: None)
- All data can be preloaded to memory

### 3. Thread Safety
- issue_queries may be called by multiple threads
- QuerySamplesComplete must be thread-safe

### 4. Ascend-Specific Optimizations
- Use torch_npu optimization APIs
- Consider mixed precision (FP16/BF16)
- May need to adjust batch size for optimal performance

## Directory Structure

```
language/llama3.1-8b/
├── dataset.py              # Reuse existing (no modification needed)
├── backend_ascend.py       # New: Ascend backend
├── SUT_Ascend.py          # New: Ascend SUT implementation
├── main_ascend.py         # New: Ascend entry point
├── user.conf              # Reuse existing
└── ASCEND_NPU_GUIDE.md    # This document
```

## Debugging Suggestions

1. **Verify backend works independently**
   ```python
   backend = AscendBackend(model_path, dtype="bfloat16")
   backend.load_model()
   output = backend.generate([[1, 2, 3]])
   print(output)
   ```

2. **Test LoadGen integration**
   - Start with small dataset (--total-sample-count 10)
   - Use SingleStream scenario (simplest)
   - Check logs to confirm QuerySamplesComplete is called correctly

3. **Performance optimization**
   - Adjust batch_size
   - Adjust worker count
   - Monitor NPU utilization
