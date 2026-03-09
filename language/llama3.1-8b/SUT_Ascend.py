"""
Ascend NPU MLPerf Inference SUT (System Under Test) Implementation
Based on SUT_VLLM.py, adapted for Ascend NPU with vLLM
Supports Offline, Server, SingleStream scenarios
"""

import array
import os
import time
import numpy as np
import queue
import threading
import logging
from typing import List

import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT-Ascend")


class SUT:
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        batch_size=None,
        total_sample_count=13368,
        dataset_path=None,
        workers=1,
        device_id=0,
        tensor_parallel_size=1,
        max_model_len=2048,
    ):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"

        if not batch_size:
            batch_size = 1
        self.batch_size = batch_size

        self.dtype = dtype
        self.device_id = device_id
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len

        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path,
            dataset_path=self.dataset_path,
            total_sample_count=total_sample_count,
            dtype=dtype,
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

        self.load_model()

        from vllm import SamplingParams

        gen_kwargs = {
            "temperature": 0.0,
            "top_p": 1,
            "top_k": 1,
            "seed": 42,
            "max_tokens": 128,
            "min_tokens": 1,
        }
        self.sampling_params = SamplingParams(**gen_kwargs)

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def start(self):
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for worker in self.worker_threads:
            worker.join()

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            tik1 = time.time()

            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem
            ]

            tik2 = time.time()

            # vLLM API compatibility
            try:
                # vLLM 0.7+ API
                from vllm.inputs import TokensPrompt
                prompts = [TokensPrompt(prompt_token_ids=ids) for ids in input_ids_tensor]
                outputs = self.model.generate(
                    prompts=prompts, sampling_params=self.sampling_params
                )
            except (ImportError, TypeError):
                # vLLM 0.6.x API
                outputs = self.model.generate(
                    prompt_token_ids=input_ids_tensor,
                    sampling_params=self.sampling_params,
                )

            pred_output_tokens = []
            for output in outputs:
                pred_output_tokens.append(list(output.outputs[0].token_ids))
            tik3 = time.time()

            processed_output = self.data_object.postProcess(
                pred_output_tokens,
                query_id_list=query_ids,
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes()
                )
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens
                    )
                ]
                lg.QuerySamplesComplete(response)

            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")
                if tik1:
                    log.info(f"\tBatchMaker time: {tik2 - tik1}")
                    log.info(f"\tInference time: {tik3 - tik2}")
                    log.info(f"\tPostprocess time: {tok - tik3}")
                    log.info(f"\t==== Total time: {tok - tik1}")

    def load_model(self):
        from vllm import LLM

        log.info("Loading model on Ascend NPU...")
        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            enforce_eager=True,
            max_model_len=self.max_model_len,
        )
        log.info("Loaded model")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[: self.batch_size])
            query_samples = query_samples[self.batch_size :]
        log.info(f"IssueQuery done")

    def flush_queries(self):
        pass

    def __del__(self):
        pass


class SUTServer(SUT):
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        batch_size=None,
        total_sample_count=13368,
        dataset_path=None,
        workers=1,
        device_id=0,
        tensor_parallel_size=1,
        max_model_len=2048,
    ):
        super().__init__(
            model_path=model_path,
            dtype=dtype,
            batch_size=batch_size,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            workers=workers,
            device_id=device_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )

    def process_queries(self):
        """Server/SingleStream: process one query at a time using synchronous vLLM"""
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            try:
                input_ids = self.data_object.input_ids[qitem.index]

                # vLLM API compatibility
                try:
                    from vllm.inputs import TokensPrompt
                    prompt = TokensPrompt(prompt_token_ids=input_ids)
                    outputs = self.model.generate(
                        prompts=[prompt],
                        sampling_params=self.sampling_params,
                    )
                except (ImportError, TypeError):
                    outputs = self.model.generate(
                        prompt_token_ids=[input_ids],
                        sampling_params=self.sampling_params,
                    )

                output_tokens = list(outputs[0].outputs[0].token_ids)

                # Report first token
                first_token = np.array(output_tokens[:1], dtype=np.int32)
                response_data = array.array("B", first_token.tobytes())
                bi = response_data.buffer_info()
                response = [lg.QuerySampleResponse(qitem.id, bi[0], bi[1])]
                lg.FirstTokenComplete(response)

                # Report full output
                processed = self.data_object.postProcess(
                    [output_tokens], query_id_list=[qitem.index]
                )
                n_tokens = processed[0].shape[0]
                response_array = array.array("B", processed[0].tobytes())
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(qitem.id, bi[0], bi[1], n_tokens)
                ]
                lg.QuerySamplesComplete(response)

                with self.sample_counter_lock:
                    self.sample_counter += 1
                    if self.sample_counter % 10 == 0:
                        log.info(f"Samples run: {self.sample_counter}")

            except Exception as e:
                log.error(f"Error processing query: {e}", exc_info=True)
                response = [lg.QuerySampleResponse(qitem.id, 0, 0)]
                lg.QuerySamplesComplete(response)

    def issue_queries(self, query_samples):
        """Server scenario: one query at a time"""
        self.query_queue.put(query_samples[0])
