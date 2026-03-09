"""
Ascend NPU MLPerf Inference Main Entry Point
Based on main.py, adapted for Ascend NPU with vLLM
Supports Offline, Server, SingleStream scenarios
"""

import subprocess
import mlperf_loadgen as lg
import argparse
import os
import logging
import sys

sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-MAIN-Ascend")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["Offline", "Server", "SingleStream"],
        default="Offline",
        help="Scenario",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="")
    parser.add_argument(
        "--accuracy", action="store_true", help="Run accuracy mode"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type of the model",
    )
    parser.add_argument(
        "--audit-conf",
        type=str,
        default="audit.conf",
        help="Audit config for LoadGen settings during compliance runs",
    )
    parser.add_argument(
        "--user-conf",
        type=str,
        default="user.conf",
        help="User config for user LoadGen settings such as target QPS",
    )
    parser.add_argument(
        "--total-sample-count",
        type=int,
        default=13368,
        help="Number of samples to use in benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Model batch-size to use in benchmark.",
    )
    parser.add_argument(
        "--output-log-dir",
        type=str,
        default="output-logs",
        help="Where logs are saved",
    )
    parser.add_argument(
        "--enable-log-trace",
        action="store_true",
        help="Enable log tracing. This file can become quite large",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers to process queries",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Ascend NPU device ID",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for model sharding",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model sequence length (controls NPU memory usage)",
    )
    parser.add_argument(
        "--lg-model-name",
        type=str,
        default="llama3_1-8b",
        choices=["llama3_1-8b", "llama3_1-8b-edge"],
        help="Model name for LoadGen configuration",
    )

    args = parser.parse_args()
    return args


scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
    "singlestream": lg.TestScenario.SingleStream,
}


def main():
    args = get_args()

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario.lower()]
    settings.FromConfig(args.user_conf, args.lg_model_name, args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    os.makedirs(args.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    from SUT_Ascend import SUT, SUTServer

    sut_map = {"offline": SUT, "server": SUTServer, "singlestream": SUTServer}
    sut_cls = sut_map[args.scenario.lower()]

    sut = sut_cls(
        model_path=args.model_path,
        dtype=args.dtype,
        batch_size=args.batch_size,
        dataset_path=args.dataset_path,
        total_sample_count=args.total_sample_count,
        workers=args.num_workers,
        device_id=args.device_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    # Start sut before loadgen starts
    sut.start()
    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    log.info("Starting Benchmark run")
    lg.StartTestWithLogSettings(
        lgSUT, sut.qsl, settings, log_settings, args.audit_conf
    )

    # Stop sut after completion
    sut.stop()

    log.info("Run Completed!")

    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)

    log.info("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()
