import os
import argparse

os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="/home/ma-user/pretrainmodel/Meta-Llama-3.1-8B-Instruct",
)
args = parser.parse_args()
MODEL = args.model

from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    tensor_parallel_size=1,
    max_model_len=2048,
    enforce_eager=True,
)

params = SamplingParams(temperature=0.0, max_tokens=32)
out = llm.generate(["Say hello in one short sentence."], params)
print(out[0].outputs[0].text)