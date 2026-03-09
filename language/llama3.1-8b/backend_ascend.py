"""
Ascend NPU Inference Backend Implementation
Supports two modes:
1. vLLM-Ascend (recommended)
2. torch_npu native implementation
"""

import torch
import logging
from typing import List, Optional, Dict, Any

log = logging.getLogger("Ascend-Backend")


class AscendBackend:
    """Ascend NPU inference backend"""
    
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device_id: int = 0,
        backend_type: str = "auto",  # "auto", "vllm", "torch_npu"
        tensor_parallel_size: int = 1,
    ):
        """
        Initialize Ascend NPU backend
        
        Args:
            model_path: Model path or HuggingFace model ID
            dtype: Data type ("float16", "bfloat16", "float32")
            device_id: NPU device ID
            backend_type: Backend type, auto will try vllm first
            tensor_parallel_size: Tensor parallel size
        """
        self.model_path = model_path
        self.dtype = dtype
        self.device_id = device_id
        self.device = f"npu:{device_id}"
        self.backend_type = backend_type
        self.tensor_parallel_size = tensor_parallel_size
        self.model = None
        self.tokenizer = None
        self.actual_backend = None
        
    def load_model(self):
        """Load model to Ascend NPU"""
        log.info(f"Loading model on NPU device {self.device_id}...")
        
        if self.backend_type in ["auto", "vllm"]:
            if self._try_load_vllm():
                return
                
        if self.backend_type in ["auto", "torch_npu"]:
            if self._try_load_torch_npu():
                return
                
        raise RuntimeError("Failed to load model with any available backend")
        
    def _try_load_vllm(self) -> bool:
        """Try to load model with vLLM-Ascend"""
        try:
            log.info("Attempting to load with vLLM-Ascend...")
            
            # Import vllm (requires vllm-ascend version)
            from vllm import LLM, SamplingParams
            
            # Initialize vLLM engine
            self.model = LLM(
                model=self.model_path,
                dtype=self.dtype,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
                # Ascend-specific configuration
                device="npu",
                enforce_eager=True,  # May be needed in some cases
            )
            
            self.actual_backend = "vllm"
            log.info("✓ Successfully loaded model with vLLM-Ascend")
            return True
            
        except ImportError as e:
            log.warning(f"vLLM not available: {e}")
            return False
        except Exception as e:
            log.error(f"Failed to load with vLLM: {e}")
            return False
            
    def _try_load_torch_npu(self) -> bool:
        """Try to load model with torch_npu native"""
        try:
            log.info("Attempting to load with torch_npu...")
            
            # Import torch_npu
            import torch_npu
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model
            torch_dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype_map.get(self.dtype, torch.bfloat16),
                device_map=self.device,
                trust_remote_code=True,
            )
            
            self.model.eval()
            self.actual_backend = "torch_npu"
            log.info("✓ Successfully loaded model with torch_npu")
            return True
            
        except ImportError as e:
            log.warning(f"torch_npu not available: {e}")
            return False
        except Exception as e:
            log.error(f"Failed to load with torch_npu: {e}")
            return False
            
    def generate(
        self,
        input_ids: List[List[int]],
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        **kwargs
    ) -> List[List[int]]:
        """
        Execute inference generation
        
        Args:
            input_ids: List of input token IDs
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            List of generated token IDs
        """
        if self.actual_backend == "vllm":
            return self._generate_vllm(
                input_ids, max_tokens, temperature, top_p, top_k, **kwargs
            )
        elif self.actual_backend == "torch_npu":
            return self._generate_torch_npu(
                input_ids, max_tokens, temperature, top_p, top_k, **kwargs
            )
        else:
            raise RuntimeError("Model not loaded")
            
    def _generate_vllm(
        self,
        input_ids: List[List[int]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        **kwargs
    ) -> List[List[int]]:
        """Generate using vLLM"""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            min_tokens=1,
            seed=42,
            **kwargs
        )
        
        # vLLM generation
        outputs = self.model.generate(
            prompt_token_ids=input_ids,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        
        # Extract generated tokens
        result = []
        for output in outputs:
            generated_tokens = list(output.outputs[0].token_ids)
            result.append(generated_tokens)
            
        return result
        
    def _generate_torch_npu(
        self,
        input_ids: List[List[int]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        **kwargs
    ) -> List[List[int]]:
        """Generate using torch_npu native"""
        import torch_npu
        
        # Convert to tensor and move to NPU
        input_tensors = torch.tensor(input_ids).to(self.device)
        
        # Generation configuration
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "min_new_tokens": 1,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if temperature > 0:
            gen_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })
            
        gen_kwargs.update(kwargs)
        
        # Execute generation
        with torch.no_grad():
            outputs = self.model.generate(
                input_tensors,
                **gen_kwargs
            )
            
        # Extract newly generated tokens (remove input portion)
        result = []
        for i, output in enumerate(outputs):
            input_len = len(input_ids[i])
            generated_tokens = output[input_len:].cpu().tolist()
            result.append(generated_tokens)
            
        return result
        
    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            "backend": self.actual_backend,
            "device": self.device,
            "dtype": self.dtype,
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        
    def __del__(self):
        """Cleanup resources"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        # Clear NPU cache
        try:
            import torch_npu
            torch_npu.npu.empty_cache()
        except:
            pass
