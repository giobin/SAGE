import os
import torch
from typing import Optional, List
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig

from src.guidance_models.base_guidance_model import BaseGuidanceModel
from src.guidance_models.common import get_messages

class TransformersModel(BaseGuidanceModel):
    def __init__(
        self,
        model_id: str,
        guidance_model_config: DictConfig,
        img_tag: str = "{image}",
        hf_cache_dir: Optional[str] = os.environ.get("HF_HOME"),
        seed: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Pure Transformers implementation for text-image to text generation.
        Compatible with TRL and PEFT for fine-tuning.
        """
        super().__init__(guidance_model_config=guidance_model_config, verbose=False)
        # Drop unsupported kwargs used for vLLM
        kwargs.pop("gpu_ids", None)
        kwargs.pop("gpu_memory_utilization", None)
        kwargs.pop("env_id", None) # used in oracle
        kwargs.pop("enforce_eager", None)
        kwargs.pop("json_output", None)
        
        self.model_id = model_id
        self.img_tag = img_tag
        self.seed = seed
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype or kwargs.pop("torch_dtype", None) or (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code, cache_dir=hf_cache_dir, use_fast=kwargs.pop("use_fast_tokenizer", True))
        
        self.generation_config = GenerationConfig(
            max_new_tokens=kwargs.pop("max_tokens", 512),
            temperature=kwargs.pop("temperature", 0.8),
            top_p=kwargs.pop("top_p", 0.95),
            top_k=kwargs.pop("top_k", 50),
            do_sample=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
        )
        self.generation_config.update(**kwargs)
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map=kwargs.pop("device_map", "auto" if torch.cuda.is_available() else None),
            trust_remote_code=trust_remote_code,
            cache_dir=hf_cache_dir,
            # attn_implementation=kwargs.pop("attn_implementation", "flash_attention_2" if torch.cuda.is_available() else None),
            **kwargs
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)

        torch.set_float32_matmul_precision('high')
        
    def __del__(self):
        import gc
        del self.processor
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def _generate_no_cache(self, prompts: str | List[str], images: Optional[List] = None, **kwargs) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        # Check correct number of images
        if images is not None:
            imgs_per_prompt = [prompt.count("{image}") for prompt in prompts]
            for i, prompt in enumerate(prompts):
                assert imgs_per_prompt[i] == len(images[i]), f"Prompt {i} expects {imgs_per_prompt[i]} images, but got {len(images[i])}"

        all_messages = [get_messages(p, img_tag=self.img_tag) for p in prompts]
        texts = [self.processor.apply_chat_template(m, add_generation_prompt=True) for m in all_messages]
        model_inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.model.device)
        gen_args = self.generation_config
        gen_args.update(**kwargs)
        out_ids = self.model.generate(**model_inputs, generation_config=gen_args)
        out_ids = out_ids[:, model_inputs["input_ids"].shape[1]:] # remove the prompt ids
        tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.tokenizer
        decoded = tok.batch_decode(out_ids, skip_special_tokens=True)
        return decoded
