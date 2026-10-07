import os
import re
import torch

from vllm import LLM
from vllm.sampling_params import SamplingParams
from typing import Optional, List, Dict, Any
from omegaconf import DictConfig
from PIL import Image

from src.guidance_models.base_guidance_model import BaseGuidanceModel
from src.guidance_models.common import get_messages, format_image, format_messages

import warnings
warnings.filterwarnings("once", module="vllm")

def _get_vllm_engine(
    model_id: str,
    dtype: torch.dtype,
    allowed_local_media_path: os.PathLike = "",
    hf_cache_dir: os.PathLike = os.environ.get("HF_HOME", None),
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.7,
    enforce_eager: bool = False, # Set to True to enforce eager mode for vLLM, avoids computing cuda graphs, reducing memory usage but potentially increasing latency (https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
    **kwargs
) -> LLM:
    """
    Helper function to create a VLLM LLM engine with model-specific parameters.
    Generation parameters adapted from https://docs.vllm.ai/en/latest/examples/offline_inference/vision_language.html
    
    Args:
        model_id (str): The name of the model to load from the Huggingface model hub.
        allowed_local_media_path (os.PathLike): Allowing API requests to read local images or videos from directories specified by the server file system.
        hf_cache_dir (os.PathLike): The directory to use for caching model files. Defaults to HF_HOME environment variable.
        dtype (torch.dtype): The data type to use for inference.
        tensor_parallel_size (int): The number of GPUs to use for tensor parallelism. Defaults to 1.
        gpu_memory_utilization (float): The fraction of GPU memory to use for the model (for each GPU). Defaults to 0.7.
        enforce_eager (bool): Whether to enforce eager mode for vLLM. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the LLM constructor.
    """

    model_id_lower = model_id.lower()
    common_kwargs = dict(
        model=model_id,
        allowed_local_media_path=allowed_local_media_path,
        dtype=dtype,
        download_dir=os.path.join(hf_cache_dir, "hub") if hf_cache_dir else None,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
        **kwargs
    )

    # Model‐specific overrides
    if "gemma" in model_id_lower:
        specific_kwargs = dict(
            max_model_len=8192,
            max_num_seqs=2,
            mm_processor_kwargs={"do_pan_and_scan": False},
        )
    elif "qwen" in model_id_lower and "2.5" in model_id_lower:
        specific_kwargs = dict(
            max_model_len=4096,
            max_num_seqs=5,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,
            },
        )
    elif "qwen" in model_id_lower and "3" in model_id_lower:
        specific_kwargs = dict(
            max_model_len=8192,
            max_num_seqs=2,
            mm_processor_kwargs={},
        )
    elif "llava" in model_id_lower:
        specific_kwargs = dict(max_model_len=4096)
    elif "mistral-small" in model_id_lower:
        specific_kwargs = dict(
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            max_model_len=65536,
            max_num_seqs=2,
            disable_mm_preprocessor_cache=True,
        )
    elif "aya-vision" in model_id_lower:
        specific_kwargs = dict(
            max_model_len=4096,
            max_num_seqs=2,
            mm_processor_kwargs={"crop_to_patches": True},
        )
    elif "molmo" in model_id_lower:
        specific_kwargs = dict(trust_remote_code=True)
    elif "internvl" in model_id_lower:
        specific_kwargs = dict(
            trust_remote_code=True,
            max_model_len=12288,
            #mm_processor_kwargs={"max_dynamic_patch": 4},
        )
    elif "glm-4.1v" in model_id_lower:
        specific_kwargs = dict(
            trust_remote_code=True,
            max_num_seqs=2,
            max_num_batched_tokens=8192)
    else:
        raise NotImplementedError(f"{model_id} not implemented")

    model = LLM(**common_kwargs, **specific_kwargs)
    return model


class VllmModel(BaseGuidanceModel):
    """
    Class for loading and using a VLLM Vision-Language Model (VLM) for image to text generation.
    """
    def __init__(self, 
                 model_id: str, 
                 guidance_model_config: DictConfig,
                 img_tag: str = "{image}",
                 hf_cache_dir: os.PathLike = os.environ.get("HF_HOME", None),
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None,
                 gpu_ids: Optional[List[int]] | int = None,
                 dtype: Optional[torch.dtype] = None,
                 gpu_memory_utilization: float = 0.7,
                 allowed_local_media_path: os.PathLike = "",
                 verbose: bool = False,
                 enforce_eager: bool = False,
                 **kwargs):
        """
        Initialize the Huggingface VLM model.
        If the model is not downloaded in the hf_cache_dir, it will be downloaded from the Huggingface model hub.
        In this case, ensure the HF_TOKEN environment variable is set to allow downloading gated models.
        
        Args:
            model_id (str): The name of the model to load (from the Huggingface model hub).
            guidance_model_config (DictConfig): The configuration object specific to the guidance model. Used for cache settings.
            img_tag (str): The tag to use in the prompt for the image. Defaults to "{image}".
            hf_cache_dir (os.PathLike): The directory to use for caching model files. Defaults to HF_HOME environment variable.
            seed (Optional[int]): The random seed to use for reproducibility. Defaults to None.
            device (Optional[torch.device]): The device to use for inference. If None, uses cuda if available, otherwise CPU.
            gpu_ids (Optional[List[int]] | int): List of GPU IDs to use. If None, uses all available GPUs. A single integer can also be provided to specify a single GPU. (Currently not implemented, vLLM uses all visible GPUs by default).
            dtype (Optional[torch.dtype]): The data type to use for inference. If None, uses bfloat16 if available, otherwise float16.
            gpu_memory_utilization (float): The fraction of GPU memory to use for the model (for each GPU). Defaults to 0.7.
            allowed_local_media_path (os.PathLike): Allowing API requests to read local images or videos from directories specified by the server file system. Defaults to an empty string, which means no local media paths are allowed.
            verbose (bool): Whether to print verbose output during model loading and inference.
            enforce_eager (bool): If True, uses eager mode for vLLM (if False, uses comipled mode). Defaults to False.
            **kwargs: Additional keyword arguments to pass to the model. Will be used to set sampling parameters like max_tokens, temperature, top_p, etc. Check vllm.sampling_params.SamplingParams for available parameters.
        """
        super().__init__(guidance_model_config=guidance_model_config, verbose=verbose)
        os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG' if verbose else 'ERROR'
        
        self.model_id = model_id
        self.img_tag = img_tag
        self.seed = seed
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # Set tensor parallel size based on the number of GPUs available
        if self.device.type == 'cuda':
            if gpu_ids is None:
                # Automatically use all available GPUs
                gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                print(f"No specific GPUs provided, vLLM will use all visible GPUs: {gpu_ids}")
            else: 
                if isinstance(gpu_ids, int):
                    gpu_ids = [gpu_ids]
                print(f"vLLM using specified GPUs: {gpu_ids}")
            tensor_parallel_size = len(gpu_ids) # Number of GPUs to use for tensor parallelism with vLLM
            self._vprint(f"Using tensor parallel size: {tensor_parallel_size} for vLLM on device {self.device}")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

        self._vprint(f"Huggingface cache dir: {hf_cache_dir}")
        print(f"Loading model {self.model_id}")
        
        self.model = _get_vllm_engine(
            model_id=self.model_id,
            allowed_local_media_path=allowed_local_media_path,
            gpu_memory_utilization=gpu_memory_utilization,
            hf_cache_dir=hf_cache_dir,
            dtype=self.dtype,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            disable_custom_all_reduce=True # otherwiswe deadlocks on multi-gpu
        )
        self.tokenizer = self.model.get_tokenizer()
        print(f"Model {self.model_id} loaded on device {self.device} with {self.dtype} precision.")
        
        self.enable_thinking = kwargs.pop("enable_thinking", None)  # None = use model default, True/False = override

        self.sampling_params = SamplingParams(
            seed=self.seed,
            max_tokens=kwargs.pop("max_tokens", 512),
            min_tokens=kwargs.pop("min_tokens", 1),
            n=kwargs.pop("num_return_sequences", 1),
            temperature=kwargs.pop("temperature", 1.0),
            top_p=kwargs.pop("top_p", 0.95),
            top_k=kwargs.pop("top_k", 64),
            presence_penalty=kwargs.pop("presence_penalty", 0.0),
            repetition_penalty=kwargs.pop("repetition_penalty", 1.0),
        )
        self._vprint(f"Sampling params: {self.sampling_params}")
        self._vprint(f"enable_thinking: {self.enable_thinking}")

    def __del__(self):
        """
        Destructor to clean up the model and free resources.
        """
        super().__del__()
        self._vprint(f"Cleaning up VllmModel on device.")
        
        if hasattr(self, 'model'):
            if hasattr(self.model, 'llm_engine') and hasattr(self.model.llm_engine, 'engine_core'):
                self.model.llm_engine.engine_core.shutdown()
        # Clean up distributed process group if initialized
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            import torch.distributed as dist
            dist.destroy_process_group()

    def _generate_no_cache(self, prompts: str | List[str], images: Optional[List[List[Image.Image]]] = None, **kwargs) -> List[str]:
        """
        Generate text from the given prompts and images.

        Parameters:
            prompts (str or list[str]): The prompts to generate text for. A list of prompts can be provided for batch generation.
            images (list[list[Image.Image]] or None): 2D list of shape (n_prompts, n_images_per_prompt) if any
            kwargs: Additional keyword arguments to pass to the model for inference (e.g. temperature, top_p, max_tokens).

        Returns:
            list[str]: The generated text for each prompt.
        """

        # Check for the correct number of images
        if images is not None:
            images_urls = []
            imgs_per_prompt = [prompt.count("{image}") for prompt in prompts]
            for i, prompt in enumerate(prompts):
                assert imgs_per_prompt[i] == len(images[i]), f"Prompt {i} expects {imgs_per_prompt[i]} images, but got {len(images[i])}"
                images_urls.append([format_image(img) for img in images[i]])
        else:
            images_urls = None

        if isinstance(prompts, str):
            prompts = [prompts]

        all_messages = [get_messages(prompt, img_tag=self.img_tag) for prompt in prompts]
        if images_urls is not None:
            all_messages = format_messages(all_messages, images_urls)
        stop_token_ids = [self.tokenizer.eos_token_id]
        
        # Sampling params are based on the defaults provided at initialization, but can be overridden by kwargs
        sampling_params = SamplingParams(
            seed=kwargs.get("seed", self.sampling_params.seed),
            max_tokens=kwargs.get("max_tokens", self.sampling_params.max_tokens),
            temperature=kwargs.get("temperature", self.sampling_params.temperature),
            top_p=kwargs.get("top_p", self.sampling_params.top_p),
            top_k=kwargs.get("top_k", self.sampling_params.top_k),
            n=kwargs.get("num_return_sequences", self.sampling_params.n),
            presence_penalty=kwargs.get("presence_penalty", self.sampling_params.presence_penalty),
            repetition_penalty=kwargs.get("repetition_penalty", self.sampling_params.repetition_penalty),
            stop_token_ids=stop_token_ids,
        )

        chat_kwargs = {}
        if self.enable_thinking is not None:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        outputs = self.model.chat(all_messages, sampling_params=sampling_params, use_tqdm=self.verbose, **chat_kwargs)


        batch_decoded_outputs = []
        for output in outputs:
            decoded_output = output.outputs[0].text
            batch_decoded_outputs.append(decoded_output)
        return batch_decoded_outputs
