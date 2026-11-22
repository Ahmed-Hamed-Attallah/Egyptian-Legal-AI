"""
Hardware-aware ML model configuration.
Handles GPU detection, device resolution, and memory-safe model parameters.
"""
import logging
from typing import Literal

import torch

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Get Hardware Info

def get_gpu_info() -> dict:
    """
    Detect GPU availability and VRAM.
    
    Returns:
        dict with keys: 'available' (bool), 'name' (str), 'vram_gb' (float)
    """
    if not torch.cuda.is_available():
        return {"available":False, "vram_gb":0, "name":None}
    try:
        gpu_name = torch.cuda.get_device_name(0)
        # FIX: Added .cuda before get_device_properties
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        return {"available": True, "name": gpu_name, "vram_gb": round(vram_gb, 2)}
    except Exception as e:
        logger.warning(f"Failed to detect GPU info: {e}")
        return {"available": False, "name": None, "vram_gb": 0.0}



def resolve_device(requested_device: str = 'auto', model_type:str = "llm") -> str:
    """
    Resolve the optimal device for a model based on hardware constraints.
    
    Args:
        requested_device: User preference from settings ('auto', 'cpu', 'cuda')
        model_type: Type of model ('embedding', 'reranker', 'llm')
    
    Returns:
        Resolved device string ('cpu' or 'cuda')
    """

    gpu_info = get_gpu_info()

    # If user explicitly requested a device, respect it (with warning)
    if requested_device != "auto":
        if requested_device == "cuda" and not gpu_info['available']:
            logger.warning(
                f"GPU not available but {model_type} requested 'cuda'. Falling back to CPU."
            )

            return "cpu"
        return requested_device

    # Auto resolution based on vram 
    if not gpu_info["available"]:
        logger.info(f"GPU not available. {model_type} will run on CPU.")
        return "cpu"
    
    vram_gb = gpu_info["vram_gb"]
    
    # Memory budgets per model type (conservative for 4GB VRAM)
    memory_budgets = {
        "embedding": 1.0,   # ~500MB model + inference buffer
        "reranker": 1.5,    # ~1GB model + inference buffer
        "llm": 2.5,         # ~1.5GB quantized model + KV cache
    }
    
    required_vram = memory_budgets.get(model_type, 1.0)
    
    if vram_gb >= required_vram:
        logger.info(
            f"GPU detected: {gpu_info['name']} ({vram_gb}GB VRAM). "
            f"{model_type} will use CUDA."
        )
        return "cuda"
    else:
        logger.warning(
            f"Insufficient VRAM ({vram_gb}GB) for {model_type} "
            f"(requires ~{required_vram}GB). Falling back to CPU."
        )
        return "cpu"


# Model-Specific Configurations


def get_embedding_config() -> dict:
    """
    Get configuration for the embedding model.
    
    Returns:
        dict with model parameters optimized for hardware
    """
    device = resolve_device(settings.EMBEDDING_DEVICE, "embedding")
    
    # Adjust batch size based on device
    batch_size = settings.EMBEDDING_BATCH_SIZE
    if device == "cpu":
        # CPU can handle larger batches (no VRAM limit, but slower)
        batch_size = min(batch_size * 2, 64)
    
    return {
        "model_name": settings.EMBEDDING_MODEL_NAME,
        "device": device,
        "batch_size": batch_size,
        "normalize_embeddings": settings.EMBEDDING_NORMALIZE,
        "dimension": settings.EMBEDDING_DIMENSION,
    }


def get_reranker_config() -> dict:
    """
    Get configuration for the reranker (cross-encoder) model.
    
    Returns:
        dict with model parameters optimized for hardware
    """
    device = resolve_device(settings.RERANKER_DEVICE, "reranker")
    
    # Reranker processes pairs, so batch size should be smaller
    batch_size = 8 if device == "cuda" else 16
    
    return {
        "model_name": settings.RERANKER_MODEL_NAME,
        "device": device,
        "batch_size": batch_size,
        "max_length": 512,  # Typical max length for cross-encoders
    }


def get_local_llm_config() -> dict:
    """
    Get configuration for the local LLM fallback model.
    
    Returns:
        dict with model parameters optimized for hardware (4GB VRAM)
    """
    device = resolve_device(settings.LOCAL_MODEL_DEVICE, "llm")
    
    # Determine dtype and quantization based on device
    if device == "cuda":
        # Force 4-bit quantization for 4GB VRAM
        dtype = torch.float16
        load_in_4bit = True
        logger.info("Local LLM configured with 4-bit quantization for GPU.")
    else:
        # CPU can use float32 (slower but no VRAM limit)
        dtype = torch.float32
        load_in_4bit = False
        logger.info("Local LLM configured for CPU inference.")
    
    return {
        "model_name": settings.LOCAL_MODEL_NAME,
        "device": device,
        "dtype": dtype,
        "load_in_4bit": load_in_4bit,
        "max_new_tokens": settings.LOCAL_MODEL_MAX_NEW_TOKENS,
        "temperature": settings.LLM_TEMPERATURE,
    }



# Utility Functions


def log_hardware_summary() -> None:
    """
    Log a summary of hardware configuration at startup.
    Useful for debugging and monitoring.
    """
    gpu_info = get_gpu_info()
    
    logger.info("=" * 60)
    logger.info("Hardware Configuration Summary")
    logger.info("=" * 60)
    
    if gpu_info["available"]:
        logger.info(f"GPU: {gpu_info['name']}")
        logger.info(f"VRAM: {gpu_info['vram_gb']} GB")
    else:
        logger.info("GPU: Not available (CPU-only mode)")
    
    logger.info(f"Embedding Device: {get_embedding_config()['device']}")
    logger.info(f"Reranker Device: {get_reranker_config()['device']}")
    logger.info(f"Local LLM Device: {get_local_llm_config()['device']}")
    logger.info("=" * 60)