import random
import copy

from collections import deque
from .base_cache import BaseCache
from typing import Any, Dict, List, Optional, Tuple

class VectorCache(BaseCache):
    def __init__(self, max_size: int = 128, vector_size: int = 10) -> None:
        """
        Initialize the VectorCache with a maximum size and vector size per key.
        Args:
            max_size (int): The maximum number of keys the cache can hold.
            vector_size (int): The number of values to store per key.
        """
        super().__init__(max_size)
        self.vector_size = vector_size
        self._cache = {}  # key -> deque
        self._cache_stats = {"hits": 0, "misses": 0}

    def get(self, key: Any) -> Optional[Any]:
        """
        Sample a value from the vector for the given key.
        Returns None if key not found or vector is empty.
        """
        entry = self._cache.get(key)
        if entry is not None and len(entry) > 0:
            self._cache_stats["hits"] += 1
            return random.choice(list(entry))
        else:
            self._cache_stats["misses"] += 1
            return None

    def is_full(self, key: Any) -> bool:
        """
        Check if the vector for the given key is full (i.e., has reached vector_size).
        Returns False if the key does not exist.
        """
        entry = self._cache.get(key)
        return entry is not None and len(entry) == self.vector_size

    def set(self, key: Any, value: Any, only_if_not_full: bool = False) -> None:
        """
        Insert value into the vector for the given key, using collections.deque as a circular buffer.
        If only_if_not_full is True, only add the value if the vector is not full.
        """
        if key not in self._cache:
            if len(self._cache) >= self.max_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = deque(maxlen=self.vector_size)
        if only_if_not_full and len(self._cache[key]) == self.vector_size:
            return
        self._cache[key].append(value)

    # Override the get_responses method to handle vector cache logic (only add to responses if the vector is full)
    def get_responses(self, prompts: List[str], images: Optional[List[Any]] = None) -> Tuple[List[Optional[Any]], Dict[int, str], Dict[int, Any]]:
        """
        Retrieve responses for a list of prompts from the cache.
        If a prompt is not found in the cache, it will be added to the prompts_for_model dictionary (with its index) for later processing.
        If images are provided, they will be included in the cache key and an images_for_model dictionary will be returned with the corresponding images for each prompt.
        
        Args:
            prompts (List[str]): A list of prompts to retrieve from the cache.
            images (Optional[List[Any]]): A list of images corresponding to the prompts. Defaults to None.
            
        Returns:
            Tuple[List[Optional[Any]], Dict[int, str], Dict[int, Any]]:
                - A list of responses for each prompt, where None indicates a cache miss.
                - A dictionary mapping the index of each prompt to the prompt itself for later processing.
                - A dictionary mapping the index of each prompt to the corresponding image (if provided).
        """
        
        responses = [ None ] * len(prompts)
        prompts_for_model = {}
        images_for_model = {}
        
        for i, prompt in enumerate(prompts):
            if images is not None:
                img_key = tuple(copy.copy(img).tobytes() for img in images[i])
                key = (prompt, img_key)
            else:
                key = (prompt, None)
            
            value = self.get(key)
            # If the vector is not full, we still want to ask the model to generate a response even if we have one in the cache (so we can populate the vector)
            if value is not None and self.is_full(key):
                responses[i] = value
            else:
                if value is not None and not self.is_full(key):
                    # We already had something in cache, but it was not full, so we ask the model to generate a new response
                    # This should count as a "cache miss" since we are not returning the cached value, even if the key exists
                    self._cache_stats["hits"] -= 1 # Remove the hit that was counted in the get method
                    self._cache_stats["misses"] += 1
                
                # Store the prompt and image for the model to process later all together in a batch
                prompts_for_model[i] = prompt
                if images is not None:
                    images_for_model[i] = images[i]
            
        return responses, prompts_for_model, images_for_model

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """
        Returns cache statistics including hits, misses, hit ratio, cache size,
        average vector length, and average diversity (unique values per key).
        """
        hit_ratio = self._cache_stats["hits"] / (self._cache_stats["hits"] + self._cache_stats["misses"]) if (self._cache_stats["hits"] + self._cache_stats["misses"]) > 0 else 0
        lengths = [len(entry) for entry in self._cache.values()]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        diversities = [len(set(entry)) for entry in self._cache.values()]
        avg_div = sum(diversities) / len(diversities) if diversities else 0
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_ratio": hit_ratio,
            "size": len(self._cache),
            "avg_vector_length": avg_len,
            "avg_diversity": avg_div
        }
