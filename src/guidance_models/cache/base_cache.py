import copy

from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

class BaseCache(ABC):
    def __init__(self, max_size: int = 128) -> None:
        """
        Initialize the base cache with a maximum size.
        
        Args:
            max_size (int): The maximum number of items the cache can hold.
        """
        self.max_size = max_size
        self._cache = None
        self._cache_stats = {"hits": 0, "misses": 0}

    def get(self, key: Any) -> Any:
        """
        Retrieve an item from the cache by its key.
        
        Parameters:
            key (Any): The key to look up in the cache.
        
        Returns:
            Any: The cached item or None if not found.
        """
        pass

    @abstractmethod
    def set(self, key: Any, value: Any) -> None:
        """
        Store an item in the cache with a specified key.
        
        Parameters:
            key (Any): The key under which to store the item.
            value (Any): The item to store in the cache.
        """
        pass
    
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
            if value is not None:
                responses[i] = value
            else:
                # Store the prompt and image for the model to process later all together in a batch
                prompts_for_model[i] = prompt
                if images is not None:
                    images_for_model[i] = images[i]
            
        return responses, prompts_for_model, images_for_model

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """
        Returns the cache statistics including hits, misses, hit ratio, and cache size.
        The hit ratio is calculated as the number of hits divided by the total number of requests (hits + misses).
        
        Returns
            dict: A dictionary containing the cache statistics:
                - "hits": Number of cache hits
                - "misses": Number of cache misses
                - "hit_ratio": The ratio of hits to total requests (hits + misses)
                - "cache_size": Current size of the cache
        """
        hit_ratio = self._cache_stats["hits"] / (self._cache_stats["hits"] + self._cache_stats["misses"]) if (self._cache_stats["hits"] + self._cache_stats["misses"]) > 0 else 0
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_ratio": hit_ratio,
            "size": len(self._cache)
        }