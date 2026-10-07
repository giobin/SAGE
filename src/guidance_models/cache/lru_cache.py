import cachebox
from .base_cache import BaseCache

class LRUCache(BaseCache):
    def __init__(self, max_size: int = 128) -> None:
        """
        Initialize the LRU cache with a maximum size.
        
        Parameters:
            max_size (int): The maximum number of items the cache can hold.
        """
        super().__init__(max_size)
        self._cache = cachebox.LRUCache(maxsize=max_size)

    def get(self, key):
        """
        Retrieve an item from the cache by its key.
        
        Parameters:
            key (Any): The key to look up in the cache.
        
        Returns:
            Any: The cached item or None if not found.
        """
        value = self._cache.get(key, None)
        if value is not None:
            self._cache_stats["hits"] += 1
        else:
            self._cache_stats["misses"] += 1
        return value

    def set(self, key, value):
        """
        Store an item in the cache with a specified key.
        
        Parameters:
            key (Any): The key under which to store the item.
            value (Any): The item to store in the cache.
        """
        self._cache[key] = value