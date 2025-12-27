import hashlib
from functools import wraps
import os
import diskcache as dc


class HashStore:
    @staticmethod
    def get_cache(base_directory, name):
        """
        Get or create a disk cache for the given hash table name.
        Each hash table gets its own subdirectory.
        """
        directory = os.path.join(base_directory, name)
        os.makedirs(directory, exist_ok=True)
        return dc.Index(directory)


class HashingService:
    # Class-level attributes for global configuration
    base_directory = "hash_store"
    custom_hash_functions = {}

    @staticmethod
    def initialize(base_directory="hash_store"):
        """Initialize the base directory for hash tables."""
        HashingService.base_directory = base_directory
        os.makedirs(base_directory, exist_ok=True)

    @staticmethod
    def register_custom_hash(table_name, hash_function):
        """Register a custom hashing function for a specific table."""
        HashingService.custom_hash_functions[table_name] = hash_function

    @staticmethod
    def _default_generate_key(*args, **kwargs):
        """Default method to generate a unique hash key based on positional and keyword arguments."""
        hash_object = hashlib.sha256()
        # Include positional arguments in the hash
        for arg in args:
            hash_object.update(str(arg).encode('utf-8'))
        # Include keyword arguments in the hash, sorted to ensure order consistency
        for key, value in sorted(kwargs.items()):
            hash_object.update(f"{key}={value}".encode('utf-8'))
        return hash_object.hexdigest()

    @staticmethod
    def _generate_key(table_name, *args, **kwargs):
        """Generate a unique key using the default or custom hashing function for the table."""
        if table_name in HashingService.custom_hash_functions:
            return HashingService.custom_hash_functions[table_name](*args, **kwargs)
        return HashingService._default_generate_key(*args, **kwargs)

    @staticmethod
    def hashable(hashtable="default", hash_self=True):
        """
        Decorator to add caching functionality with an option to include/exclude `self` in the hash.

        Parameters:
        - hashtable (str): The name of the hash table.
        - hash_self (bool): Whether to include the `self` object in the hash key.
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Handle `self` or `cls` exclusion based on `hash_self`
                filtered_args = args if hash_self else args[1:]
                # Get the cache for the specified hash table
                cache = HashStore.get_cache(HashingService.base_directory, hashtable)
                # Generate a unique cache key
                key = HashingService._generate_key(hashtable, *filtered_args, **kwargs)
                # Check if the result is already cached
                if key in cache:
                    return cache[key]
                # Compute the result and store it in the cache
                result = func(*args, **kwargs)
                cache[key] = result
                return result

            return wrapper

        return decorator

    @staticmethod
    def clear_cache(hashtable="default"):
        """Clear the cache for a specific hash table."""
        cache = HashStore.get_cache(HashingService.base_directory, hashtable)
        cache.clear()

    @staticmethod
    def get_cache_contents(hashtable="default"):
        """Get the contents of a specific hash table cache."""
        cache = HashStore.get_cache(HashingService.base_directory, hashtable)
        return dict(cache)
