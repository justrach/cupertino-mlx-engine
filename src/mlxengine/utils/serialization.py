import collections.abc
from typing import Any
# Import the base Model class from satya to check instance types
# Assuming satya is importable here, adjust if needed
try:
    from satya import Model
except ImportError:
    # Define a placeholder if satya is not available in this context
    # This allows the function structure to be defined, but it won't
    # correctly identify satya models without the actual import.
    class Model: pass

def recursive_to_dict(item: Any) -> Any:
    """Recursively converts satya.Model instances to dictionaries."""
    if isinstance(item, Model):
        # Call .dict() on the model instance
        try:
            # Use .dict() which is standard for Pydantic/BaseModel
            # Adjust if satya uses a different method (e.g., .model_dump())
            d = item.dict()
        except AttributeError:
             # Fallback if .dict() doesn't exist
             # This example assumes fields are attributes or stored in __fields__
             try:
                 # Attempt common fallback for models storing fields in __fields__
                 d = {f: getattr(item, f) for f in item.__fields__}
             except AttributeError:
                 # Last resort: try vars(), might include internals
                 try:
                     d = vars(item)
                 except TypeError: # vars() doesn't work on all objects
                    # Final fallback: return as is, hoping it's serializable or error later
                    return item

        # Recursively process the dictionary values
        return recursive_to_dict(d)
    elif isinstance(item, collections.abc.Mapping):
        # If it's a dictionary-like object, process its values
        return {k: recursive_to_dict(v) for k, v in item.items()}
    elif isinstance(item, collections.abc.Sequence) and not isinstance(item, (str, bytes)):
        # If it's a list/tuple-like object (but not string/bytes), process its elements
        return [recursive_to_dict(elem) for elem in item]
    else:
        # Assume it's a primitive type (int, str, float, bool, None) or already processed
        return item 