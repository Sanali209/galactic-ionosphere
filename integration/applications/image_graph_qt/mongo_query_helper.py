import re
import loguru
from SLM.mongoext.mongoQueryTextParser2 import parse_mongo_query_with_normalization

class MongoQueryHelper:
    """Helper class for parsing and processing MongoDB queries."""

    def parse_and_process(self, query_text: str) -> dict | None:
        """Parses the query text and applies post-processing."""
        if not query_text:
            loguru.logger.info("Query text is empty.")
            return None

        mongo_filter = None
        try:
            # Parsing the query text into a MongoDB filter structure
            parsed_filter = parse_mongo_query_with_normalization(query_text)
            # Post-processing (e.g., anchoring regex for paths)
            mongo_filter = self._postprocess_mongo_filter(parsed_filter)
            loguru.logger.debug(f"Parsed and processed MongoDB filter: {mongo_filter}")
            return mongo_filter
        except Exception as parse_error:
            loguru.logger.warning(f"Could not parse as structured query ('{query_text}'): {parse_error}. Falling back to text search.")
            # Fallback to simple text search if parsing fails
            return {"$text": {"$search": query_text}}

    def _postprocess_mongo_filter(self, filter_obj, parent_key=None):
        """Recursively processes the filter, e.g., escaping paths for regex."""
        if isinstance(filter_obj, dict):
            new_filter = {}
            for key, value in filter_obj.items():
                # Example: Ensure local_path regex starts from the beginning
                if key == "$regex" and parent_key == "local_path":
                    str_value = str(value) if value is not None else ""
                    # Anchor regex to the start and escape
                    new_filter[key] = "^" + re.escape(str_value)
                # Recursive calls for nested structures
                elif key in ["$in", "$or", "$and"]:
                     if isinstance(value, list):
                          new_filter[key] = [self._postprocess_mongo_filter(v, key) for v in value]
                     else:
                          # Keep non-list values as is (might indicate an error in query)
                          loguru.logger.warning(f"Expected list for key '{key}', got {type(value)}. Keeping original value.")
                          new_filter[key] = value
                elif key == "$not":
                    new_filter[key] = self._postprocess_mongo_filter(value, key)
                elif key == "$elemMatch":
                    new_filter[key] = self._postprocess_mongo_filter(value, key)
                # Process other keys recursively
                else:
                    new_filter[key] = self._postprocess_mongo_filter(value, key)
            return new_filter
        elif isinstance(filter_obj, list):
            # Process lists recursively (e.g., within $or, $and)
            return [self._postprocess_mongo_filter(item, parent_key) for item in filter_obj]
        else:
            # Return value as is if not dict or list
            return filter_obj
