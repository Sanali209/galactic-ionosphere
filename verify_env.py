import sys
print(f"Python: {sys.version}")

try:
    import pyexiv2
    print("SUCCESS: pyexiv2 imported")
except ImportError as e:
    print(f"FAILURE: pyexiv2 not found: {e}")

try:
    import qdrant_client
    print("SUCCESS: qdrant_client imported")
except ImportError as e:
    print(f"FAILURE: qdrant_client not found: {e}")

try:
    import motor.motor_asyncio
    print("SUCCESS: motor imported")
except ImportError as e:
    print(f"FAILURE: motor not found: {e}")
    
try:
    from pymongo import AsyncMongoClient
    print("SUCCESS: pymongo.AsyncMongoClient imported")
except ImportError:
    print("INFO: pymongo.AsyncMongoClient NOT found (Expected if using motor)")
except Exception as e:
    print(f"FAILURE: checking pymongo: {e}")
