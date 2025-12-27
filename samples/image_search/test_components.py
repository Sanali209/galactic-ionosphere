"""
Test script for Image Search components
"""
import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

async def test_search_service():
    """Test SearchService directly"""
    print("\n" + "="*60)
    print("TEST 1: SearchService")
    print("="*60)
    
    from src.core.locator import ServiceLocator
    from src.core.search_service import SearchService
    
    sl = ServiceLocator()
    service = SearchService(sl, None)
    await service.initialize()
    
    results = await service.search_images("test", 5)
    print(f"✅ SearchService returned {len(results)} results")
    
    if results:
        print(f"   Sample result: {results[0].title}")
    
    return len(results) > 0

async def test_database():
    """Test database connection and ORM"""
    print("\n" + "="*60)
    print("TEST 2: Database & ORM")
    print("="*60)
    
    from src.core.database.manager import DatabaseManager
    from src.models.search_history import SearchHistory
    from datetime import datetime
    
    db = DatabaseManager()
    await db.connect("localhost", 27017, "image_search_test_db")
    print("✅ Database connected")
    
    # Create test record
    record = SearchHistory()
    record.query = "test_query"
    record.timestamp = datetime.now()
    record.result_count = 5
    await record.save()
    print(f"✅ Test record created: {record._id}")
    
    # Find records
    records = await SearchHistory.find({"query": "test_query"})
    print(f"✅ Found {len(records)} test records")
    
    return True

async def test_task_system():
    """Test TaskSystem registration and execution"""
    print("\n" + "="*60)
    print("TEST 3: TaskSystem")
    print("="*60)
    
    from src.core.locator import ServiceLocator
    from src.core.tasks.system import TaskSystem
    
    sl = ServiceLocator()
    sl.init("config.json")
    
    task_system = sl.get_system(TaskSystem)
    
    # Register test handler
    async def test_handler(arg1: str):
        print(f"  Handler executed with: {arg1}")
        return f"Success: {arg1}"
    
    task_system.register_handler("test", test_handler)
    print("✅ Handler registered")
    
    # Submit task
    await task_system.submit("test", "Test Task", "test_arg")
    print("✅ Task submitted")
    
    await asyncio.sleep(2)  # Wait for execution
    
    return True

async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("IMAGE SEARCH - COMPONENT TESTS")
    print("="*60)
    
    results = {}
    
    try:
        results['search'] = await test_search_service()
    except Exception as e:
        print(f"❌ SearchService test failed: {e}")
        results['search'] = False
    
    try:
        results['database'] = await test_database()
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        results['database'] = False
    
    try:
        results['tasks'] = await test_task_system()
    except Exception as e:
        print(f"❌ TaskSystem test failed: {e}")
        results['tasks'] = False
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")
    
    all_passed = all(results.values())
    print("\n" + ("="*60))
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
