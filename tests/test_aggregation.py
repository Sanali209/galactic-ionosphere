import pytest
import asyncio
from src.core.database.orm import CollectionRecord, FieldPropInfo

# --- Test Models ---
class Product(CollectionRecord, table="products"):
    name = FieldPropInfo("name", default="", field_type=str)
    category = FieldPropInfo("category", default="generic", field_type=str)
    price = FieldPropInfo("price", default=0, field_type=int)

class Electronics(Product):
    voltage = FieldPropInfo("voltage", default=110, field_type=int)

@pytest.fixture
async def sample_data(db_teardown):
    p1 = Product(name="Apple", category="fruit", price=10)
    p2 = Product(name="Banana", category="fruit", price=5)
    e1 = Electronics(name="Radio", category="gadget", price=50, voltage=220)
    
    await p1.save()
    await p2.save()
    await e1.save()
    return [p1, p2, e1]

@pytest.mark.asyncio
async def test_aggregate_raw_group(sample_data):
    # Calculate average price
    pipeline = [
        {"$group": {"_id": None, "avg_price": {"$avg": "$price"}}}
    ]
    results = await Product.aggregate(pipeline)
    assert len(results) == 1
    # prices: 10, 5, 50 -> sum=65, avg=21.66
    assert 21 < results[0]['avg_price'] < 22

@pytest.mark.asyncio
async def test_aggregate_hydration(sample_data):
    # Find fruits and return as objects
    pipeline = [
        {"$match": {"category": "fruit"}}
    ]
    results = await Product.aggregate(pipeline, as_model=True)
    assert len(results) == 2
    assert isinstance(results[0], Product)
    assert results[0].category == "fruit"

@pytest.mark.asyncio
async def test_aggregate_polymorphism_auto_filter(sample_data):
    # Counting all raw docs in collection (should be 3)
    coll = Product.get_collection()
    count = await coll.count_documents({})
    assert count == 3
    
    # Aggregating via Electronics subclass should ONLY see the 1 Electronic item
    # Because aggregate() automatically injects match on _cls
    results = await Electronics.aggregate([], as_model=True)
    
    assert len(results) == 1
    assert isinstance(results[0], Electronics)
    assert results[0].name == "Radio"
    
    # Aggregating via Product (Base) should see all 3
    results_all = await Product.aggregate([], as_model=True)
    assert len(results_all) == 3
