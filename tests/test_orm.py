import pytest
import asyncio
from src.core.database.orm import CollectionRecord, FieldPropInfo

# --- Test Models ---
class UserRecordForTest(CollectionRecord, table="test_users"):
    username = FieldPropInfo("username", default="anon", field_type=str)
    score = FieldPropInfo("score", default=0, field_type=int)

class AdminRecordForTest(UserRecordForTest):
    permissions = FieldPropInfo("permissions", default=[], field_type=list)

@pytest.mark.asyncio
async def test_orm_create_save_load(db_teardown):
    user = UserRecordForTest(username="tester", score=100)
    await user.save()
    
    assert user.id is not None
    
    # Load back
    loaded = await UserRecordForTest.get(user.id)
    assert loaded is not None
    assert loaded.username == "tester"
    assert loaded.score == 100

@pytest.mark.asyncio
async def test_orm_reactivity():
    user = UserRecordForTest()
    events = []
    
    def on_change(obj, field, val):
        events.append((field, val))
        
    user.on_change.connect(on_change)
    user.score = 50
    
    assert len(events) == 1
    assert events[0] == ("score", 50)

@pytest.mark.asyncio
async def test_orm_polymorphism(db_teardown):
    admin = AdminRecordForTest(username="admin", permissions=["root"])
    await admin.save()
    
    # Load via base class
    loaded = await UserRecordForTest.get(admin.id)
    assert isinstance(loaded, AdminRecordForTest)
    assert loaded.permissions == ["root"]

@pytest.mark.asyncio
async def test_field_validator(caplog):
    from loguru import logger
    
    # Propagate loguru to standard logging for caplog
    logger.add(lambda msg: pd_log_sink(msg, caplog), format="{message}")
    
    # Or simpler: verify loguru creates output differently or use caplog handler
    # Standard python logging capture works if we add a sink that writes to a handler
    # But for simplicity, let's just use a list sink
    
    log_messages = []
    handler_id = logger.add(lambda msg: log_messages.append(msg))

    def positive_only(val):
        return val >= 0
        
    class ValidatedRecord(CollectionRecord, table="test_valid"):
        count = FieldPropInfo("count", default=0, validator=positive_only)
        
    rec = ValidatedRecord()
    rec.count = -5
    
    logger.remove(handler_id)
    
    assert any("Validation failed" in str(m) for m in log_messages)

def pd_log_sink(msg, caplog):
    pass
