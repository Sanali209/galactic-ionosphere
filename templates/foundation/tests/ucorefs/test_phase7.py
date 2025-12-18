"""
UCoreFS Phase 7 Tests - Rules Engine

Tests for:
- Rule model
- Conditions (extensible)
- Actions (extensible)
- RulesEngine
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId


class TestRuleModel:
    """Tests for Rule model."""
    
    def test_rule_creation(self):
        """Test creating a rule."""
        from src.ucorefs.rules.models import Rule
        
        rule = Rule(
            name="Auto-tag vacation photos",
            trigger="on_import",
            conditions=[
                {"type": "path_contains", "params": {"substring": "vacation"}}
            ],
            actions=[
                {"type": "add_tag", "params": {"tag_id": str(ObjectId())}}
            ]
        )
        
        assert rule.name == "Auto-tag vacation photos"
        assert rule.trigger == "on_import"
        assert len(rule.conditions) == 1
        assert len(rule.actions) == 1


class TestConditions:
    """Tests for rule conditions."""
    
    def test_path_contains_condition(self):
        """Test PathContainsCondition."""
        from src.ucorefs.rules.conditions import PathContainsCondition
        from src.ucorefs.models.file_record import FileRecord
        
        condition = PathContainsCondition("vacation")
        
        file1 = FileRecord(path="/photos/vacation/beach.jpg")
        file2 = FileRecord(path="/photos/work/office.jpg")
        
        assert condition.evaluate(file1) == True
        assert condition.evaluate(file2) == False
    
    def test_extension_in_condition(self):
        """Test ExtensionInCondition."""
        from src.ucorefs.rules.conditions import ExtensionInCondition
        from src.ucorefs.models.file_record import FileRecord
        
        condition = ExtensionInCondition(["jpg", "png"])
        
        file1 = FileRecord(extension="jpg")
        file2 = FileRecord(extension="txt")
        
        assert condition.evaluate(file1) == True
        assert condition.evaluate(file2) == False
    
    def test_rating_gte_condition(self):
        """Test RatingGteCondition."""
        from src.ucorefs.rules.conditions import RatingGteCondition
        from src.ucorefs.models.file_record import FileRecord
        
        condition = RatingGteCondition(4)
        
        file1 = FileRecord(rating=5)
        file2 = FileRecord(rating=3)
        
        assert condition.evaluate(file1) == True
        assert condition.evaluate(file2) == False


class TestActions:
    """Tests for rule actions."""
    
    @pytest.mark.asyncio
    async def test_add_tag_action(self):
        """Test AddTagAction."""
        from src.ucorefs.rules.actions import AddTagAction
        from src.ucorefs.models.file_record import FileRecord
        
        tag_id = ObjectId()
        action = AddTagAction(str(tag_id))
        
        file = FileRecord(tag_ids=[])
        file.save = AsyncMock()
        
        result = await action.execute(file)
        
        assert result == True
        assert tag_id in file.tag_ids
    
    @pytest.mark.asyncio
    async def test_set_rating_action(self):
        """Test SetRatingAction."""
        from src.ucorefs.rules.actions import SetRatingAction
        from src.ucorefs.models.file_record import FileRecord
        
        action = SetRatingAction(5)
        
        file = FileRecord(rating=0)
        file.save = AsyncMock()
        
        result = await action.execute(file)
        
        assert result == True
        assert file.rating == 5


class TestRulesEngine:
    """Tests for RulesEngine."""
    
    @pytest.fixture
    def mock_locator(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_config(self):
        return MagicMock()
    
    @pytest.mark.asyncio
    async def test_rules_engine_initialize(self, mock_locator, mock_config):
        """Test RulesEngine initialization."""
        from src.ucorefs.rules.engine import RulesEngine
        
        engine = RulesEngine(mock_locator, mock_config)
        await engine.initialize()
        
        assert engine.is_ready == True
    
    @pytest.mark.asyncio
    async def test_evaluate_conditions_all_match(self, mock_locator, mock_config):
        """Test condition evaluation with all matching."""
        from src.ucorefs.rules.engine import RulesEngine
        from src.ucorefs.rules.models import Rule
        from src.ucorefs.models.file_record import FileRecord
        
        engine = RulesEngine(mock_locator, mock_config)
        
        rule = Rule(
            conditions=[
                {"type": "extension_in", "params": {"extensions": ["jpg"]}},
                {"type": "rating_gte", "params": {"threshold": 3}}
            ]
        )
        
        file = FileRecord(extension="jpg", rating=4)
        
        result = await engine._evaluate_conditions(rule, file, {})
        
        assert result == True
    
    @pytest.mark.asyncio
    async def test_evaluate_conditions_one_fails(self, mock_locator, mock_config):
        """Test condition evaluation with one failing."""
        from src.ucorefs.rules.engine import RulesEngine
        from src.ucorefs.rules.models import Rule
        from src.ucorefs.models.file_record import FileRecord
        
        engine = RulesEngine(mock_locator, mock_config)
        
        rule = Rule(
            conditions=[
                {"type": "extension_in", "params": {"extensions": ["jpg"]}},
                {"type": "rating_gte", "params": {"threshold": 5}}
            ]
        )
        
        file = FileRecord(extension="jpg", rating=3)  # Rating too low
        
        result = await engine._evaluate_conditions(rule, file, {})
        
        assert result == False
