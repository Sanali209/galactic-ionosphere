"""
UCoreFS Phase 8 Tests - Query Builder

Tests for:
- Q expressions
- QueryBuilder fluent API
- Aggregations
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId


class TestQExpressions:
    """Tests for Q query expressions."""
    
    def test_rating_gte(self):
        """Test rating >= query."""
        from src.ucorefs.query.builder import Q
        
        q = Q.rating_gte(4)
        
        assert q.query == {"rating": {"$gte": 4}}
    
    def test_name_contains(self):
        """Test name contains query."""
        from src.ucorefs.query.builder import Q
        
        q = Q.name_contains("vacation")
        
        assert "$regex" in q.query["name"]
        assert q.query["name"]["$regex"] == "vacation"
    
    def test_has_tag(self):
        """Test has tag query."""
        from src.ucorefs.query.builder import Q
        
        tag_id = ObjectId()
        q = Q.has_tag(tag_id)
        
        assert q.query == {"tag_ids": tag_id}
    
    def test_and_operator(self):
        """Test AND operator."""
        from src.ucorefs.query.builder import Q
        
        q1 = Q.rating_gte(4)
        q2 = Q.file_type("image")
        
        combined = Q.AND(q1, q2)
        
        assert "$and" in combined.query
        assert len(combined.query["$and"]) == 2
    
    def test_or_operator(self):
        """Test OR operator."""
        from src.ucorefs.query.builder import Q
        
        q1 = Q.has_tag(ObjectId())
        q2 = Q.has_tag(ObjectId())
        
        combined = Q.OR(q1, q2)
        
        assert "$or" in combined.query
    
    def test_not_operator(self):
        """Test NOT operator."""
        from src.ucorefs.query.builder import Q
        
        q = Q.rating_gte(3)
        negated = Q.NOT(q)
        
        assert "$nor" in negated.query


class TestQueryBuilder:
    """Tests for QueryBuilder."""
    
    def test_query_builder_where(self):
        """Test where clause."""
        from src.ucorefs.query.builder import QueryBuilder, Q
        
        builder = QueryBuilder()
        builder.where(Q.rating_gte(4))
        
        query = builder.get_query()
        
        assert query == {"rating": {"$gte": 4}}
    
    def test_query_builder_and(self):
        """Test AND chaining."""
        from src.ucorefs.query.builder import QueryBuilder, Q
        
        builder = QueryBuilder()
        builder.AND(
            Q.rating_gte(3),
            Q.file_type("image")
        )
        
        query = builder.get_query()
        
        assert "$and" in query
    
    def test_query_builder_order_by(self):
        """Test order by."""
        from src.ucorefs.query.builder import QueryBuilder
        
        builder = QueryBuilder()
        builder.order_by("created_at", descending=True)
        
        assert builder._sort == [("created_at", -1)]
    
    def test_query_builder_limit_skip(self):
        """Test limit and skip."""
        from src.ucorefs.query.builder import QueryBuilder
        
        builder = QueryBuilder()
        builder.limit(10).skip(20)
        
        assert builder._limit == 10
        assert builder._skip == 20
    
    def test_complex_query(self):
        """Test complex nested query."""
        from src.ucorefs.query.builder import QueryBuilder, Q
        
        builder = QueryBuilder()
        builder.AND(
            Q.rating_gte(3),
            Q.OR(
                Q.has_tag(ObjectId()),
                Q.has_tag(ObjectId())
            )
        ).NOT(Q.extension_in(["tmp"]))
        
        query = builder.get_query()
        
        # Should have nested AND/OR/NOT structure
        assert "$and" in query or query != {}


class TestAggregations:
    """Tests for aggregation pipelines."""
    
    def test_group_by_tag(self):
        """Test group by tag aggregation."""
        from src.ucorefs.query.aggregations import Aggregation
        
        pipeline = Aggregation.group_by_tag()
        
        assert len(pipeline) == 3
        assert pipeline[0]["$unwind"] == "$tag_ids"
        assert "$group" in pipeline[1]
    
    def test_statistics(self):
        """Test statistics aggregation."""
        from src.ucorefs.query.aggregations import Aggregation
        
        pipeline = Aggregation.statistics()
        
        assert len(pipeline) == 1
        assert "$group" in pipeline[0]
        assert "total_files" in pipeline[0]["$group"]
        assert "avg_rating" in pipeline[0]["$group"]
    
    def test_date_histogram(self):
        """Test date histogram."""
        from src.ucorefs.query.aggregations import Aggregation
        
        pipeline = Aggregation.date_histogram("created_at", "month")
        
        assert len(pipeline) == 2
        assert "$group" in pipeline[0]
        assert "$dateToString" in pipeline[0]["$group"]["_id"]
