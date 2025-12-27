"""
UCoreFS - Aggregation Pipelines

Aggregation queries for statistics and grouping.
"""
from typing import Any, Dict, List
from datetime import datetime


class Aggregation:
    """
    Aggregation pipeline builder.
    
    Supports:
    - Group by tag, album, date, rating
    - Statistics (count, size, avg_rating)
    - Date histograms
    """
    
    @staticmethod
    def group_by_tag() -> List[Dict[str, Any]]:
        """
        Group files by tag with counts.
        
        Returns:
            Aggregation pipeline
        """
        return [
            {"$unwind": "$tag_ids"},
            {
                "$group": {
                    "_id": "$tag_ids",
                    "count": {"$sum": 1},
                    "total_size": {"$sum": "$size_bytes"}
                }
            },
            {"$sort": {"count": -1}}
        ]
    
    @staticmethod
    def group_by_album() -> List[Dict[str, Any]]:
        """
        Group files by album with counts.
        
        Returns:
            Aggregation pipeline
        """
        return [
            {"$unwind": "$album_ids"},
            {
                "$group": {
                    "_id": "$album_ids",
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"count": -1}}
        ]
    
    @staticmethod
    def group_by_rating() -> List[Dict[str, Any]]:
        """
        Group files by rating.
        
        Returns:
            Aggregation pipeline
        """
        return [
            {
                "$group": {
                    "_id": "$rating",
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": -1}}
        ]
    
    @staticmethod
    def date_histogram(field: str = "created_at", interval: str = "month") -> List[Dict[str, Any]]:
        """
        Create date histogram.
        
        Args:
            field: Date field
            interval: month, day, year
            
        Returns:
            Aggregation pipeline
        """
        date_format = {
            "year": "%Y",
            "month": "%Y-%m",
            "day": "%Y-%m-%d"
        }.get(interval, "%Y-%m")
        
        return [
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": date_format,
                            "date": f"${field}"
                        }
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]
    
    @staticmethod
    def statistics() -> List[Dict[str, Any]]:
        """
        Calculate overall statistics.
        
        Returns:
            Aggregation pipeline
        """
        return [
            {
                "$group": {
                    "_id": None,
                    "total_files": {"$sum": 1},
                    "total_size": {"$sum": "$size_bytes"},
                    "avg_rating": {"$avg": "$rating"},
                    "max_rating": {"$max": "$rating"},
                    "min_rating": {"$min": "$rating"}
                }
            }
        ]
    
    @staticmethod
    def group_by_file_type() -> List[Dict[str, Any]]:
        """
        Group files by type.
        
        Returns:
            Aggregation pipeline
        """
        return [
            {
                "$group": {
                    "_id": "$file_type",
                    "count": {"$sum": 1},
                    "total_size": {"$sum": "$size_bytes"}
                }
            },
            {"$sort": {"count": -1}}
        ]
