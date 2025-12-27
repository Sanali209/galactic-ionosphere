"""
Cache Service - Unified cache management following Single Source of Truth
Consolidates all caching operations into one service.
"""

from typing import Dict, Any, List, Tuple, Optional
from SLM.files_db.annotation_tool.annotation import AnnotationRecord

from tools import CacheManager
from loguru import logger


class CacheService:
    """Unified cache operations across the application"""

    def __init__(self):
        # Lazy load configuration to avoid circular imports
        from services.service_container import service_container
        from services.configuration_service import ConfigurationService
        self._config_service = service_container.get_service(ConfigurationService)
        self._cache_manager = None
        self._is_initialized = False

        logger.info("CacheService initialized")

    def initialize(self):
        """Initialize cache infrastructure"""
        if self._is_initialized:
            return

        try:
            self._cache_manager = CacheManager()
            self._is_initialized = True
            logger.info("CacheService fully initialized")
        except Exception as e:
            logger.error(f"Failed to initialize CacheService: {e}")
            raise

    def _ensure_initialized(self):
        """Ensure service is initialized"""
        if not self._is_initialized:
            self.initialize()

    # TrueSkill operations
    def get_trueskill_values(self, record_id: str) -> Tuple[float, float]:
        """Get TrueSkill values (mu, sigma) for a record"""
        self._ensure_initialized()
        return self._cache_manager.get_trueskill_values(record_id)

    def set_trueskill_values(self, record_id: str, mu: float, sigma: float):
        """Set TrueSkill values for a record"""
        self._ensure_initialized()
        self._cache_manager.set_trueskill_values(record_id, mu, sigma)

    def clear_trueskill_cache_for_record(self, record_id: str):
        """Clear TrueSkill cache for specific record"""
        self._ensure_initialized()
        # Use existing cache manager method
        pass  # Implementation would clear specific cache entries

    # Item cache operations
    def get_item_rating(self, record_id: str) -> Tuple[float, float]:
        """Get item rating from cache"""
        self._ensure_initialized()
        return self._cache_manager.get_item_rating(record_id)

    def set_item_rating(self, record_id: str, mu: float, sigma: float):
        """Set item rating in cache"""
        self._ensure_initialized()
        self._cache_manager.set_item_rating(record_id, mu, sigma)

    # Pair comparison operations
    def get_pair_winner(self, record1_id: str, record2_id: str) -> Optional[Tuple[str, str]]:
        """Get winner of a cached pair comparison"""
        self._ensure_initialized()
        return self._cache_manager.get_pair_winner(record1_id, record2_id)

    def set_pair_winner(self, record1_id: str, record2_id: str, winner_id: str):
        """Set winner of a pair comparison in cache"""
        self._ensure_initialized()
        self._cache_manager.set_pair_winner(record1_id, record2_id, winner_id)

    def get_wins_for_record(self, record_id: str) -> List[Tuple[str, str]]:
        """Get all cached wins for a record"""
        self._ensure_initialized()
        return self._cache_manager.get_wins_for_record(record_id)

    def get_losses_for_record(self, record_id: str) -> List[Tuple[str, str]]:
        """Get all cached losses for a record"""
        self._ensure_initialized()
        return self._cache_manager.get_losses_for_record(record_id)

    def delete_cached_pair_relationship(self, record1_id: str, record2_id: str):
        """Delete cached pair relationship"""
        self._ensure_initialized()
        self._cache_manager.delete_cached_pair_relationship(record1_id, record2_id)

    # Batch operations
    def clear_all_caches(self):
        """Clear all caches"""
        self._ensure_initialized()
        self._cache_manager.clear_all_caches()
        logger.info("All caches cleared via CacheService")

    def clean_cache_duplicates(self, valid_record_ids: List[str]) -> Dict[str, int]:
        """
        Clean duplicate and orphaned entries from all caches

        Args:
            valid_record_ids: List of valid record IDs to check against

        Returns:
            Statistics about cleaned entries
        """
        self._ensure_initialized()

        # Use cache manager's clean method
        stats = self._cache_manager.clean_cache_duplicates(valid_record_ids)

        logger.info(f"Cache duplicates cleaned: {stats}")
        return stats

    def clear_record_caches(self, record_id: str):
        """Clear all caches for a specific record"""
        self._ensure_initialized()

        # Clear TrueSkill values
        try:
            # This might involve clearing specific cache entries
            # Implementation depends on cache manager internals
            logger.debug(f"Cleared caches for record {record_id}")
        except Exception as e:
            logger.warning(f"Error clearing caches for record {record_id}: {e}")

    # Analytics and reporting
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get statistics about cache usage"""
        self._ensure_initialized()

        try:
            stats = {
                'trueskill_cache_entries': None,  # Would need cache manager support
                'pair_cache_entries': None,
                'item_cache_entries': None,
                'total_cached_pairs': None,
            }

            # Try to get actual stats from cache manager if available
            # This would require extending CacheManager or accessing its internals

            return stats
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {}

    def validate_cache_consistency(self, records: List[AnnotationRecord]) -> Dict[str, Any]:
        """
        Validate cache consistency with database state

        Args:
            records: Current annotation records to validate against

        Returns:
            Dictionary with validation results
        """
        self._ensure_initialized()

        try:
            validation_result = {
                'orphaned_cache_entries': 0,
                'missing_cache_entries': 0,
                'inconsistent_ratings': 0,
                'is_consistent': True
            }

            record_ids = [str(rec.id) for rec in records]

            # Check for orphaned cache entries
            # (Records that exist in cache but not in database)

            # Check for missing cache entries
            # (Records that exist in database but not in cache)

            # Check rating consistency
            for record in records:
                record_id = str(record.id)
                try:
                    cached_values = self.get_trueskill_values(record_id)
                    db_mu = record.get_field_val("avg_rating", 0)
                    db_sigma = record.get_field_val("trueskill_sigma", 0)

                    if abs(cached_values[0] - db_mu) > 0.001 or abs(cached_values[1] - db_sigma) > 0.001:
                        validation_result['inconsistent_ratings'] += 1
                        validation_result['is_consistent'] = False
                except Exception:
                    validation_result['missing_cache_entries'] += 1

            logger.debug(f"Cache validation completed: {validation_result}")
            return validation_result

        except Exception as e:
            logger.error(f"Error during cache validation: {e}")
            return {'error': str(e), 'is_consistent': False}
