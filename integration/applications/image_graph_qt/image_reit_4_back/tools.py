from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List

from diskcache import Cache, Index
from loguru import logger

from constants import PAIRS_CACHE_PATH, TRUESKILL_CACHE_PATH, ITEMS_CACHE_PATH, \
    DEFAULT_MU, MODEL_SIGMA


class CacheKeys:
    """Centralized cache key management"""

    @staticmethod
    def trueskill_mu(record_id: str) -> str:
        return f"{record_id}_mu"

    @staticmethod
    def trueskill_sigma(record_id: str) -> str:
        return f"{record_id}_sigma"

    @staticmethod
    def item_mu(record_id: str) -> str:
        return f"{record_id}_mu"

    @staticmethod
    def item_sigma(record_id: str) -> str:
        return f"{record_id}_sigma"

    @staticmethod
    def pair(first_id: str, second_id: str) -> str:
        return f"{first_id}_{second_id}"


class DataChangeEvent(Enum):
    """Events for data changes"""
    RATING_UPDATED = "rating_updated"
    RECORD_ADDED = "record_added"
    RECORD_REMOVED = "record_removed"
    CACHE_CLEARED = "cache_cleared"


@dataclass
class DataChangeNotification:
    """Notification for data changes"""
    event_type: DataChangeEvent
    record_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class CacheManager:
    """Centralized cache management system"""

    def __init__(self):
        self.pairs_cache = Cache(PAIRS_CACHE_PATH)
        self.pairs_index = Index(PAIRS_CACHE_PATH)
        self.trueskill_cache = Cache(TRUESKILL_CACHE_PATH)
        self.items_cache = Cache(ITEMS_CACHE_PATH)

    def get_trueskill_values(self, record_id: str) -> Tuple[float, float]:
        """Get cached TrueSkill values for a record"""
        mu_key = CacheKeys.trueskill_mu(record_id)
        sigma_key = CacheKeys.trueskill_sigma(record_id)

        mu = self.trueskill_cache.get(mu_key)
        sigma = self.trueskill_cache.get(sigma_key)

        if mu is None or sigma is None:
            # Fallback to record values with bounds checking
            mu = DEFAULT_MU
            sigma = MODEL_SIGMA
        else:
            # Ensure sigma is within reasonable bounds
            sigma = min(sigma, MODEL_SIGMA)  # Cap sigma at initial value
            mu = max(mu, 0.1)  # Ensure mu doesn't go too low

        return mu, sigma

    def set_trueskill_values(self, record_id: str, mu: float, sigma: float):
        """Set cached TrueSkill values for a record"""
        mu_key = CacheKeys.trueskill_mu(record_id)
        sigma_key = CacheKeys.trueskill_sigma(record_id)

        self.trueskill_cache.set(mu_key, mu)
        self.trueskill_cache.set(sigma_key, sigma)

    def clear_trueskill_cache(self, record_id: str):
        """Clear cached TrueSkill values for a record"""
        mu_key = CacheKeys.trueskill_mu(record_id)
        sigma_key = CacheKeys.trueskill_sigma(record_id)

        self.trueskill_cache.delete(mu_key)
        self.trueskill_cache.delete(sigma_key)

    def get_item_rating(self, record_id: str) -> float:
        """Get cached item rating"""
        mu_key = CacheKeys.item_mu(record_id)
        sigma_key = CacheKeys.item_sigma(record_id)

        mu = self.items_cache.get(mu_key)
        sigma = self.items_cache.get(sigma_key)

        if mu is None or sigma is None:
            return DEFAULT_MU

        return mu - 3 * sigma

    def set_item_rating(self, record_id: str, mu: float, sigma: float):
        """Set cached item rating"""
        mu_key = CacheKeys.item_mu(record_id)
        sigma_key = CacheKeys.item_sigma(record_id)

        self.items_cache.set(mu_key, mu)
        self.items_cache.set(sigma_key, sigma)

    def clear_item_cache(self, record_id: str):
        """Clear cached item rating"""
        mu_key = CacheKeys.item_mu(record_id)
        sigma_key = CacheKeys.item_sigma(record_id)

        self.items_cache.delete(mu_key)
        self.items_cache.delete(sigma_key)

    def get_pair_winner(self, first_id: str, second_id: str) -> Optional[Tuple[str, str]]:
        """Get cached pair comparison result"""
        pair_key = CacheKeys.pair(first_id, second_id)
        reverse_pair_key = CacheKeys.pair(second_id, first_id)

        winner_id = self.pairs_cache.get(pair_key)
        if winner_id is not None:
            return winner_id, (first_id if winner_id != first_id else second_id)

        winner_id = self.pairs_cache.get(reverse_pair_key)
        if winner_id is not None:
            return winner_id, (first_id if winner_id != first_id else second_id)

        return None

    def set_pair_winner(self, first_id: str, second_id: str, winner_id: str):
        """Set cached pair comparison result"""
        pair_key = CacheKeys.pair(first_id, second_id)
        reverse_pair_key = CacheKeys.pair(second_id, first_id)

        self.pairs_cache.set(pair_key, winner_id)
        self.pairs_cache.set(reverse_pair_key, winner_id)

        # Update index for quick lookup
        loser_id = first_id if winner_id != first_id else second_id
        self.pairs_index[winner_id] = loser_id

    def clear_all_caches(self):
        """Clear all caches"""
        self.pairs_cache.clear()
        self.trueskill_cache.clear()
        self.items_cache.clear()
        logger.info("All caches cleared")

    def clear_record_caches(self, record_id: str):
        """Clear all caches related to a specific record"""
        self.clear_trueskill_cache(record_id)
        self.clear_item_cache(record_id)
        logger.debug(f"Cleared all caches for record {record_id}")

    def get_wins_for_record(self, record_id: str) -> List[Tuple[str, str]]:
        """Get all records this record has beaten

        Returns:
            List of tuples (opponent_id, pair_key) for records this record won against
        """
        wins = []
        try:
            for key in self.pairs_cache:
                winner_id = self.pairs_cache.get(key)
                if winner_id == record_id:
                    parts = key.split('_')
                    if len(parts) >= 2:
                        first_id = '_'.join(parts[:-1]) if len(parts) > 2 else parts[0]
                        second_id = parts[-1]
                        opponent_id = first_id if first_id != record_id else second_id
                        if opponent_id != record_id:
                            wins.append((opponent_id, key))
            unique_wins = []
            seen_opponents = set()
            for opponent_id, pair_key in wins:
                if opponent_id not in seen_opponents:
                    unique_wins.append((opponent_id, pair_key))
                    seen_opponents.add(opponent_id)
            return unique_wins
        except Exception as e:
            logger.error(f"Error getting wins for record {record_id}: {e}")
            return []

    def get_losses_for_record(self, record_id: str) -> List[Tuple[str, str]]:
        """Get all records this record has lost to

        Returns:
            List of tuples (opponent_id, pair_key) for records this record lost to
        """
        losses = []
        try:
            for key in self.pairs_cache:
                winner_id = self.pairs_cache.get(key)
                if winner_id != record_id:
                    parts = key.split('_')
                    if len(parts) >= 2:
                        first_id = '_'.join(parts[:-1]) if len(parts) > 2 else parts[0]
                        second_id = parts[-1]
                        if first_id == record_id or second_id == record_id:
                            opponent_id = winner_id
                            if opponent_id != record_id:
                                losses.append((opponent_id, key))
            unique_losses = []
            seen_opponents = set()
            for opponent_id, pair_key in losses:
                if opponent_id not in seen_opponents:
                    unique_losses.append((opponent_id, pair_key))
                    seen_opponents.add(opponent_id)
            return unique_losses
        except Exception as e:
            logger.error(f"Error getting losses for record {record_id}: {e}")
            return []

    def delete_cached_pair_relationship(self, first_id: str, second_id: str):
        """Delete specific pair relationship from cache"""
        try:
            pair_key = CacheKeys.pair(first_id, second_id)
            reverse_pair_key = CacheKeys.pair(second_id, first_id)
            self.pairs_cache.delete(pair_key)
            self.pairs_cache.delete(reverse_pair_key)
            if first_id in self.pairs_index:
                del self.pairs_index[first_id]
            if second_id in self.pairs_index:
                del self.pairs_index[second_id]
            logger.debug(f"Deleted cached pair relationship: {first_id} vs {second_id}")
        except Exception as e:
            logger.error(f"Error deleting cached pair relationship {first_id}-{second_id}: {e}")

    def clear_rating_caches_only(self, record_id: str):
        """Clear only TrueSkill and item caches, preserve pairs cache

        This method should be used when resetting sigma/mu values to preserve
        valuable comparison history stored in the pairs cache while ensuring
        rating calculations are refreshed.
        """
        self.clear_trueskill_cache(record_id)
        self.clear_item_cache(record_id)
        logger.debug(f"Cleared rating caches (preserving pairs cache) for record {record_id}")

    def clean_cache_duplicates(self, valid_record_ids: List[str]) -> Dict[str, int]:
        """Clean duplicate and orphaned entries from all caches
        
        Args:
            valid_record_ids: List of valid record IDs that should be kept
            
        Returns:
            Dictionary with statistics about cleaned entries
        """
        stats = {
            'orphaned_pairs': 0,
            'inconsistent_pairs': 0,
            'orphaned_trueskill': 0,
            'orphaned_items': 0,
            'total_cleaned': 0
        }
        
        valid_ids_set = set(valid_record_ids)
        
        try:
            # 1. Clean orphaned and inconsistent pairs
            pairs_to_delete = []
            pair_winners = {}  # Track winners for each normalized pair
            
            for key in self.pairs_cache:
                winner_id = self.pairs_cache.get(key)
                
                # Parse pair key
                parts = key.split('_')
                if len(parts) < 2:
                    pairs_to_delete.append(key)
                    stats['orphaned_pairs'] += 1
                    continue
                
                first_id = '_'.join(parts[:-1]) if len(parts) > 2 else parts[0]
                second_id = parts[-1]
                
                # Check if both records exist
                if first_id not in valid_ids_set or second_id not in valid_ids_set:
                    pairs_to_delete.append(key)
                    stats['orphaned_pairs'] += 1
                    continue
                
                # Check for inconsistent pairs (forward/reverse don't match)
                normalized_key = tuple(sorted([first_id, second_id]))
                if normalized_key in pair_winners:
                    # We've seen this pair before, check consistency
                    if pair_winners[normalized_key] != winner_id:
                        logger.warning(f"Inconsistent pair found: {key} - removing")
                        pairs_to_delete.append(key)
                        stats['inconsistent_pairs'] += 1
                else:
                    pair_winners[normalized_key] = winner_id
            
            # Delete orphaned/inconsistent pairs
            for key in pairs_to_delete:
                self.pairs_cache.delete(key)
            
            # 2. Clean orphaned TrueSkill cache entries
            trueskill_keys_to_delete = []
            for key in self.trueskill_cache:
                # Extract record_id from key (remove _mu or _sigma suffix)
                if key.endswith('_mu') or key.endswith('_sigma'):
                    record_id = key.rsplit('_', 1)[0]
                    if record_id not in valid_ids_set:
                        trueskill_keys_to_delete.append(key)
                        stats['orphaned_trueskill'] += 1
            
            for key in trueskill_keys_to_delete:
                self.trueskill_cache.delete(key)
            
            # 3. Clean orphaned items cache entries
            items_keys_to_delete = []
            for key in self.items_cache:
                # Extract record_id from key (remove _mu or _sigma suffix)
                if key.endswith('_mu') or key.endswith('_sigma'):
                    record_id = key.rsplit('_', 1)[0]
                    if record_id not in valid_ids_set:
                        items_keys_to_delete.append(key)
                        stats['orphaned_items'] += 1
            
            for key in items_keys_to_delete:
                self.items_cache.delete(key)
            
            # Calculate total
            stats['total_cleaned'] = (
                stats['orphaned_pairs'] + 
                stats['inconsistent_pairs'] + 
                stats['orphaned_trueskill'] + 
                stats['orphaned_items']
            )
            
            logger.info(f"Cache cleanup completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error cleaning cache duplicates: {e}")
            return stats
