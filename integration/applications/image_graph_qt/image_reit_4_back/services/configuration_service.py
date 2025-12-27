"""
Configuration Service - Single source of truth for all application configuration
Consolidates constants.py and model_config.py into unified interface.
"""
import os
from typing import Dict, Any, Optional
from loguru import logger


class ApplicationConfig:
    """Unified configuration management"""

    def __init__(self):
        # Load constants
        self.default_job_name: str = "rating_competition"
        self.initial_rating_min: int = 1
        self.initial_rating_max: int = 10
        self.default_rating: float = 5.0
        self.default_mu: float = 25.0
        self.model_sigma: float = 8.333

        # Load specific settings
        self.accessible_quality: float = 0.1
        self.auto_load_round: int = 100000
        self.auto_win_pairs: bool = True
        self.anti_stack_count: int = 3
        self.add_rand_count: int = 100

        # Cache configuration
        self.cache_dir: str = r"D:\data\rc"
        self.pairs_cache_path: str = os.path.join(self.cache_dir, "pairs_cache")
        self.trueskill_cache_path: str = os.path.join(self.cache_dir, "trueskill_cache")
        self.items_cache_path: str = os.path.join(self.cache_dir, "items_cache")

        # Model configuration (from existing model_config.py)
        self.model_sigma_config = None
        self._load_model_config()

        logger.info("ApplicationConfig initialized with consolidated settings")

    def _load_model_config(self):
        """Load model configuration (equivalent to model_config.py)"""
        try:
            # Try to import the existing model config
            from model_config import model_sigma_config as existing_config
            self.model_sigma_config = existing_config
            logger.debug("Loaded existing model sigma configuration")
        except ImportError:
            logger.warning("Could not load existing model config, creating default")
            # Create a minimal config if import fails
            self.model_sigma_config = self._create_default_model_config()

    def _create_default_model_config(self):
        """Create default model config (simplified version)"""
        class DefaultModelConfig:
            def __init__(self):
                self.calculated_sigma = self.model_sigma
                self.use_llm_sigma = False
                self.use_predictions = False

            def get_sigma_for_new_items(self):
                return self.calculated_sigma

            def is_model_loaded(self):
                return False

        return DefaultModelConfig()

    # Getters for backward compatibility
    @property
    def DEFAULT_JOB_NAME(self) -> str:
        return self.default_job_name

    @property
    def INITIAL_RATING_MIN(self) -> int:
        return self.initial_rating_min

    @property
    def INITIAL_RATING_MAX(self) -> int:
        return self.initial_rating_max

    @property
    def DEFAULT_RATING(self) -> float:
        return self.default_rating

    @property
    def DEFAULT_MU(self) -> float:
        return self.default_mu

    @property
    def MODEL_SIGMA(self) -> float:
        return self.model_sigma

    @property
    def ACESSIBLE_QUALITY(self) -> float:
        return self.accessible_quality

    @property
    def AUTO_LOAD_ROUND(self) -> int:
        return self.auto_load_round

    @property
    def AUTO_WIN_PAIRS(self) -> bool:
        return self.auto_win_pairs

    @property
    def ANTI_STACK_COUNT(self) -> int:
        return self.anti_stack_count

    @property
    def ADD_RAND_COUNT(self) -> int:
        return self.add_rand_count

    @property
    def CACHE_DIR(self) -> str:
        return self.cache_dir

    @property
    def PAIRS_CACHE_PATH(self) -> str:
        return self.pairs_cache_path

    @property
    def TRUESKILL_CACHE_PATH(self) -> str:
        return self.trueskill_cache_path

    @property
    def ITEMS_CACHE_PATH(self) -> str:
        return self.items_cache_path

    def get_model_sigma_config(self):
        """Get model sigma configuration"""
        return self.model_sigma_config

    def get_sigma_for_new_items(self) -> float:
        """Get appropriate sigma value for new items"""
        if self.model_sigma_config:
            return self.model_sigma_config.get_sigma_for_new_items()
        return self.model_sigma

    def validate_config(self) -> bool:
        """Validate configuration values"""
        try:
            # Check rating ranges
            assert self.initial_rating_min < self.initial_rating_max
            assert 0 < self.default_rating <= self.initial_rating_max

            # Check mu/sigma values
            assert self.default_mu > 0
            assert self.model_sigma > 0

            # Check paths exist
            os.makedirs(self.cache_dir, exist_ok=True)

            logger.debug("Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_job_creation_data(self) -> Dict[str, Any]:
        """Get data structure for creating new annotation job"""
        return {
            "name": self.default_job_name,
            "type": "multiclass/image",
            "choices": [str(i) for i in range(self.initial_rating_min, self.initial_rating_max + 1)]
        }

    def create_cache_paths(self):
        """Ensure cache directories exist"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.debug(f"Cache directory verified: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            raise


# Global configuration instance
application_config = ApplicationConfig()


class ConfigurationService:
    """Service wrapper for configuration access"""

    def __init__(self):
        self._config = application_config

    def get_config(self) -> ApplicationConfig:
        """Get the application configuration"""
        return self._config

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        try:
            return getattr(self._config, key.lower(), default)
        except AttributeError:
            logger.warning(f"Configuration key not found: {key}")
            return default

    def validate_configuration(self) -> bool:
        """Validate current configuration"""
        return self._config.validate_config()
