"""
Unit tests for SystemBundle pattern.
"""
import pytest
from unittest.mock import Mock, call, patch

from src.core.bootstrap import SystemBundle, ApplicationBuilder
from src.core.base_system import BaseSystem


class MockSystemA(BaseSystem):
    """Mock system A for testing."""
    pass


class MockSystemB(BaseSystem):
    """Mock system B for testing."""
    pass


class MockSystemC(BaseSystem):
    """Mock system C for testing."""
    pass


class TestBundle(SystemBundle):
    """Test bundle for unit tests."""
    
    def register(self, builder: "ApplicationBuilder") -> None:
        builder.add_system(MockSystemA)
        builder.add_system(MockSystemB)


class TestBundle2(SystemBundle):
    """Second test bundle for unit tests."""
    
    def register(self, builder: "ApplicationBuilder") -> None:
        builder.add_system(MockSystemC)


class TestSystemBundle:
    """Tests for SystemBundle ABC and ApplicationBuilder.add_bundle()."""
    
    def test_bundle_registers_systems(self):
        """Verify bundle calls add_system for each service."""
        builder = ApplicationBuilder("Test", "config.json")
        bundle = TestBundle()
        
        # Initial state: no custom systems
        assert len(builder._systems) == 0
        
        # Add bundle
        builder.add_bundle(bundle)
        
        # Verify systems were registered
        assert len(builder._systems) == 2
        assert MockSystemA in builder._systems
        assert MockSystemB in builder._systems
    
    def test_add_bundle_returns_builder(self):
        """Verify add_bundle returns self for fluent chaining."""
        builder = ApplicationBuilder("Test", "config.json")
        bundle = TestBundle()
        
        result = builder.add_bundle(bundle)
        
        assert result is builder
    
    def test_multiple_bundles(self):
        """Verify multiple bundles can be chained."""
        builder = ApplicationBuilder("Test", "config.json")
        
        result = (
            builder
            .add_bundle(TestBundle())
            .add_bundle(TestBundle2())
        )
        
        assert result is builder
        assert len(builder._systems) == 3
        assert MockSystemA in builder._systems
        assert MockSystemB in builder._systems
        assert MockSystemC in builder._systems
    
    def test_bundle_preserves_order(self):
        """Verify systems are registered in bundle order."""
        builder = ApplicationBuilder("Test", "config.json")
        builder.add_bundle(TestBundle())
        
        assert builder._systems[0] == MockSystemA
        assert builder._systems[1] == MockSystemB
    
    def test_can_mix_bundles_and_individual_systems(self):
        """Verify bundles work alongside individual add_system calls."""
        builder = ApplicationBuilder("Test", "config.json")
        
        builder.add_system(MockSystemC)
        builder.add_bundle(TestBundle())
        
        assert len(builder._systems) == 3
        assert builder._systems[0] == MockSystemC  # Added first
        assert builder._systems[1] == MockSystemA  # From bundle
        assert builder._systems[2] == MockSystemB  # From bundle


class TestUCoreFSBundle:
    """Tests for UCoreFSBundle import and registration."""
    
    def test_ucorefs_bundle_can_import(self):
        """Verify UCoreFSBundle can be imported without errors."""
        from src.ucorefs.bundle import UCoreFSBundle
        
        bundle = UCoreFSBundle()
        assert bundle is not None
    
    def test_ucorefs_bundle_registers_services(self):
        """Verify UCoreFSBundle registers expected number of services."""
        from src.ucorefs.bundle import UCoreFSBundle
        
        builder = ApplicationBuilder("Test", "config.json")
        bundle = UCoreFSBundle()
        
        bundle.register(builder)
        
        # Should register 17 UCoreFS services
        assert len(builder._systems) == 17


@pytest.mark.skip(reason="uexplorer_src not on path in test environment")
class TestUExplorerUIBundle:
    """Tests for UExplorerUIBundle import and registration.
    
    Note: These tests are skipped because uexplorer_src requires
    specific path setup that's handled in main.py.
    """
    
    def test_uexplorer_bundle_can_import(self):
        """Verify UExplorerUIBundle can be imported without errors."""
        pass
    
    def test_uexplorer_bundle_registers_services(self):
        """Verify UExplorerUIBundle registers expected number of services."""
        pass
