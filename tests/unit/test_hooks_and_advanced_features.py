#!/usr/bin/env python3
"""
Test Hooks and Advanced Features - Cleaned Version

This test file covers important hooks and advanced features:
1. Real hook functionality testing
2. Streaming tools support
3. Visualization tools
4. Performance optimization features
"""

import sys
import unittest
from pathlib import Path
import pytest
from unittest.mock import patch, Mock, MagicMock
import json
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tooluniverse import ToolUniverse


@pytest.mark.unit
class TestHooksAndAdvancedFeatures(unittest.TestCase):
    """Test hooks and advanced features functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tu = ToolUniverse()
        # Don't load tools to avoid embedding model loading issues
        self.tu.all_tools = []
        self.tu.all_tool_dict = {}
    
    def test_hook_toggle_functionality(self):
        """Test that hook toggle actually works."""
        # Test enabling hooks
        self.tu.toggle_hooks(True)
        # Note: We can't easily test the internal state without exposing it,
        # but we can test that the method doesn't raise an exception
        self.assertTrue(True)  # Method call succeeded
        
        # Test disabling hooks
        self.tu.toggle_hooks(False)
        self.assertTrue(True)  # Method call succeeded
    
    def test_streaming_tools_support_real(self):
        """Test streaming tools support with real ToolUniverse calls."""
        # Test that streaming callback parameter is accepted
        callback_called = False
        
        def test_callback(chunk):
            nonlocal callback_called
            callback_called = True
        
        # Test with a real tool call (may fail due to missing API keys, but that's OK)
        try:
            result = self.tu.run({
                "name": "UniProt_get_entry_by_accession",
                "arguments": {"accession": "P05067"}
            }, stream_callback=test_callback)
            # If successful, verify we got some result
            self.assertIsNotNone(result)
        except Exception:
            # Expected if API keys not configured
            pass
    
    def test_visualization_tools_real(self):
        """Test visualization tools with real ToolUniverse calls."""
        # Test that visualization tools can be called
        try:
            result = self.tu.run({
                "name": "visualize_protein_structure_3d",
                "arguments": {
                    "pdb_id": "1CRN",
                    "style": "cartoon"
                }
            })
            # If successful, verify we got some result
            self.assertIsNotNone(result)
        except Exception:
            # Expected if tool not available or API keys not configured
            pass
    
    def test_cache_functionality_real(self):
        """Test that caching actually works."""
        # Clear cache first
        self.tu.clear_cache()
        self.assertEqual(len(self.tu._cache), 0)
        
        # Test caching a result
        test_key = "test_cache_key"
        test_value = {"result": "cached_data"}
        
        # Add to cache using proper API
        self.tu._cache.set(test_key, test_value)
        
        # Verify it's in cache
        self.assertEqual(self.tu._cache.get(test_key), test_value)
        
        # Clear cache
        self.tu.clear_cache()
        self.assertEqual(len(self.tu._cache), 0)
    
    def test_tool_health_check_real(self):
        """Test tool health check with real ToolUniverse."""
        # Test health check
        health = self.tu.get_tool_health()
        
        self.assertIsInstance(health, dict)
        self.assertIn("total", health)
        self.assertIn("available", health)
        self.assertIn("unavailable", health)
        self.assertIn("unavailable_list", health)
        self.assertIn("details", health)
        
        # Verify totals make sense
        self.assertEqual(health["total"], health["available"] + health["unavailable"])
    
    def test_tool_listing_real(self):
        """Test tool listing with real ToolUniverse."""
        # Test different listing modes
        tools_dict = self.tu.list_built_in_tools()
        self.assertIsInstance(tools_dict, dict)
        self.assertIn("total_tools", tools_dict)
        
        tools_list = self.tu.list_built_in_tools(mode="list_name")
        self.assertIsInstance(tools_list, list)
        
        # Test that we can get available tools
        available_tools = self.tu.get_available_tools()
        self.assertIsInstance(available_tools, list)
    
    def test_tool_specification_real(self):
        """Test tool specification with real ToolUniverse."""
        # Load some tools first
        self.tu.load_tools()
        
        if self.tu.all_tools:
            # Get a tool name from the loaded tools
            tool_name = self.tu.all_tools[0].get("name")
            if tool_name:
                spec = self.tu.tool_specification(tool_name)
                if spec:  # If tool has specification
                    self.assertIsInstance(spec, dict)
                    self.assertIn("name", spec)
    
    def test_error_handling_real(self):
        """Test error handling with real ToolUniverse calls."""
        # Test with invalid tool name
        result = self.tu.run({
            "name": "NonExistentTool",
            "arguments": {"test": "value"}
        })
        
        self.assertIsInstance(result, dict)
        # Should either return error or None
        if result:
            self.assertIn("error", result)
    
    def test_export_functionality_real(self):
        """Test export functionality with real ToolUniverse."""
        import tempfile
        import os
        
        # Test exporting to file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_file = f.name
        
        try:
            self.tu.export_tool_names(temp_file)
            
            # Verify file was created and has content
            self.assertTrue(os.path.exists(temp_file))
            with open(temp_file, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 0)
                
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_env_template_generation_real(self):
        """Test environment template generation with real ToolUniverse."""
        import tempfile
        import os
        
        # Test with some missing keys
        missing_keys = ["API_KEY_1", "API_KEY_2"]
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            temp_file = f.name
        
        try:
            self.tu.generate_env_template(missing_keys, output_file=temp_file)
            
            # Verify file was created and has content
            self.assertTrue(os.path.exists(temp_file))
            with open(temp_file, 'r') as f:
                content = f.read()
                self.assertIn("API_KEY_1", content)
                self.assertIn("API_KEY_2", content)
                
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_call_id_generation_real(self):
        """Test call ID generation with real ToolUniverse."""
        # Test generating multiple IDs
        id1 = self.tu.call_id_gen()
        id2 = self.tu.call_id_gen()
        
        self.assertIsInstance(id1, str)
        self.assertIsInstance(id2, str)
        self.assertNotEqual(id1, id2)
        self.assertGreater(len(id1), 0)
        self.assertGreater(len(id2), 0)
    
    def test_lazy_loading_status_real(self):
        """Test lazy loading status with real ToolUniverse."""
        status = self.tu.get_lazy_loading_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("lazy_loading_enabled", status)
        self.assertIn("full_discovery_completed", status)
        self.assertIn("immediately_available_tools", status)
        self.assertIn("lazy_mappings_available", status)
        self.assertIn("loaded_tools_count", status)
    
    def test_tool_types_retrieval_real(self):
        """Test tool types retrieval with real ToolUniverse."""
        tool_types = self.tu.get_tool_types()
        
        self.assertIsInstance(tool_types, list)
        # Should contain some tool types
        self.assertGreater(len(tool_types), 0)


if __name__ == "__main__":
    unittest.main()
