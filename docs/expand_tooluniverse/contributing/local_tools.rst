Contributing Local Tools to ToolUniverse
=========================================

This guide covers how to contribute local Python tools to the ToolUniverse repository. Local tools run within the ToolUniverse process and are available to all users.

.. note::
   **Key Difference**: Contributing to the repository requires additional steps compared to using tools locally. The most critical step is modifying ``__init__.py`` in 4 specific locations.

Quick Overview
--------------

**10 Steps to Contribute a Local Tool:**

1. **Environment Setup** - Fork, clone, install dependencies
2. **Create Tool File** - Python class in ``src/tooluniverse/``
3. **Register Tool** - Use ``@register_tool('Type')`` decorator
4. **Create Config** - JSON file in ``data/xxx_tools.json``
5. **🔑 Modify __init__.py** - Add tool in 4 locations (critical!)
6. **Write Tests** - >90% coverage required
7. **Code Quality** - Pre-commit hooks (automatic)
8. **Documentation** - Docstrings and examples
9. **Create Examples** - Working examples in ``examples/``
10. **Submit PR** - Follow contribution guidelines

Step-by-Step Guide
------------------

Step 1: Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Fork the repository on GitHub first
   git clone https://github.com/yourusername/ToolUniverse.git
   cd ToolUniverse
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   
   # Install pre-commit hooks
   ./setup_precommit.sh

Step 2: Create Tool File
~~~~~~~~~~~~~~~~~~~~~~~~~

Create your tool file in ``src/tooluniverse/xxx_tool.py``:

.. code-block:: python

   from tooluniverse.tool_registry import register_tool
   from tooluniverse.base_tool import BaseTool
   from typing import Dict, Any

   @register_tool('MyNewTool')  # Note: No config here for contributions
   class MyNewTool(BaseTool):
       """My new tool for ToolUniverse."""
       
       def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
           """Execute the tool."""
           # Your tool logic here
           input_value = arguments.get('input', '')
           return {
               "result": input_value.upper(),
               "success": True
           }
       
       def validate_input(self, **kwargs) -> None:
           """Validate input parameters."""
           input_val = kwargs.get('input')
           if not input_val:
               raise ValueError("Input is required")

Step 3: Register Tool
~~~~~~~~~~~~~~~~~~~~~~

The ``@register_tool('MyNewTool')`` decorator registers your tool class. Note that for contributions, we don't include the config in the decorator - that goes in a separate JSON file.

Step 4: Create Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create or edit ``src/tooluniverse/data/xxx_tools.json``:

.. code-block:: json

   [
     {
       "name": "my_new_tool",
       "type": "MyNewTool",
       "description": "Convert text to uppercase",
       "parameter": {
         "type": "object",
         "properties": {
           "input": {
             "type": "string",
             "description": "Text to convert to uppercase"
           }
         },
         "required": ["input"]
       },
       "examples": [
         {
           "description": "Convert text to uppercase",
           "arguments": {"input": "hello world"}
         }
       ],
       "tags": ["text", "utility"],
       "author": "Your Name <your.email@example.com>",
       "version": "1.0.0"
     }
   ]

Step 5: 🔑 Modify __init__.py (Critical!)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the most important step that's often missed. You must modify ``src/tooluniverse/__init__.py`` in **4 specific locations**:

**Location 1 (~105-165 lines): Add type declaration**
.. code-block:: python

   # Find the section with other tool type declarations
   MyNewTool: Any

**Location 2 (~173-258 lines): Add import statement (non-lazy section)**
.. code-block:: python

   # Find the non-lazy loading section
   from .my_new_tool import MyNewTool

**Location 3 (~260-360 lines): Add lazy import proxy**
.. code-block:: python

   # Find the else block for lazy loading
   MyNewTool = _LazyImportProxy("my_new_tool", "MyNewTool")

**Location 4 (~362-449 lines): Add to __all__ list**
.. code-block:: python

   __all__ = [
       # ... existing tools ...
       "MyNewTool",
   ]

**Verification:**
.. code-block:: python

   # Test that your tool can be imported
   from tooluniverse import MyNewTool
   print(MyNewTool)  # Should show the class or lazy proxy

Step 6: Write Tests
~~~~~~~~~~~~~~~~~~~

Create tests in ``tests/unit/test_my_new_tool.py``:

.. code-block:: python

   import pytest
   from tooluniverse.my_new_tool import MyNewTool

   class TestMyNewTool:
       def setup_method(self):
           self.tool = MyNewTool()

       def test_success(self):
           """Test successful execution."""
           result = self.tool.run({"input": "hello"})
           assert result["success"] is True
           assert result["result"] == "HELLO"

       def test_validation(self):
           """Test input validation."""
           with pytest.raises(ValueError):
               self.tool.validate_input(input="")

       def test_empty_input(self):
           """Test empty input handling."""
           result = self.tool.run({"input": ""})
           assert result["success"] is True
           assert result["result"] == ""

Run tests with coverage:
.. code-block:: bash

   pytest tests/unit/test_my_new_tool.py --cov=tooluniverse --cov-report=html

Step 7: Code Quality Check (Automatic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-commit hooks will automatically run when you commit:

.. code-block:: bash

   git add .
   git commit -m "feat: add MyNewTool"
   # Pre-commit will run: Black, Flake8, Ruff, etc.
   # If checks fail, fix the issues and commit again

Step 8: Documentation
~~~~~~~~~~~~~~~~~~~~~

Add comprehensive docstrings to your tool class:

.. code-block:: python

   class MyNewTool(BaseTool):
       """
       Convert text to uppercase.
       
       This tool takes a string input and returns it converted to uppercase.
       Useful for text processing workflows.
       
       Args:
           input (str): The text to convert to uppercase
           
       Returns:
           dict: Result dictionary with 'result' and 'success' keys
           
       Example:
           >>> tool = MyNewTool()
           >>> result = tool.run({"input": "hello"})
           >>> print(result["result"])
           HELLO
       """
```

Step 9: Create Examples
~~~~~~~~~~~~~~~~~~~~~~~~

Create ``examples/my_new_tool_example.py``:

.. code-block:: python

   """Example usage of MyNewTool."""
   
   from tooluniverse import ToolUniverse

   def main():
       # Initialize ToolUniverse
       tu = ToolUniverse()
       tu.load_tools()
       
       # Use the tool
       result = tu.run({
           "name": "my_new_tool",
           "arguments": {"input": "hello world"}
       })
       
       print(f"Result: {result}")
       
       # Test with different inputs
       test_inputs = ["hello", "world", "python"]
       for text in test_inputs:
           result = tu.run({
               "name": "my_new_tool",
               "arguments": {"input": text}
           })
           print(f"'{text}' -> '{result.get('result', 'ERROR')}'")

   if __name__ == "__main__":
       main()

Step 10: Submit Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create feature branch
   git checkout -b feature/add-my-new-tool
   
   # Add all files
   git add src/tooluniverse/my_new_tool.py
   git add src/tooluniverse/data/xxx_tools.json
   git add src/tooluniverse/__init__.py
   git add tests/unit/test_my_new_tool.py
   git add examples/my_new_tool_example.py
   
   # Commit with descriptive message
   git commit -m "feat: add MyNewTool for text processing
   
   - Implement MyNewTool class with uppercase conversion
   - Add comprehensive unit tests with >95% coverage
   - Include usage examples and documentation
   - Support input validation and error handling
   
   Closes #[issue-number]"
   
   # Push and create PR
   git push origin feature/add-my-new-tool

**PR Template:**
.. code-block:: markdown

   ## Description
   
   This PR adds MyNewTool, a new local tool for text processing.
   
   ## Changes Made
   
   - ✅ **Tool Implementation**: Complete MyNewTool class
   - ✅ **Testing**: Unit tests with >95% coverage
   - ✅ **Documentation**: Comprehensive docstrings and examples
   - ✅ **Configuration**: JSON config in data/xxx_tools.json
   - ✅ **Integration**: Modified __init__.py in 4 locations
   
   ## Testing
   
   ```bash
   pytest tests/unit/test_my_new_tool.py --cov=tooluniverse
   python examples/my_new_tool_example.py
   ```
   
   ## Checklist
   
   - [x] Tests pass locally
   - [x] Code follows project style guidelines
   - [x] Documentation is complete
   - [x] __init__.py modified in all 4 locations
   - [x] Examples work as expected

Common Mistakes
----------------

**❌ Most Common: Forgetting to modify __init__.py**
- Tool won't be importable: ``ImportError: cannot import name 'MyNewTool'``
- Solution: Check all 4 locations in __init__.py

**❌ Config in wrong place**
- Don't put config in ``@register_tool()`` decorator
- Put it in ``data/xxx_tools.json`` instead

**❌ Wrong file location**
- Tool file must be in ``src/tooluniverse/``
- Not in your project directory

**❌ Missing tests**
- Coverage must be >90%
- Test both success and error cases

**❌ Import errors**
- Check module name matches file name
- Check class name matches exactly (case-sensitive)

Troubleshooting
---------------

**ImportError: cannot import name 'MyNewTool'**
.. code-block:: python

   # Check if tool is in __all__ list
   from tooluniverse import __all__
   print("MyNewTool" in __all__)  # Should be True
   
   # Check if import statement exists
   # Look for: from .my_new_tool import MyNewTool

**AttributeError: module 'tooluniverse' has no attribute 'MyNewTool'**
- Verify the tool name is in ``__all__`` list
- Check that the tool name matches the class name exactly

**Tool not found when using ToolUniverse**
.. code-block:: python

   # Verify tool loads correctly
   from tooluniverse import ToolUniverse
   tu = ToolUniverse()
   tu.load_tools()
   
   # Check if tool is in the loaded tools
   tools = tu.list_tools()
   print("my_new_tool" in tools)  # Should be True

Next Steps
----------

After successfully contributing your local tool:

* 🚀 **Remote Tools**: :doc:`remote_tools` - Learn about contributing remote tools
* 🔍 **Architecture**: :doc:`../reference/architecture` - Understand ToolUniverse internals
* 📊 **Comparison**: Review the tool type comparison table in :doc:`../contributing/index`

.. tip::
   **Success Tips**: Start with simple tools, test thoroughly, and ask for help in GitHub discussions if you get stuck!
