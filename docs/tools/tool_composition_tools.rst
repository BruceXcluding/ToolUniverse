Tool Composition Tools
======================

**Configuration File**: ``tool_composition_tools.json``
**Tool Type**: Local
**Tools Count**: 2

This page contains all tools defined in the ``tool_composition_tools.json`` configuration file.

Available Tools
---------------

**ToolCompatibilityAnalyzer** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyzes two tool specifications to determine if one tool's output can be used as input for anoth...

.. dropdown:: ToolCompatibilityAnalyzer tool specification

   **Tool Information:**

   * **Name**: ``ToolCompatibilityAnalyzer``
   * **Type**: ``AgenticTool``
   * **Description**: Analyzes two tool specifications to determine if one tool's output can be used as input for another tool. Returns compatibility information and suggested parameter mappings.

   **Parameters:**

   * ``source_tool`` (string) (required)
     The source tool specification (JSON string with name, description, parameter schema, and example outputs)

   * ``target_tool`` (string) (required)
     The target tool specification (JSON string with name, description, parameter schema)

   * ``analysis_depth`` (string) (required)
     Level of analysis depth - quick for basic compatibility, detailed for parameter mapping, comprehensive for semantic analysis

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "ToolCompatibilityAnalyzer",
          "arguments": {
              "source_tool": "example_value",
              "target_tool": "example_value",
              "analysis_depth": "example_value"
          }
      }
      result = tu.run(query)


**ToolGraphComposer** (Type: ComposeTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Builds a comprehensive graph of tool compatibility relationships in ToolUniverse. Analyzes all av...

.. dropdown:: ToolGraphComposer tool specification

   **Tool Information:**

   * **Name**: ``ToolGraphComposer``
   * **Type**: ``ComposeTool``
   * **Description**: Builds a comprehensive graph of tool compatibility relationships in ToolUniverse. Analyzes all available tools and creates a directed graph showing which tools can be composed together.

   **Parameters:**

   * ``output_path`` (string) (required)
     Path to save the generated graph files (JSON and pickle formats)

   * ``analysis_depth`` (string) (required)
     Level of compatibility analysis to perform

   * ``min_compatibility_score`` (integer) (required)
     Minimum compatibility score to create an edge in the graph

   * ``exclude_categories`` (array) (required)
     Tool categories to exclude from analysis (e.g., ['tool_finder', 'special_tools'])

   * ``max_tools_per_category`` (integer) (required)
     Maximum number of tools to analyze per category (for performance)

   * ``force_rebuild`` (boolean) (required)
     Whether to force rebuild even if cached graph exists

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "ToolGraphComposer",
          "arguments": {
              "output_path": "example_value",
              "analysis_depth": "example_value",
              "min_compatibility_score": 10,
              "exclude_categories": ["item1", "item2"],
              "max_tools_per_category": 10,
              "force_rebuild": true
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
