Output Summarization Tools
==========================

**Configuration File**: ``output_summarization_tools.json``
**Tool Type**: Local
**Tools Count**: 2

This page contains all tools defined in the ``output_summarization_tools.json`` configuration file.

Available Tools
---------------

**OutputSummarizationComposer** (Type: ComposeTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Composes output summarization workflow by chunking long outputs, processing each chunk with AI su...

.. dropdown:: OutputSummarizationComposer tool specification

   **Tool Information:**

   * **Name**: ``OutputSummarizationComposer``
   * **Type**: ``ComposeTool``
   * **Description**: Composes output summarization workflow by chunking long outputs, processing each chunk with AI summarization, and merging results

   **Parameters:**

   * ``tool_output`` (string) (required)
     The original tool output to be summarized

   * ``query_context`` (string) (required)
     Context about the original query

   * ``tool_name`` (string) (required)
     Name of the tool that generated the output

   * ``chunk_size`` (integer) (optional)
     Size of each chunk for processing

   * ``focus_areas`` (string) (optional)
     Areas to focus on in summarization

   * ``max_summary_length`` (integer) (optional)
     Maximum length of final summary

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "OutputSummarizationComposer",
          "arguments": {
              "tool_output": "example_value",
              "query_context": "example_value",
              "tool_name": "example_value"
          }
      }
      result = tu.run(query)


**ToolOutputSummarizer** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI-powered tool for summarizing long tool outputs, focusing on key information relevant to the or...

.. dropdown:: ToolOutputSummarizer tool specification

   **Tool Information:**

   * **Name**: ``ToolOutputSummarizer``
   * **Type**: ``AgenticTool``
   * **Description**: AI-powered tool for summarizing long tool outputs, focusing on key information relevant to the original query

   **Parameters:**

   * ``tool_output`` (string) (required)
     The original tool output to be summarized

   * ``query_context`` (string) (required)
     Context about the original query that triggered the tool

   * ``tool_name`` (string) (required)
     Name of the tool that generated the output

   * ``focus_areas`` (string) (optional)
     Specific areas to focus on in the summary

   * ``max_length`` (integer) (optional)
     Maximum length of the summary in characters

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "ToolOutputSummarizer",
          "arguments": {
              "tool_output": "example_value",
              "query_context": "example_value",
              "tool_name": "example_value"
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
