Multi Agent Tools
=================

**Configuration File**: ``multi_agent_tools.json``
**Tool Type**: Local
**Tools Count**: 6

This page contains all tools defined in the ``multi_agent_tools.json`` configuration file.

Available Tools
---------------

**IntentAnalyzerAgent** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI agent that analyzes user research intent and creates comprehensive search plans

.. dropdown:: IntentAnalyzerAgent tool specification

   **Tool Information:**

   * **Name**: ``IntentAnalyzerAgent``
   * **Type**: ``AgenticTool``
   * **Description**: AI agent that analyzes user research intent and creates comprehensive search plans

   **Parameters:**

   * ``user_query`` (string) (required)
     The research query to analyze

   * ``context`` (string) (optional)
     Context information from previous steps

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "IntentAnalyzerAgent",
          "arguments": {
              "user_query": "example_value"
          }
      }
      result = tu.run(query)


**KeywordExtractorAgent** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI agent that extracts and refines search keywords for research plans

.. dropdown:: KeywordExtractorAgent tool specification

   **Tool Information:**

   * **Name**: ``KeywordExtractorAgent``
   * **Type**: ``AgenticTool``
   * **Description**: AI agent that extracts and refines search keywords for research plans

   **Parameters:**

   * ``plan_title`` (string) (required)
     The title of the search plan

   * ``plan_description`` (string) (required)
     The description of the search plan

   * ``current_keywords`` (string) (required)
     Current keywords for the plan (comma-separated)

   * ``context`` (string) (optional)
     Context information from previous steps

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "KeywordExtractorAgent",
          "arguments": {
              "plan_title": "example_value",
              "plan_description": "example_value",
              "current_keywords": "example_value"
          }
      }
      result = tu.run(query)


**MultiAgentLiteratureSearch** (Type: ComposeTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-agent literature search system that uses AI agents to analyze intent, extract keywords, exe...

.. dropdown:: MultiAgentLiteratureSearch tool specification

   **Tool Information:**

   * **Name**: ``MultiAgentLiteratureSearch``
   * **Type**: ``ComposeTool``
   * **Description**: Multi-agent literature search system that uses AI agents to analyze intent, extract keywords, execute parallel searches, summarize results, and check quality iteratively

   **Parameters:**

   * ``query`` (string) (required)
     The research query to search for

   * ``max_iterations`` (integer) (optional)
     Maximum number of iterations (default: 3)

   * ``quality_threshold`` (number) (optional)
     Quality threshold for completion (default: 0.7)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "MultiAgentLiteratureSearch",
          "arguments": {
              "query": "example_value"
          }
      }
      result = tu.run(query)


**OverallSummaryAgent** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI agent that generates comprehensive overall summary of multi-agent search results

.. dropdown:: OverallSummaryAgent tool specification

   **Tool Information:**

   * **Name**: ``OverallSummaryAgent``
   * **Type**: ``AgenticTool``
   * **Description**: AI agent that generates comprehensive overall summary of multi-agent search results

   **Parameters:**

   * ``user_query`` (string) (required)
     The original research query

   * ``user_intent`` (string) (required)
     The analyzed user intent

   * ``total_papers`` (string) (required)
     Total number of papers found

   * ``total_plans`` (string) (required)
     Total number of search plans executed

   * ``iterations`` (string) (required)
     Number of iterations performed

   * ``plan_summaries`` (string) (required)
     Summaries of all search plans

   * ``context`` (string) (optional)
     Context information from previous steps

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "OverallSummaryAgent",
          "arguments": {
              "user_query": "example_value",
              "user_intent": "example_value",
              "total_papers": "example_value",
              "total_plans": "example_value",
              "iterations": "example_value",
              "plan_summaries": "example_value"
          }
      }
      result = tu.run(query)


**QualityCheckerAgent** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI agent that checks search result quality and suggests improvements

.. dropdown:: QualityCheckerAgent tool specification

   **Tool Information:**

   * **Name**: ``QualityCheckerAgent``
   * **Type**: ``AgenticTool``
   * **Description**: AI agent that checks search result quality and suggests improvements

   **Parameters:**

   * ``plans_analysis`` (string) (required)
     Analysis of current search plans and their quality scores

   * ``context`` (string) (optional)
     Context information from previous steps

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "QualityCheckerAgent",
          "arguments": {
              "plans_analysis": "example_value"
          }
      }
      result = tu.run(query)


**ResultSummarizerAgent** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI agent that summarizes search results for research plans

.. dropdown:: ResultSummarizerAgent tool specification

   **Tool Information:**

   * **Name**: ``ResultSummarizerAgent``
   * **Type**: ``AgenticTool``
   * **Description**: AI agent that summarizes search results for research plans

   **Parameters:**

   * ``plan_title`` (string) (required)
     The title of the search plan

   * ``plan_description`` (string) (required)
     The description of the search plan

   * ``paper_count`` (string) (required)
     Number of papers found

   * ``papers_text`` (string) (required)
     Formatted text of the papers to summarize

   * ``context`` (string) (optional)
     Context information from previous steps

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "ResultSummarizerAgent",
          "arguments": {
              "plan_title": "example_value",
              "plan_description": "example_value",
              "paper_count": "example_value",
              "papers_text": "example_value"
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
