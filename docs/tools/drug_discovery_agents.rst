Drug Discovery Agents
=====================

**Configuration File**: ``drug_discovery_agents.json``
**Tool Type**: Local
**Tools Count**: 7

This page contains all tools defined in the ``drug_discovery_agents.json`` configuration file.

Available Tools
---------------

**ADMETAnalyzerAgent** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI agent that analyzes ADMET data and provides insights on drug-likeness and safety profiles

.. dropdown:: ADMETAnalyzerAgent tool specification

   **Tool Information:**

   * **Name**: ``ADMETAnalyzerAgent``
   * **Type**: ``AgenticTool``
   * **Description**: AI agent that analyzes ADMET data and provides insights on drug-likeness and safety profiles

   **Parameters:**

   * ``compounds`` (string) (required)
     List of compounds to analyze (comma-separated)

   * ``admet_data`` (string) (required)
     ADMET data from computational tools to analyze

   * ``disease_context`` (string) (optional)
     Disease context for ADMET evaluation

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "ADMETAnalyzerAgent",
          "arguments": {
              "compounds": "example_value",
              "admet_data": "example_value"
          }
      }
      result = tu.run(query)


**ClinicalTrialDesignAgent** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI agent that designs clinical trial protocols based on preclinical data and regulatory requirements

.. dropdown:: ClinicalTrialDesignAgent tool specification

   **Tool Information:**

   * **Name**: ``ClinicalTrialDesignAgent``
   * **Type**: ``AgenticTool``
   * **Description**: AI agent that designs clinical trial protocols based on preclinical data and regulatory requirements

   **Parameters:**

   * ``drug_name`` (string) (required)
     Name of the drug candidate

   * ``indication`` (string) (required)
     Disease indication

   * ``preclinical_data`` (string) (optional)
     Preclinical efficacy and safety data

   * ``target_population`` (string) (optional)
     Target patient population

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "ClinicalTrialDesignAgent",
          "arguments": {
              "drug_name": "example_value",
              "indication": "example_value"
          }
      }
      result = tu.run(query)


**CompoundDiscoveryAgent** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI agent that analyzes potential drug compounds using multiple strategies and LLM reasoning

.. dropdown:: CompoundDiscoveryAgent tool specification

   **Tool Information:**

   * **Name**: ``CompoundDiscoveryAgent``
   * **Type**: ``AgenticTool``
   * **Description**: AI agent that analyzes potential drug compounds using multiple strategies and LLM reasoning

   **Parameters:**

   * ``disease_name`` (string) (required)
     Name of the disease

   * ``targets`` (string) (required)
     List of therapeutic targets (comma-separated)

   * ``context`` (string) (optional)
     Additional context or specific requirements

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "CompoundDiscoveryAgent",
          "arguments": {
              "disease_name": "example_value",
              "targets": "example_value"
          }
      }
      result = tu.run(query)


**DiseaseAnalyzerAgent** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI agent that analyzes disease characteristics and identifies potential therapeutic targets using...

.. dropdown:: DiseaseAnalyzerAgent tool specification

   **Tool Information:**

   * **Name**: ``DiseaseAnalyzerAgent``
   * **Type**: ``AgenticTool``
   * **Description**: AI agent that analyzes disease characteristics and identifies potential therapeutic targets using LLM reasoning

   **Parameters:**

   * ``disease_name`` (string) (required)
     Name of the disease to analyze

   * ``context`` (string) (optional)
     Additional context or specific focus areas

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "DiseaseAnalyzerAgent",
          "arguments": {
              "disease_name": "example_value"
          }
      }
      result = tu.run(query)


**DrugInteractionAnalyzerAgent** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI agent that analyzes drug-drug interactions and provides clinical recommendations

.. dropdown:: DrugInteractionAnalyzerAgent tool specification

   **Tool Information:**

   * **Name**: ``DrugInteractionAnalyzerAgent``
   * **Type**: ``AgenticTool``
   * **Description**: AI agent that analyzes drug-drug interactions and provides clinical recommendations

   **Parameters:**

   * ``compounds`` (string) (required)
     List of compounds to analyze for interactions (comma-separated)

   * ``patient_context`` (string) (optional)
     Patient context (age, comorbidities, medications, etc.)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "DrugInteractionAnalyzerAgent",
          "arguments": {
              "compounds": "example_value"
          }
      }
      result = tu.run(query)


**DrugOptimizationAgent** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI agent that analyzes drug optimization strategies based on ADMET and efficacy data

.. dropdown:: DrugOptimizationAgent tool specification

   **Tool Information:**

   * **Name**: ``DrugOptimizationAgent``
   * **Type**: ``AgenticTool``
   * **Description**: AI agent that analyzes drug optimization strategies based on ADMET and efficacy data

   **Parameters:**

   * ``compounds`` (string) (required)
     List of compounds to optimize (comma-separated)

   * ``admet_data`` (string) (optional)
     ADMET properties and issues

   * ``efficacy_data`` (string) (optional)
     Efficacy and potency data

   * ``target_profile`` (string) (optional)
     Target profile and requirements

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "DrugOptimizationAgent",
          "arguments": {
              "compounds": "example_value"
          }
      }
      result = tu.run(query)


**LiteratureSynthesisAgent** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI agent that synthesizes literature findings and provides evidence-based insights

.. dropdown:: LiteratureSynthesisAgent tool specification

   **Tool Information:**

   * **Name**: ``LiteratureSynthesisAgent``
   * **Type**: ``AgenticTool``
   * **Description**: AI agent that synthesizes literature findings and provides evidence-based insights

   **Parameters:**

   * ``topic`` (string) (required)
     Research topic or question

   * ``literature_data`` (string) (required)
     Literature findings or abstracts to synthesize

   * ``focus_area`` (string) (optional)
     Specific focus area for synthesis

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "LiteratureSynthesisAgent",
          "arguments": {
              "topic": "example_value",
              "literature_data": "example_value"
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
