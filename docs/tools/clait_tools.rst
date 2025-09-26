Clait Tools
===========

**Configuration File**: ``clait_tools.json``
**Tool Type**: Local
**Tools Count**: 3

This page contains all tools defined in the ``clait_tools.json`` configuration file.

Available Tools
---------------

**AdverseEventICDMapper** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extracts adverse events from narrative clinical or pharmacovigilance text and maps each event to ...

.. dropdown:: AdverseEventICDMapper tool specification

   **Tool Information:**

   * **Name**: ``AdverseEventICDMapper``
   * **Type**: ``AgenticTool``
   * **Description**: Extracts adverse events from narrative clinical or pharmacovigilance text and maps each event to the most specific ICD-10-CM code.

   **Parameters:**

   * ``source_text`` (string) (required)
     Unstructured narrative text that may mention adverse events.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "AdverseEventICDMapper",
          "arguments": {
              "source_text": "example_value"
          }
      }
      result = tu.run(query)


**AdverseEventPredictionQuestionGenerator** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates a set of personalized adverse‐event prediction questions for a given disease and drug, ...

.. dropdown:: AdverseEventPredictionQuestionGenerator tool specification

   **Tool Information:**

   * **Name**: ``AdverseEventPredictionQuestionGenerator``
   * **Type**: ``AgenticTool``
   * **Description**: Generates a set of personalized adverse‐event prediction questions for a given disease and drug, across multiple patient subgroups.

   **Parameters:**

   * ``disease_name`` (string) (required)
     The name of the disease or condition

   * ``drug_name`` (string) (required)
     The name of the drug

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "AdverseEventPredictionQuestionGenerator",
          "arguments": {
              "disease_name": "example_value",
              "drug_name": "example_value"
          }
      }
      result = tu.run(query)


**AdverseEventPredictionQuestionGeneratorWithContext** (Type: AgenticTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates a set of personalized adverse‐event prediction questions for a given disease and drug, ...

.. dropdown:: AdverseEventPredictionQuestionGeneratorWithContext tool specification

   **Tool Information:**

   * **Name**: ``AdverseEventPredictionQuestionGeneratorWithContext``
   * **Type**: ``AgenticTool``
   * **Description**: Generates a set of personalized adverse‐event prediction questions for a given disease and drug, incorporating additional context information such as patient medical history, clinical findings, or research data.

   **Parameters:**

   * ``disease_name`` (string) (required)
     The name of the disease or condition

   * ``drug_name`` (string) (required)
     The name of the drug

   * ``context_information`` (string) (required)
     Additional context information such as patient medical history, clinical findings, research data, or other relevant background information that should inform the adverse event prediction questions

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "AdverseEventPredictionQuestionGeneratorWithContext",
          "arguments": {
              "disease_name": "example_value",
              "drug_name": "example_value",
              "context_information": "example_value"
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
