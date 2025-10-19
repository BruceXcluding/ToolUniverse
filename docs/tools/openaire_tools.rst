Openaire Tools
==============

**Configuration File**: ``openaire_tools.json``
**Tool Type**: Local
**Tools Count**: 1

This page contains all tools defined in the ``openaire_tools.json`` configuration file.

Available Tools
---------------

**OpenAIRE_search_publications** (Type: OpenAIRETool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search OpenAIRE Explore for research products including publications, datasets, and software. Ope...

.. dropdown:: OpenAIRE_search_publications tool specification

   **Tool Information:**

   * **Name**: ``OpenAIRE_search_publications``
   * **Type**: ``OpenAIRETool``
   * **Description**: Search OpenAIRE Explore for research products including publications, datasets, and software. OpenAIRE is the European open science platform that provides access to research outputs from EU-funded projects.

   **Parameters:**

   * ``query`` (string) (required)
     Search query for OpenAIRE research products. Use keywords to search across titles, abstracts, and metadata.

   * ``max_results`` (integer) (required)
     Maximum number of results to return. Default is 10, maximum is 100.

   * ``type`` (string) (required)
     Type of research product to search: 'publications', 'datasets', or 'software'. Default is 'publications'.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "OpenAIRE_search_publications",
          "arguments": {
              "query": "example_value",
              "max_results": 10,
              "type": "example_value"
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
