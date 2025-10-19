Idmap Tools
===========

**Configuration File**: ``idmap_tools.json``
**Tool Type**: Local
**Tools Count**: 3

This page contains all tools defined in the ``idmap_tools.json`` configuration file.

Available Tools
---------------

**OpenTargets_get_disease_ids_by_efoId** (Type: OpenTarget)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given an EFO ID, retrieve all cross-referenced external disease IDs including MONDO, OMIM, MeSH, ...

.. dropdown:: OpenTargets_get_disease_ids_by_efoId tool specification

   **Tool Information:**

   * **Name**: ``OpenTargets_get_disease_ids_by_efoId``
   * **Type**: ``OpenTarget``
   * **Description**: Given an EFO ID, retrieve all cross-referenced external disease IDs including MONDO, OMIM, MeSH, MedDRA, NCIt, ICD10, Orphanet, UMLS.

   **Parameters:**

   * ``efoId`` (string) (required)
     The EFO ID of the disease or phenotype.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "OpenTargets_get_disease_ids_by_efoId",
          "arguments": {
              "efoId": "example_value"
          }
      }
      result = tu.run(query)


**OpenTargets_get_disease_ids_by_name** (Type: OpenTarget)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a disease or phenotype name, find all cross-referenced external IDs (e.g., OMIM, MONDO, MeS...

.. dropdown:: OpenTargets_get_disease_ids_by_name tool specification

   **Tool Information:**

   * **Name**: ``OpenTargets_get_disease_ids_by_name``
   * **Type**: ``OpenTarget``
   * **Description**: Given a disease or phenotype name, find all cross-referenced external IDs (e.g., OMIM, MONDO, MeSH, ICD10, UMLS, MedDRA, NCIt, Orphanet) using Open Targets GraphQL API.

   **Parameters:**

   * ``name`` (string) (required)
     The name of the disease or phenotype (e.g. 'rheumatoid arthritis').

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "OpenTargets_get_disease_ids_by_name",
          "arguments": {
              "name": "example_value"
          }
      }
      result = tu.run(query)


**OpenTargets_map_any_disease_id_to_all_other_ids** (Type: OpenTarget)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given any known disease or phenotype ID (EFO, OMIM, MONDO, UMLS, ICD10, MedDRA, etc.), return all...

.. dropdown:: OpenTargets_map_any_disease_id_to_all_other_ids tool specification

   **Tool Information:**

   * **Name**: ``OpenTargets_map_any_disease_id_to_all_other_ids``
   * **Type**: ``OpenTarget``
   * **Description**: Given any known disease or phenotype ID (EFO, OMIM, MONDO, UMLS, ICD10, MedDRA, etc.), return all known cross-referenced IDs including the EFO ID.

   **Parameters:**

   * ``inputId`` (string) (required)
     Any known disease ID (e.g. OMIM:604302, UMLS:C0003873, ICD10:M05, EFO_0000685, etc.)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "OpenTargets_map_any_disease_id_to_all_other_ids",
          "arguments": {
              "inputId": "example_value"
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
