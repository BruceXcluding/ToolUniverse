Disease Target Score Tools
==========================

**Configuration File**: ``disease_target_score_tools.json``
**Tool Type**: Local
**Tools Count**: 10

This page contains all tools defined in the ``disease_target_score_tools.json`` configuration file.

Available Tools
---------------

**cancer_biomarkers_disease_target_score** (Type: DiseaseTargetScoreTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract disease-target association scores from cancer biomarkers data. This includes known cancer...

.. dropdown:: cancer_biomarkers_disease_target_score tool specification

   **Tool Information:**

   * **Name**: ``cancer_biomarkers_disease_target_score``
   * **Type**: ``DiseaseTargetScoreTool``
   * **Description**: Extract disease-target association scores from cancer biomarkers data. This includes known cancer biomarkers.

   **Parameters:**

   * ``efoId`` (string) (required)
     The EFO (Experimental Factor Ontology) ID of the disease, e.g., 'EFO_0000339' for chronic myelogenous leukemia

   * ``pageSize`` (integer) (optional)
     Number of results per page (default: 100, max: 100)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "cancer_biomarkers_disease_target_score",
          "arguments": {
              "efoId": "example_value"
          }
      }
      result = tu.run(query)


**cancer_gene_census_disease_target_score** (Type: DiseaseTargetScoreTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract disease-target association scores from Cancer Gene Census. This provides curated cancer g...

.. dropdown:: cancer_gene_census_disease_target_score tool specification

   **Tool Information:**

   * **Name**: ``cancer_gene_census_disease_target_score``
   * **Type**: ``DiseaseTargetScoreTool``
   * **Description**: Extract disease-target association scores from Cancer Gene Census. This provides curated cancer gene data.

   **Parameters:**

   * ``efoId`` (string) (required)
     The EFO (Experimental Factor Ontology) ID of the disease, e.g., 'EFO_0000339' for chronic myelogenous leukemia

   * ``pageSize`` (integer) (optional)
     Number of results per page (default: 100, max: 100)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "cancer_gene_census_disease_target_score",
          "arguments": {
              "efoId": "example_value"
          }
      }
      result = tu.run(query)


**chembl_disease_target_score** (Type: DiseaseTargetScoreTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract disease-target association scores specifically from ChEMBL database. ChEMBL provides bioa...

.. dropdown:: chembl_disease_target_score tool specification

   **Tool Information:**

   * **Name**: ``chembl_disease_target_score``
   * **Type**: ``DiseaseTargetScoreTool``
   * **Description**: Extract disease-target association scores specifically from ChEMBL database. ChEMBL provides bioactivity data for drug-target interactions.

   **Parameters:**

   * ``efoId`` (string) (required)
     The EFO (Experimental Factor Ontology) ID of the disease, e.g., 'EFO_0000339' for chronic myelogenous leukemia

   * ``pageSize`` (integer) (optional)
     Number of results per page (default: 100, max: 100)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "chembl_disease_target_score",
          "arguments": {
              "efoId": "example_value"
          }
      }
      result = tu.run(query)


**disease_target_score** (Type: DiseaseTargetScoreTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract disease-target association scores from a specific data source using GraphQL API. This too...

.. dropdown:: disease_target_score tool specification

   **Tool Information:**

   * **Name**: ``disease_target_score``
   * **Type**: ``DiseaseTargetScoreTool``
   * **Description**: Extract disease-target association scores from a specific data source using GraphQL API. This tool retrieves all targets associated with a disease and their scores from a specified datasource (e.g., chembl, eva, cancer_gene_census, etc.).

   **Parameters:**

   * ``efoId`` (string) (required)
     The EFO (Experimental Factor Ontology) ID of the disease, e.g., 'EFO_0000339' for chronic myelogenous leukemia

   * ``datasourceId`` (string) (required)
     The datasource ID to extract scores from. Available options: 'chembl', 'eva', 'eva_somatic', 'cancer_gene_census', 'cancer_biomarkers', 'europepmc', 'expression_atlas', 'genomics_england', 'impc', 'reactome', 'uniprot_literature', 'uniprot_variants'

   * ``pageSize`` (integer) (optional)
     Number of results per page (default: 100, max: 100)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "disease_target_score",
          "arguments": {
              "efoId": "example_value",
              "datasourceId": "example_value"
          }
      }
      result = tu.run(query)


**europepmc_disease_target_score** (Type: DiseaseTargetScoreTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract disease-target association scores from Europe PMC literature. This includes literature-ba...

.. dropdown:: europepmc_disease_target_score tool specification

   **Tool Information:**

   * **Name**: ``europepmc_disease_target_score``
   * **Type**: ``DiseaseTargetScoreTool``
   * **Description**: Extract disease-target association scores from Europe PMC literature. This includes literature-based evidence.

   **Parameters:**

   * ``efoId`` (string) (required)
     The EFO (Experimental Factor Ontology) ID of the disease, e.g., 'EFO_0000339' for chronic myelogenous leukemia

   * ``pageSize`` (integer) (optional)
     Number of results per page (default: 100, max: 100)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "europepmc_disease_target_score",
          "arguments": {
              "efoId": "example_value"
          }
      }
      result = tu.run(query)


**eva_disease_target_score** (Type: DiseaseTargetScoreTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract disease-target association scores from EVA (European Variation Archive). EVA provides gen...

.. dropdown:: eva_disease_target_score tool specification

   **Tool Information:**

   * **Name**: ``eva_disease_target_score``
   * **Type**: ``DiseaseTargetScoreTool``
   * **Description**: Extract disease-target association scores from EVA (European Variation Archive). EVA provides genetic variant data.

   **Parameters:**

   * ``efoId`` (string) (required)
     The EFO (Experimental Factor Ontology) ID of the disease, e.g., 'EFO_0000339' for chronic myelogenous leukemia

   * ``pageSize`` (integer) (optional)
     Number of results per page (default: 100, max: 100)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "eva_disease_target_score",
          "arguments": {
              "efoId": "example_value"
          }
      }
      result = tu.run(query)


**eva_somatic_disease_target_score** (Type: DiseaseTargetScoreTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract disease-target association scores from EVA somatic mutations. This includes somatic varia...

.. dropdown:: eva_somatic_disease_target_score tool specification

   **Tool Information:**

   * **Name**: ``eva_somatic_disease_target_score``
   * **Type**: ``DiseaseTargetScoreTool``
   * **Description**: Extract disease-target association scores from EVA somatic mutations. This includes somatic variant data.

   **Parameters:**

   * ``efoId`` (string) (required)
     The EFO (Experimental Factor Ontology) ID of the disease, e.g., 'EFO_0000339' for chronic myelogenous leukemia

   * ``pageSize`` (integer) (optional)
     Number of results per page (default: 100, max: 100)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "eva_somatic_disease_target_score",
          "arguments": {
              "efoId": "example_value"
          }
      }
      result = tu.run(query)


**expression_atlas_disease_target_score** (Type: DiseaseTargetScoreTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract disease-target association scores from Expression Atlas. This provides gene expression data.

.. dropdown:: expression_atlas_disease_target_score tool specification

   **Tool Information:**

   * **Name**: ``expression_atlas_disease_target_score``
   * **Type**: ``DiseaseTargetScoreTool``
   * **Description**: Extract disease-target association scores from Expression Atlas. This provides gene expression data.

   **Parameters:**

   * ``efoId`` (string) (required)
     The EFO (Experimental Factor Ontology) ID of the disease, e.g., 'EFO_0000339' for chronic myelogenous leukemia

   * ``pageSize`` (integer) (optional)
     Number of results per page (default: 100, max: 100)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "expression_atlas_disease_target_score",
          "arguments": {
              "efoId": "example_value"
          }
      }
      result = tu.run(query)


**genomics_england_disease_target_score** (Type: DiseaseTargetScoreTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract disease-target association scores from Genomics England data. This includes clinical geno...

.. dropdown:: genomics_england_disease_target_score tool specification

   **Tool Information:**

   * **Name**: ``genomics_england_disease_target_score``
   * **Type**: ``DiseaseTargetScoreTool``
   * **Description**: Extract disease-target association scores from Genomics England data. This includes clinical genomics evidence.

   **Parameters:**

   * ``efoId`` (string) (required)
     The EFO (Experimental Factor Ontology) ID of the disease, e.g., 'EFO_0000339' for chronic myelogenous leukemia

   * ``pageSize`` (integer) (optional)
     Number of results per page (default: 100, max: 100)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "genomics_england_disease_target_score",
          "arguments": {
              "efoId": "example_value"
          }
      }
      result = tu.run(query)


**reactome_disease_target_score** (Type: DiseaseTargetScoreTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract disease-target association scores from Reactome pathway data. This includes pathway-based...

.. dropdown:: reactome_disease_target_score tool specification

   **Tool Information:**

   * **Name**: ``reactome_disease_target_score``
   * **Type**: ``DiseaseTargetScoreTool``
   * **Description**: Extract disease-target association scores from Reactome pathway data. This includes pathway-based evidence.

   **Parameters:**

   * ``efoId`` (string) (required)
     The EFO (Experimental Factor Ontology) ID of the disease, e.g., 'EFO_0000339' for chronic myelogenous leukemia

   * ``pageSize`` (integer) (optional)
     Number of results per page (default: 100, max: 100)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "reactome_disease_target_score",
          "arguments": {
              "efoId": "example_value"
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
