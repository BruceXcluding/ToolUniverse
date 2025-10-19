Depmap Tools
============

**Configuration File**: ``remote_tools/depmap_tools.json``
**Tool Type**: Remote
**Tools Count**: 1

This page contains all tools defined in the ``depmap_tools.json`` configuration file.

Available Tools
---------------

**compute_depmap24q2_gene_correlations** (Type: RemoteTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyzes gene-gene correlations from DepMap CRISPR knockout screening data. This tool validates g...

.. dropdown:: compute_depmap24q2_gene_correlations tool specification

   **Tool Information:**

   * **Name**: ``compute_depmap24q2_gene_correlations``
   * **Type**: ``RemoteTool``
   * **Description**: Analyzes gene-gene correlations from DepMap CRISPR knockout screening data. This tool validates genetic interactions using empirical cell viability data from 1,320+ cancer cell lines in the DepMap 24Q2 dataset. It determines if two genes have correlated knockout effects, providing insights into genetic dependencies and synthetic lethal relationships.

   **Parameters:**

   * ``gene_a`` (string) (required)
     First gene symbol for correlation analysis (e.g., 'BRAF', 'TP53'). Must use standard HUGO gene nomenclature.

   * ``gene_b`` (string) (required)
     Second gene symbol for correlation analysis (e.g., 'MAPK1', 'MDM2'). Must use standard HUGO gene nomenclature.

   * ``data_dir`` (string) (required)
     Path to directory containing DepMap correlation matrices. If None, uses DEPMAP_DATA_PATH/depmap_24q2.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "compute_depmap24q2_gene_correlations",
          "arguments": {
              "gene_a": "example_value",
              "gene_b": "example_value",
              "data_dir": "example_value"
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`remote_tools` - Remote Tools Setup
