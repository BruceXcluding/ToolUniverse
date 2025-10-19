Pinnacle Tools
==============

**Configuration File**: ``remote_tools/pinnacle_tools.json``
**Tool Type**: Remote
**Tools Count**: 1

This page contains all tools defined in the ``pinnacle_tools.json`` configuration file.

Available Tools
---------------

**run_pinnacle_ppi_retrieval** (Type: RemoteTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieves cell-type-specific protein-protein interaction embeddings from PINNACLE. This tool prov...

.. dropdown:: run_pinnacle_ppi_retrieval tool specification

   **Tool Information:**

   * **Name**: ``run_pinnacle_ppi_retrieval``
   * **Type**: ``RemoteTool``
   * **Description**: Retrieves cell-type-specific protein-protein interaction embeddings from PINNACLE. This tool provides access to pre-computed PINNACLE (Protein Interaction Network Contextualized Learning) embeddings that represent protein-protein interactions in specific cellular contexts. These embeddings encode functional relationships between proteins as dense vector representations, capturing both direct physical interactions and functional associations.

   **Parameters:**

   * ``cell_type`` (string) (required)
     Target cell type for embedding retrieval. Supports flexible naming: Standard formats: 'b_cell', 'hepatocyte', 'cardiomyocyte'; Alternative formats: 'B-cell', 'T cell', 'NK cells'; Tissue types: 'liver', 'heart', 'brain', 'immune'. The tool performs intelligent matching to find the best available match.

   * ``embed_path`` (string) (required)
     Path to the PINNACLE embeddings file (.pth format). If None, uses PINNACLE_DATA_PATH/pinnacle_embeds/ppi_embed_dict.pth.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "run_pinnacle_ppi_retrieval",
          "arguments": {
              "cell_type": "example_value",
              "embed_path": "example_value"
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`remote_tools` - Remote Tools Setup
