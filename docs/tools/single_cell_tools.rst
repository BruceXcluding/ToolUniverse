Single Cell Tools
=================

**Configuration File**: ``packages/single_cell_tools.json``
**Tool Type**: Local
**Tools Count**: 14

This page contains all tools defined in the ``single_cell_tools.json`` configuration file.

Available Tools
---------------

**get_anndata_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about AnnData – annotated data for computational biology

.. dropdown:: get_anndata_info tool specification

   **Tool Information:**

   * **Name**: ``get_anndata_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about AnnData – annotated data for computational biology

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_anndata_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_cellrank_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the cellrank package. Trajectory inference and cell fate mapping in single-...

.. dropdown:: get_cellrank_info tool specification

   **Tool Information:**

   * **Name**: ``get_cellrank_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the cellrank package. Trajectory inference and cell fate mapping in single-cell data

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_cellrank_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_episcanpy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the episcanpy package. Epigenomics single cell analysis in Python

.. dropdown:: get_episcanpy_info tool specification

   **Tool Information:**

   * **Name**: ``get_episcanpy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the episcanpy package. Epigenomics single cell analysis in Python

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_episcanpy_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_mudata_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about MuData – multimodal annotated data for computational biology

.. dropdown:: get_mudata_info tool specification

   **Tool Information:**

   * **Name**: ``get_mudata_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about MuData – multimodal annotated data for computational biology

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_mudata_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_palantir_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the palantir package. Algorithm for modeling continuous cell state transitions

.. dropdown:: get_palantir_info tool specification

   **Tool Information:**

   * **Name**: ``get_palantir_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the palantir package. Algorithm for modeling continuous cell state transitions

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_palantir_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_pyscenic_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about pySCENIC – single-cell regulatory network inference

.. dropdown:: get_pyscenic_info tool specification

   **Tool Information:**

   * **Name**: ``get_pyscenic_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about pySCENIC – single-cell regulatory network inference

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about pySCENIC

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pyscenic_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_scanorama_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the scanorama package. Batch correction and integration of single-cell data

.. dropdown:: get_scanorama_info tool specification

   **Tool Information:**

   * **Name**: ``get_scanorama_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the scanorama package. Batch correction and integration of single-cell data

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_scanorama_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_scanpy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Scanpy – scalable single-cell analysis in Python

.. dropdown:: get_scanpy_info tool specification

   **Tool Information:**

   * **Name**: ``get_scanpy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Scanpy – scalable single-cell analysis in Python

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_scanpy_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_scrublet_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Scrublet – single-cell doublet detection

.. dropdown:: get_scrublet_info tool specification

   **Tool Information:**

   * **Name**: ``get_scrublet_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Scrublet – single-cell doublet detection

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_scrublet_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_scvelo_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about scVelo – RNA velocity analysis in single cells

.. dropdown:: get_scvelo_info tool specification

   **Tool Information:**

   * **Name**: ``get_scvelo_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about scVelo – RNA velocity analysis in single cells

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_scvelo_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_scvi_tools_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the scvi-tools package. Deep probabilistic analysis of single-cell omics data

.. dropdown:: get_scvi_tools_info tool specification

   **Tool Information:**

   * **Name**: ``get_scvi_tools_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the scvi-tools package. Deep probabilistic analysis of single-cell omics data

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_scvi_tools_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_souporcell_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about souporcell – scRNA-seq genotype clustering

.. dropdown:: get_souporcell_info tool specification

   **Tool Information:**

   * **Name**: ``get_souporcell_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about souporcell – scRNA-seq genotype clustering

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about souporcell

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_souporcell_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_tiledbsoma_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about TileDB-SOMA – single-cell data storage with TileDB

.. dropdown:: get_tiledbsoma_info tool specification

   **Tool Information:**

   * **Name**: ``get_tiledbsoma_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about TileDB-SOMA – single-cell data storage with TileDB

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about TileDB-SOMA

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_tiledbsoma_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_velocyto_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the velocyto package. RNA velocity analysis for single cell RNA-seq data

.. dropdown:: get_velocyto_info tool specification

   **Tool Information:**

   * **Name**: ``get_velocyto_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the velocyto package. RNA velocity analysis for single cell RNA-seq data

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_velocyto_info",
          "arguments": {
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
