Genomics Tools
==============

**Configuration File**: ``packages/genomics_tools.json``
**Tool Type**: Local
**Tools Count**: 20

This page contains all tools defined in the ``genomics_tools.json`` configuration file.

Available Tools
---------------

**get_arboreto_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Arboreto – gene regulatory network inference

.. dropdown:: get_arboreto_info tool specification

   **Tool Information:**

   * **Name**: ``get_arboreto_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Arboreto – gene regulatory network inference

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about Arboreto

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_arboreto_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_cellxgene_census_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about cellxgene-census – access to the CELLxGENE Census single-cell...

.. dropdown:: get_cellxgene_census_info tool specification

   **Tool Information:**

   * **Name**: ``get_cellxgene_census_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about cellxgene-census – access to the CELLxGENE Census single-cell data

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about cellxgene-census

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_cellxgene_census_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_clair3_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Clair3 – variant calling for long-read sequencing

.. dropdown:: get_clair3_info tool specification

   **Tool Information:**

   * **Name**: ``get_clair3_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Clair3 – variant calling for long-read sequencing

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about Clair3

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_clair3_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_cyvcf2_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about cyvcf2 – fast VCF/BCF file processing

.. dropdown:: get_cyvcf2_info tool specification

   **Tool Information:**

   * **Name**: ``get_cyvcf2_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about cyvcf2 – fast VCF/BCF file processing

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_cyvcf2_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_deeptools_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about deepTools – deep sequencing data processing

.. dropdown:: get_deeptools_info tool specification

   **Tool Information:**

   * **Name**: ``get_deeptools_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about deepTools – deep sequencing data processing

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about deepTools

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_deeptools_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_gseapy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about GSEApy – Gene Set Enrichment Analysis in Python

.. dropdown:: get_gseapy_info tool specification

   **Tool Information:**

   * **Name**: ``get_gseapy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about GSEApy – Gene Set Enrichment Analysis in Python

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_gseapy_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_jcvi_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about JCVI – genome assembly and comparative genomics

.. dropdown:: get_jcvi_info tool specification

   **Tool Information:**

   * **Name**: ``get_jcvi_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about JCVI – genome assembly and comparative genomics

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about JCVI

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_jcvi_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_kipoiseq_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the kipoiseq package. Kipoi sequence utilities for genomics deep learning

.. dropdown:: get_kipoiseq_info tool specification

   **Tool Information:**

   * **Name**: ``get_kipoiseq_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the kipoiseq package. Kipoi sequence utilities for genomics deep learning

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_kipoiseq_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_poretools_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the poretools package. Python package: poretools

.. dropdown:: get_poretools_info tool specification

   **Tool Information:**

   * **Name**: ``get_poretools_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the poretools package. Python package: poretools

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_poretools_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_pybedtools_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about pybedtools – Python wrapper for BEDTools

.. dropdown:: get_pybedtools_info tool specification

   **Tool Information:**

   * **Name**: ``get_pybedtools_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about pybedtools – Python wrapper for BEDTools

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pybedtools_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_pydeseq2_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about PyDESeq2 – RNA-seq differential expression analysis

.. dropdown:: get_pydeseq2_info tool specification

   **Tool Information:**

   * **Name**: ``get_pydeseq2_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about PyDESeq2 – RNA-seq differential expression analysis

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about PyDESeq2

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pydeseq2_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_pyensembl_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the pyensembl package. Python interface to Ensembl reference genome metadata

.. dropdown:: get_pyensembl_info tool specification

   **Tool Information:**

   * **Name**: ``get_pyensembl_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the pyensembl package. Python interface to Ensembl reference genome metadata

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pyensembl_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_pyfaidx_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about pyfaidx – efficient FASTA file indexing and random access

.. dropdown:: get_pyfaidx_info tool specification

   **Tool Information:**

   * **Name**: ``get_pyfaidx_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about pyfaidx – efficient FASTA file indexing and random access

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pyfaidx_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_pyfasta_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the pyfasta package. Python library for efficient random access to fasta su...

.. dropdown:: get_pyfasta_info tool specification

   **Tool Information:**

   * **Name**: ``get_pyfasta_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the pyfasta package. Python library for efficient random access to fasta subsequences

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pyfasta_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_pyliftover_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about PyLiftover – genomic coordinate conversion between assemblies

.. dropdown:: get_pyliftover_info tool specification

   **Tool Information:**

   * **Name**: ``get_pyliftover_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about PyLiftover – genomic coordinate conversion between assemblies

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pyliftover_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_pyranges_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about PyRanges – efficient genomic interval operations

.. dropdown:: get_pyranges_info tool specification

   **Tool Information:**

   * **Name**: ``get_pyranges_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about PyRanges – efficient genomic interval operations

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pyranges_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_pysam_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about pysam – interface to SAM/BAM/CRAM files

.. dropdown:: get_pysam_info tool specification

   **Tool Information:**

   * **Name**: ``get_pysam_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about pysam – interface to SAM/BAM/CRAM files

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pysam_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_pyvcf_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the pyvcf package. Python library for parsing and manipulating VCF files

.. dropdown:: get_pyvcf_info tool specification

   **Tool Information:**

   * **Name**: ``get_pyvcf_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the pyvcf package. Python library for parsing and manipulating VCF files

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pyvcf_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_reportlab_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about ReportLab – PDF generation library

.. dropdown:: get_reportlab_info tool specification

   **Tool Information:**

   * **Name**: ``get_reportlab_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about ReportLab – PDF generation library

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_reportlab_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_viennarna_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about ViennaRNA – RNA structure prediction and analysis

.. dropdown:: get_viennarna_info tool specification

   **Tool Information:**

   * **Name**: ``get_viennarna_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about ViennaRNA – RNA structure prediction and analysis

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_viennarna_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
