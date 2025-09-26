Bioinformatics Core Tools
=========================

**Configuration File**: ``packages/bioinformatics_core_tools.json``
**Tool Type**: Local
**Tools Count**: 37

This page contains all tools defined in the ``bioinformatics_core_tools.json`` configuration file.

Available Tools
---------------

**get_arxiv_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about arxiv – access to arXiv preprint repository

.. dropdown:: get_arxiv_info tool specification

   **Tool Information:**

   * **Name**: ``get_arxiv_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about arxiv – access to arXiv preprint repository

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_arxiv_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_biopython_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Biopython – powerful tools for computational molecular biolog...

.. dropdown:: get_biopython_info tool specification

   **Tool Information:**

   * **Name**: ``get_biopython_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Biopython – powerful tools for computational molecular biology and bioinformatics

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_biopython_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_bioservices_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the bioservices package. Python package: bioservices

.. dropdown:: get_bioservices_info tool specification

   **Tool Information:**

   * **Name**: ``get_bioservices_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the bioservices package. Python package: bioservices

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_bioservices_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_biotite_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Biotite – comprehensive computational molecular biology library

.. dropdown:: get_biotite_info tool specification

   **Tool Information:**

   * **Name**: ``get_biotite_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Biotite – comprehensive computational molecular biology library

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_biotite_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_cryosparc_tools_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about cryosparc-tools – interface to CryoSPARC cryo-EM processing

.. dropdown:: get_cryosparc_tools_info tool specification

   **Tool Information:**

   * **Name**: ``get_cryosparc_tools_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about cryosparc-tools – interface to CryoSPARC cryo-EM processing

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about cryosparc-tools

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_cryosparc_tools_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_dendropy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the dendropy package. Python package: dendropy

.. dropdown:: get_dendropy_info tool specification

   **Tool Information:**

   * **Name**: ``get_dendropy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the dendropy package. Python package: dendropy

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_dendropy_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_ete3_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the ete3 package. Python package: ete3

.. dropdown:: get_ete3_info tool specification

   **Tool Information:**

   * **Name**: ``get_ete3_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the ete3 package. Python package: ete3

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_ete3_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_fanc_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about FAN-C – framework for analyzing nuclear contacts

.. dropdown:: get_fanc_info tool specification

   **Tool Information:**

   * **Name**: ``get_fanc_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about FAN-C – framework for analyzing nuclear contacts

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about FAN-C

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_fanc_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_flask_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Flask - a lightweight WSGI web application framework

.. dropdown:: get_flask_info tool specification

   **Tool Information:**

   * **Name**: ``get_flask_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Flask - a lightweight WSGI web application framework

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_flask_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_flowio_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about FlowIO – FCS file I/O for flow cytometry

.. dropdown:: get_flowio_info tool specification

   **Tool Information:**

   * **Name**: ``get_flowio_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about FlowIO – FCS file I/O for flow cytometry

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about FlowIO

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_flowio_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_flowkit_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about FlowKit – flow cytometry analysis toolkit

.. dropdown:: get_flowkit_info tool specification

   **Tool Information:**

   * **Name**: ``get_flowkit_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about FlowKit – flow cytometry analysis toolkit

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about FlowKit

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_flowkit_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_gget_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about gget – genomics command-line tool and Python package

.. dropdown:: get_gget_info tool specification

   **Tool Information:**

   * **Name**: ``get_gget_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about gget – genomics command-line tool and Python package

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_gget_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_googlesearch_python_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about googlesearch-python – Google search automation

.. dropdown:: get_googlesearch_python_info tool specification

   **Tool Information:**

   * **Name**: ``get_googlesearch_python_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about googlesearch-python – Google search automation

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about googlesearch-python

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_googlesearch_python_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_khmer_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about khmer – nucleotide sequence k-mer analysis

.. dropdown:: get_khmer_info tool specification

   **Tool Information:**

   * **Name**: ``get_khmer_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about khmer – nucleotide sequence k-mer analysis

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about khmer

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_khmer_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_lifelines_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about lifelines – survival analysis in Python

.. dropdown:: get_lifelines_info tool specification

   **Tool Information:**

   * **Name**: ``get_lifelines_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about lifelines – survival analysis in Python

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_lifelines_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_loompy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about loompy – efficient storage for large omics datasets

.. dropdown:: get_loompy_info tool specification

   **Tool Information:**

   * **Name**: ``get_loompy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about loompy – efficient storage for large omics datasets

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about loompy

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_loompy_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_mageck_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about MAGeCK – CRISPR screen analysis toolkit

.. dropdown:: get_mageck_info tool specification

   **Tool Information:**

   * **Name**: ``get_mageck_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about MAGeCK – CRISPR screen analysis toolkit

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about MAGeCK

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_mageck_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_msprime_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about msprime – coalescent simulation framework

.. dropdown:: get_msprime_info tool specification

   **Tool Information:**

   * **Name**: ``get_msprime_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about msprime – coalescent simulation framework

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_msprime_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_networkx_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about NetworkX – network analysis library

.. dropdown:: get_networkx_info tool specification

   **Tool Information:**

   * **Name**: ``get_networkx_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about NetworkX – network analysis library

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about NetworkX

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_networkx_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_numba_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Numba – JIT compiler for Python

.. dropdown:: get_numba_info tool specification

   **Tool Information:**

   * **Name**: ``get_numba_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Numba – JIT compiler for Python

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about Numba

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_numba_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_pdbfixer_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about PDBFixer – protein structure preparation tool

.. dropdown:: get_pdbfixer_info tool specification

   **Tool Information:**

   * **Name**: ``get_pdbfixer_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about PDBFixer – protein structure preparation tool

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about PDBFixer

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pdbfixer_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_plip_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about PLIP – protein-ligand interaction profiler

.. dropdown:: get_plip_info tool specification

   **Tool Information:**

   * **Name**: ``get_plip_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about PLIP – protein-ligand interaction profiler

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about PLIP

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_plip_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_poliastro_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about poliastro – astrodynamics library

.. dropdown:: get_poliastro_info tool specification

   **Tool Information:**

   * **Name**: ``get_poliastro_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about poliastro – astrodynamics library

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about poliastro

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_poliastro_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_prody_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about ProDy – protein dynamics analysis

.. dropdown:: get_prody_info tool specification

   **Tool Information:**

   * **Name**: ``get_prody_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about ProDy – protein dynamics analysis

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about ProDy

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_prody_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_pybigwig_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about pyBigWig – BigWig file access in Python

.. dropdown:: get_pybigwig_info tool specification

   **Tool Information:**

   * **Name**: ``get_pybigwig_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about pyBigWig – BigWig file access in Python

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about pyBigWig

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pybigwig_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_pykalman_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about PyKalman – Kalman filtering and smoothing

.. dropdown:: get_pykalman_info tool specification

   **Tool Information:**

   * **Name**: ``get_pykalman_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about PyKalman – Kalman filtering and smoothing

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about PyKalman

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pykalman_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_pymassspec_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about PyMassSpec – mass spectrometry data analysis

.. dropdown:: get_pymassspec_info tool specification

   **Tool Information:**

   * **Name**: ``get_pymassspec_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about PyMassSpec – mass spectrometry data analysis

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about PyMassSpec

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pymassspec_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_pymed_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about PyMed – PubMed access in Python

.. dropdown:: get_pymed_info tool specification

   **Tool Information:**

   * **Name**: ``get_pymed_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about PyMed – PubMed access in Python

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pymed_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_pypdf2_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about PyPDF2 – PDF manipulation library

.. dropdown:: get_pypdf2_info tool specification

   **Tool Information:**

   * **Name**: ``get_pypdf2_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about PyPDF2 – PDF manipulation library

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about PyPDF2

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pypdf2_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_pyscreener_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about PyScreener – high-throughput virtual screening in Python

.. dropdown:: get_pyscreener_info tool specification

   **Tool Information:**

   * **Name**: ``get_pyscreener_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about PyScreener – high-throughput virtual screening in Python

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about PyScreener

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pyscreener_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_pytdc_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about PyTDC – Therapeutics Data Commons in Python

.. dropdown:: get_pytdc_info tool specification

   **Tool Information:**

   * **Name**: ``get_pytdc_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about PyTDC – Therapeutics Data Commons in Python

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about PyTDC

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pytdc_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_requests_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Requests - Python HTTP library for humans

.. dropdown:: get_requests_info tool specification

   **Tool Information:**

   * **Name**: ``get_requests_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Requests - Python HTTP library for humans

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_requests_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_ruptures_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about ruptures – change point detection library

.. dropdown:: get_ruptures_info tool specification

   **Tool Information:**

   * **Name**: ``get_ruptures_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about ruptures – change point detection library

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about ruptures

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_ruptures_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_scholarly_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about scholarly – Google Scholar data retrieval

.. dropdown:: get_scholarly_info tool specification

   **Tool Information:**

   * **Name**: ``get_scholarly_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about scholarly – Google Scholar data retrieval

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about scholarly

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_scholarly_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_scikit_bio_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about scikit-bio – bioinformatics library built on scientific Pytho...

.. dropdown:: get_scikit_bio_info tool specification

   **Tool Information:**

   * **Name**: ``get_scikit_bio_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about scikit-bio – bioinformatics library built on scientific Python stack

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_scikit_bio_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_trackpy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about trackpy – particle tracking toolkit for Python

.. dropdown:: get_trackpy_info tool specification

   **Tool Information:**

   * **Name**: ``get_trackpy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about trackpy – particle tracking toolkit for Python

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about trackpy

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_trackpy_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_tskit_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about tskit – tree sequence toolkit for population genetics

.. dropdown:: get_tskit_info tool specification

   **Tool Information:**

   * **Name**: ``get_tskit_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about tskit – tree sequence toolkit for population genetics

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about tskit

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_tskit_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
