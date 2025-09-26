Cheminformatics Tools
=====================

**Configuration File**: ``packages/cheminformatics_tools.json``
**Tool Type**: Local
**Tools Count**: 12

This page contains all tools defined in the ``cheminformatics_tools.json`` configuration file.

Available Tools
---------------

**get_chembl_webresource_client_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the chembl-webresource-client package. Python client for ChEMBL web services

.. dropdown:: get_chembl_webresource_client_info tool specification

   **Tool Information:**

   * **Name**: ``get_chembl_webresource_client_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the chembl-webresource-client package. Python client for ChEMBL web services

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_chembl_webresource_client_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_cobra_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about COBRApy – constraint-based metabolic modeling

.. dropdown:: get_cobra_info tool specification

   **Tool Information:**

   * **Name**: ``get_cobra_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about COBRApy – constraint-based metabolic modeling

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_cobra_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_datamol_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the datamol package. Molecular manipulation made easy

.. dropdown:: get_datamol_info tool specification

   **Tool Information:**

   * **Name**: ``get_datamol_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the datamol package. Molecular manipulation made easy

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_datamol_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_deepchem_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about DeepChem – an open-source toolkit that brings advanced AI/ML ...

.. dropdown:: get_deepchem_info tool specification

   **Tool Information:**

   * **Name**: ``get_deepchem_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about DeepChem – an open-source toolkit that brings advanced AI/ML techniques to drug discovery, materials science and quantum chemistry.

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_deepchem_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_dscribe_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about DScribe – a library for generating machine-learning descripto...

.. dropdown:: get_dscribe_info tool specification

   **Tool Information:**

   * **Name**: ``get_dscribe_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about DScribe – a library for generating machine-learning descriptors for materials and molecules.

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_dscribe_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_molfeat_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the molfeat package. Simple and robust molecular featurization

.. dropdown:: get_molfeat_info tool specification

   **Tool Information:**

   * **Name**: ``get_molfeat_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the molfeat package. Simple and robust molecular featurization

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_molfeat_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_molvs_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the molvs package. Molecule validation and standardization

.. dropdown:: get_molvs_info tool specification

   **Tool Information:**

   * **Name**: ``get_molvs_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the molvs package. Molecule validation and standardization

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_molvs_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_mordred_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the mordred package. Molecular descriptor calculator

.. dropdown:: get_mordred_info tool specification

   **Tool Information:**

   * **Name**: ``get_mordred_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the mordred package. Molecular descriptor calculator

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_mordred_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_openbabel_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about OpenBabel – chemical format conversion and analysis

.. dropdown:: get_openbabel_info tool specification

   **Tool Information:**

   * **Name**: ``get_openbabel_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about OpenBabel – chemical format conversion and analysis

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_openbabel_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_openchem_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about OpenChem – deep learning toolkit for drug discovery

.. dropdown:: get_openchem_info tool specification

   **Tool Information:**

   * **Name**: ``get_openchem_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about OpenChem – deep learning toolkit for drug discovery

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about OpenChem

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_openchem_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_pubchempy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the pubchempy package. Python interface for PubChem REST API

.. dropdown:: get_pubchempy_info tool specification

   **Tool Information:**

   * **Name**: ``get_pubchempy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the pubchempy package. Python interface for PubChem REST API

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pubchempy_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_rdkit_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about RDKit – cheminformatics and machine learning toolkit

.. dropdown:: get_rdkit_info tool specification

   **Tool Information:**

   * **Name**: ``get_rdkit_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about RDKit – cheminformatics and machine learning toolkit

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_rdkit_info",
          "arguments": {
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
