Structural Biology Tools
========================

**Configuration File**: ``packages/structural_biology_tools.json``
**Tool Type**: Local
**Tools Count**: 12

This page contains all tools defined in the ``structural_biology_tools.json`` configuration file.

Available Tools
---------------

**get_ase_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about ASE (Atomic Simulation Environment) – a toolkit for building,...

.. dropdown:: get_ase_info tool specification

   **Tool Information:**

   * **Name**: ``get_ase_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about ASE (Atomic Simulation Environment) – a toolkit for building, running and analysing atomistic simulations.

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_ase_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_biopandas_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about BioPandas – pandas-based molecular structure analysis

.. dropdown:: get_biopandas_info tool specification

   **Tool Information:**

   * **Name**: ``get_biopandas_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about BioPandas – pandas-based molecular structure analysis

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_biopandas_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_descriptastorus_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Descriptastorus – molecular descriptor calculation

.. dropdown:: get_descriptastorus_info tool specification

   **Tool Information:**

   * **Name**: ``get_descriptastorus_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Descriptastorus – molecular descriptor calculation

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about Descriptastorus

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_descriptastorus_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_diffdock_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about DiffDock – diffusion model for molecular docking

.. dropdown:: get_diffdock_info tool specification

   **Tool Information:**

   * **Name**: ``get_diffdock_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about DiffDock – diffusion model for molecular docking

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about DiffDock

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_diffdock_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_freesasa_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the freesasa package. Calculate solvent accessible surface areas of proteins

.. dropdown:: get_freesasa_info tool specification

   **Tool Information:**

   * **Name**: ``get_freesasa_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the freesasa package. Calculate solvent accessible surface areas of proteins

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_freesasa_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_htmd_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the htmd package. High throughput molecular dynamics platform

.. dropdown:: get_htmd_info tool specification

   **Tool Information:**

   * **Name**: ``get_htmd_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the htmd package. High throughput molecular dynamics platform

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_htmd_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_mdanalysis_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about MDAnalysis – molecular dynamics trajectory analysis

.. dropdown:: get_mdanalysis_info tool specification

   **Tool Information:**

   * **Name**: ``get_mdanalysis_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about MDAnalysis – molecular dynamics trajectory analysis

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about MDAnalysis

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_mdanalysis_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_mdtraj_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the mdtraj package. Modern library for molecular dynamics trajectory analysis

.. dropdown:: get_mdtraj_info tool specification

   **Tool Information:**

   * **Name**: ``get_mdtraj_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the mdtraj package. Modern library for molecular dynamics trajectory analysis

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_mdtraj_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_nglview_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the nglview package. Jupyter widget for molecular visualization

.. dropdown:: get_nglview_info tool specification

   **Tool Information:**

   * **Name**: ``get_nglview_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the nglview package. Jupyter widget for molecular visualization

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_nglview_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_openmm_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about OpenMM – molecular dynamics simulation toolkit

.. dropdown:: get_openmm_info tool specification

   **Tool Information:**

   * **Name**: ``get_openmm_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about OpenMM – molecular dynamics simulation toolkit

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_openmm_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_pyrosetta_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the pyrosetta package. Python interface to Rosetta macromolecular modeling ...

.. dropdown:: get_pyrosetta_info tool specification

   **Tool Information:**

   * **Name**: ``get_pyrosetta_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the pyrosetta package. Python interface to Rosetta macromolecular modeling suite

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pyrosetta_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_pyscf_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about PySCF – a versatile quantum-chemistry framework in Python.

.. dropdown:: get_pyscf_info tool specification

   **Tool Information:**

   * **Name**: ``get_pyscf_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about PySCF – a versatile quantum-chemistry framework in Python.

   **Parameters:**

   * ``include_examples`` (boolean) (optional)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pyscf_info",
          "arguments": {
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
