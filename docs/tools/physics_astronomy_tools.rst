Physics Astronomy Tools
=======================

**Configuration File**: ``packages/physics_astronomy_tools.json``
**Tool Type**: Local
**Tools Count**: 5

This page contains all tools defined in the ``physics_astronomy_tools.json`` configuration file.

Available Tools
---------------

**get_astropy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the astropy package. Astronomy and astrophysics library

.. dropdown:: get_astropy_info tool specification

   **Tool Information:**

   * **Name**: ``get_astropy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the astropy package. Astronomy and astrophysics library

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_astropy_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_galpy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the galpy package. Galactic dynamics library

.. dropdown:: get_galpy_info tool specification

   **Tool Information:**

   * **Name**: ``get_galpy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the galpy package. Galactic dynamics library

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_galpy_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_pyephem_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the pyephem package. Astronomical computations for Python

.. dropdown:: get_pyephem_info tool specification

   **Tool Information:**

   * **Name**: ``get_pyephem_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the pyephem package. Astronomical computations for Python

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pyephem_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_qutip_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the qutip package. Quantum toolbox in Python

.. dropdown:: get_qutip_info tool specification

   **Tool Information:**

   * **Name**: ``get_qutip_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the qutip package. Quantum toolbox in Python

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_qutip_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_sunpy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the sunpy package. Solar data analysis library

.. dropdown:: get_sunpy_info tool specification

   **Tool Information:**

   * **Name**: ``get_sunpy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the sunpy package. Solar data analysis library

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_sunpy_info",
          "arguments": {
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
