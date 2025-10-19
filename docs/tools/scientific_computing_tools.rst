Scientific Computing Tools
==========================

**Configuration File**: ``packages/scientific_computing_tools.json``
**Tool Type**: Local
**Tools Count**: 16

This page contains all tools defined in the ``scientific_computing_tools.json`` configuration file.

Available Tools
---------------

**get_cooler_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Cooler – sparse Hi-C contact matrix storage

.. dropdown:: get_cooler_info tool specification

   **Tool Information:**

   * **Name**: ``get_cooler_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Cooler – sparse Hi-C contact matrix storage

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_cooler_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_cupy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the cupy package. NumPy-compatible array library accelerated with CUDA

.. dropdown:: get_cupy_info tool specification

   **Tool Information:**

   * **Name**: ``get_cupy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the cupy package. NumPy-compatible array library accelerated with CUDA

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_cupy_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_dask_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the dask package. Parallel computing with task scheduling

.. dropdown:: get_dask_info tool specification

   **Tool Information:**

   * **Name**: ``get_dask_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the dask package. Parallel computing with task scheduling

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_dask_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_flowutils_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about FlowUtils – flow cytometry utilities and algorithms

.. dropdown:: get_flowutils_info tool specification

   **Tool Information:**

   * **Name**: ``get_flowutils_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about FlowUtils – flow cytometry utilities and algorithms

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about FlowUtils

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_flowutils_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_h5py_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about h5py – HDF5 for Python

.. dropdown:: get_h5py_info tool specification

   **Tool Information:**

   * **Name**: ``get_h5py_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about h5py – HDF5 for Python

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_h5py_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_joblib_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the joblib package. Lightweight pipelining with Python functions

.. dropdown:: get_joblib_info tool specification

   **Tool Information:**

   * **Name**: ``get_joblib_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the joblib package. Lightweight pipelining with Python functions

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_joblib_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_numpy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about NumPy - the fundamental package for scientific computing with...

.. dropdown:: get_numpy_info tool specification

   **Tool Information:**

   * **Name**: ``get_numpy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about NumPy - the fundamental package for scientific computing with Python

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_numpy_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_optlang_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about optlang – optimization language for mathematical programming

.. dropdown:: get_optlang_info tool specification

   **Tool Information:**

   * **Name**: ``get_optlang_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about optlang – optimization language for mathematical programming

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about optlang

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_optlang_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_pandas_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about pandas - powerful data structures and data analysis tools for...

.. dropdown:: get_pandas_info tool specification

   **Tool Information:**

   * **Name**: ``get_pandas_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about pandas - powerful data structures and data analysis tools for Python

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pandas_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_patsy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the patsy package. Python library for describing statistical models

.. dropdown:: get_patsy_info tool specification

   **Tool Information:**

   * **Name**: ``get_patsy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the patsy package. Python library for describing statistical models

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_patsy_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_scipy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about SciPy – fundamental algorithms for scientific computing

.. dropdown:: get_scipy_info tool specification

   **Tool Information:**

   * **Name**: ``get_scipy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about SciPy – fundamental algorithms for scientific computing

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about SciPy

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_scipy_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_sympy_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about SymPy – symbolic mathematics library

.. dropdown:: get_sympy_info tool specification

   **Tool Information:**

   * **Name**: ``get_sympy_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about SymPy – symbolic mathematics library

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about SymPy

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_sympy_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_tiledb_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about TileDB – modern database for array data

.. dropdown:: get_tiledb_info tool specification

   **Tool Information:**

   * **Name**: ``get_tiledb_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about TileDB – modern database for array data

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about TileDB

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_tiledb_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_tqdm_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about tqdm – fast progress bars for Python

.. dropdown:: get_tqdm_info tool specification

   **Tool Information:**

   * **Name**: ``get_tqdm_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about tqdm – fast progress bars for Python

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_tqdm_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_xarray_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the xarray package. N-D labeled arrays and datasets in Python

.. dropdown:: get_xarray_info tool specification

   **Tool Information:**

   * **Name**: ``get_xarray_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the xarray package. N-D labeled arrays and datasets in Python

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_xarray_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_zarr_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the zarr package. Chunked, compressed, N-dimensional arrays

.. dropdown:: get_zarr_info tool specification

   **Tool Information:**

   * **Name**: ``get_zarr_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the zarr package. Chunked, compressed, N-dimensional arrays

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_zarr_info",
          "arguments": {
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
