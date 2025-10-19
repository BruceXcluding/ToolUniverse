Visualization Tools
===================

**Configuration File**: ``packages/visualization_tools.json``
**Tool Type**: Local
**Tools Count**: 13

This page contains all tools defined in the ``visualization_tools.json`` configuration file.

Available Tools
---------------

**get_altair_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the altair package. Declarative statistical visualization library

.. dropdown:: get_altair_info tool specification

   **Tool Information:**

   * **Name**: ``get_altair_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the altair package. Declarative statistical visualization library

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_altair_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_bokeh_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the bokeh package. Interactive visualization library for modern web browsers

.. dropdown:: get_bokeh_info tool specification

   **Tool Information:**

   * **Name**: ``get_bokeh_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the bokeh package. Interactive visualization library for modern web browsers

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_bokeh_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_cellpose_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Cellpose – cell segmentation algorithm

.. dropdown:: get_cellpose_info tool specification

   **Tool Information:**

   * **Name**: ``get_cellpose_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Cellpose – cell segmentation algorithm

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_cellpose_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_datashader_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the datashader package. Graphics pipeline system for creating meaningful vi...

.. dropdown:: get_datashader_info tool specification

   **Tool Information:**

   * **Name**: ``get_datashader_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the datashader package. Graphics pipeline system for creating meaningful visualizations of large datasets

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_datashader_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_holoviews_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the holoviews package. Declarative data visualization in Python

.. dropdown:: get_holoviews_info tool specification

   **Tool Information:**

   * **Name**: ``get_holoviews_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the holoviews package. Declarative data visualization in Python

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_holoviews_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_igraph_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about igraph – network analysis and visualization

.. dropdown:: get_igraph_info tool specification

   **Tool Information:**

   * **Name**: ``get_igraph_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about igraph – network analysis and visualization

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_igraph_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_matplotlib_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Matplotlib – comprehensive library for creating visualization...

.. dropdown:: get_matplotlib_info tool specification

   **Tool Information:**

   * **Name**: ``get_matplotlib_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Matplotlib – comprehensive library for creating visualizations in Python

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_matplotlib_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_opencv_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about OpenCV-Python – computer vision library

.. dropdown:: get_opencv_info tool specification

   **Tool Information:**

   * **Name**: ``get_opencv_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about OpenCV-Python – computer vision library

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_opencv_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


**get_plantcv_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about PlantCV – plant phenotyping with image analysis

.. dropdown:: get_plantcv_info tool specification

   **Tool Information:**

   * **Name**: ``get_plantcv_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about PlantCV – plant phenotyping with image analysis

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about PlantCV

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_plantcv_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_plotly_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the plotly package. Interactive plotting library for Python

.. dropdown:: get_plotly_info tool specification

   **Tool Information:**

   * **Name**: ``get_plotly_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the plotly package. Interactive plotting library for Python

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_plotly_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_pyvis_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the pyvis package. Python library for visualizing networks

.. dropdown:: get_pyvis_info tool specification

   **Tool Information:**

   * **Name**: ``get_pyvis_info``
   * **Type**: ``PackageTool``
   * **Description**: Get information about the pyvis package. Python library for visualizing networks

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_pyvis_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_scikit_image_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about scikit-image – image processing in Python

.. dropdown:: get_scikit_image_info tool specification

   **Tool Information:**

   * **Name**: ``get_scikit_image_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about scikit-image – image processing in Python

   **Parameters:**

   * ``info_type`` (string) (required)
     Type of information to retrieve about scikit-image

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_scikit_image_info",
          "arguments": {
              "info_type": "example_value"
          }
      }
      result = tu.run(query)


**get_seaborn_info** (Type: PackageTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get comprehensive information about Seaborn – statistical data visualization

.. dropdown:: get_seaborn_info tool specification

   **Tool Information:**

   * **Name**: ``get_seaborn_info``
   * **Type**: ``PackageTool``
   * **Description**: Get comprehensive information about Seaborn – statistical data visualization

   **Parameters:**

   * ``include_examples`` (boolean) (required)
     Whether to include usage examples and quick start guide

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_seaborn_info",
          "arguments": {
              "include_examples": true
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
