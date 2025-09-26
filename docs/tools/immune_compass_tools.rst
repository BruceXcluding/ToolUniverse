Immune Compass Tools
====================

**Configuration File**: ``remote_tools/immune_compass_tools.json``
**Tool Type**: Remote
**Tools Count**: 1

This page contains all tools defined in the ``immune_compass_tools.json`` configuration file.

Available Tools
---------------

**run_compass_prediction** (Type: RemoteTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Predicts immune checkpoint inhibitor (ICI) response using COMPASS model. This tool analyzes singl...

.. dropdown:: run_compass_prediction tool specification

   **Tool Information:**

   * **Name**: ``run_compass_prediction``
   * **Type**: ``RemoteTool``
   * **Description**: Predicts immune checkpoint inhibitor (ICI) response using COMPASS model. This tool analyzes single-sample tumor gene expression data to predict patient responsiveness to immune checkpoint inhibitor therapy. The COMPASS model leverages immune cell concept analysis to provide both a binary prediction and interpretable insights into the immune microenvironment factors driving the prediction.

   **Parameters:**

   * ``gene_expression_data_path`` (string) (required)
     Path to the TPM expression data file. Keys should be standard gene symbols (e.g., 'CD274', 'PDCD1', 'CTLA4'). Values should be normalized expression in TPM (Transcripts Per Million). Minimum ~100 genes recommended for reliable predictions.

   * ``threshold`` (number) (optional)
     Probability threshold for responder classification (0.0-1.0). Values ≥ threshold classify sample as likely responder. Default 0.5 provides balanced sensitivity/specificity. Consider lower thresholds (~0.3) for higher sensitivity.

   * ``root_path`` (string) (optional)
     Path to the directory containing model checkpoints. If None, uses COMPASS_MODEL_PATH/immune-compass/checkpoint.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "run_compass_prediction",
          "arguments": {
              "gene_expression_data_path": "example_value"
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`remote_tools` - Remote Tools Setup
