Expert Feedback Tools
=====================

**Configuration File**: ``remote_tools/expert_feedback_tools.json``
**Tool Type**: Remote
**Tools Count**: 6

This page contains all tools defined in the ``expert_feedback_tools.json`` configuration file.

Available Tools
---------------

**consult_human_expert** (Type: RemoteTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consult a human expert for complex scientific questions requiring human judgment. This tool submi...

.. dropdown:: consult_human_expert tool specification

   **Tool Information:**

   * **Name**: ``consult_human_expert``
   * **Type**: ``RemoteTool``
   * **Description**: Consult a human expert for complex scientific questions requiring human judgment. This tool submits questions to human experts who can provide clinical decision support, analysis validation, treatment recommendations, and specialized opinions.

   **Parameters:**

   * ``question`` (string) (required)
     The scientific question or case requiring expert consultation

   * ``specialty`` (string) (required)
     Area of expertise needed (e.g., 'cardiology', 'oncology', 'pharmacology', 'neurology', 'emergency', 'general')

   * ``priority`` (string) (required)
     Request priority level

   * ``context`` (string) (required)
     Additional context or background information about the case

   * ``timeout_minutes`` (integer) (required)
     How long to wait for expert response (default: 5 minutes)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "consult_human_expert",
          "arguments": {
              "question": "example_value",
              "specialty": "example_value",
              "priority": "example_value",
              "context": "example_value",
              "timeout_minutes": 10
          }
      }
      result = tu.run(query)


**get_expert_response** (Type: RemoteTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check if an expert response is available for a previous consultation request. Use this to retriev...

.. dropdown:: get_expert_response tool specification

   **Tool Information:**

   * **Name**: ``get_expert_response``
   * **Type**: ``RemoteTool``
   * **Description**: Check if an expert response is available for a previous consultation request. Use this to retrieve responses when the initial consultation is complete.

   **Parameters:**

   * ``request_id`` (string) (required)
     The ID of the expert consultation request to check

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_expert_response",
          "arguments": {
              "request_id": "example_value"
          }
      }
      result = tu.run(query)


**get_expert_status** (Type: RemoteTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get current status of the human expert system including availability, pending requests, and syste...

.. dropdown:: get_expert_status tool specification

   **Tool Information:**

   * **Name**: ``get_expert_status``
   * **Type**: ``RemoteTool``
   * **Description**: Get current status of the human expert system including availability, pending requests, and system statistics.

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_expert_status",
          "arguments": {
          }
      }
      result = tu.run(query)


**list_pending_expert_requests** (Type: RemoteTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List all pending expert consultation requests (for expert use to see waiting questions).

.. dropdown:: list_pending_expert_requests tool specification

   **Tool Information:**

   * **Name**: ``list_pending_expert_requests``
   * **Type**: ``RemoteTool``
   * **Description**: List all pending expert consultation requests (for expert use to see waiting questions).

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "list_pending_expert_requests",
          "arguments": {
          }
      }
      result = tu.run(query)


**mcp_auto_loader_human_expert** (Type: RemoteTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatically discover and load all tools from Human Expert MCP Server. Can register discovered t...

.. dropdown:: mcp_auto_loader_human_expert tool specification

   **Tool Information:**

   * **Name**: ``mcp_auto_loader_human_expert``
   * **Type**: ``RemoteTool``
   * **Description**: Automatically discover and load all tools from Human Expert MCP Server. Can register discovered tools as individual ToolUniverse tools for expert consultation in complex scientific decisions requiring human judgment.

   **Parameters:**

   No parameters required.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "mcp_auto_loader_human_expert",
          "arguments": {
          }
      }
      result = tu.run(query)


**submit_expert_response** (Type: RemoteTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Submit expert response to a consultation request (for use by human experts through the expert int...

.. dropdown:: submit_expert_response tool specification

   **Tool Information:**

   * **Name**: ``submit_expert_response``
   * **Type**: ``RemoteTool``
   * **Description**: Submit expert response to a consultation request (for use by human experts through the expert interface).

   **Parameters:**

   * ``request_id`` (string) (required)
     The ID of the consultation request to respond to

   * ``response`` (string) (required)
     The expert's response and recommendations

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "submit_expert_response",
          "arguments": {
              "request_id": "example_value",
              "response": "example_value"
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`remote_tools` - Remote Tools Setup
