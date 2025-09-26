Uspto Tools
===========

**Configuration File**: ``uspto_tools.json``
**Tool Type**: Local
**Tools Count**: 6

This page contains all tools defined in the ``uspto_tools.json`` configuration file.

Available Tools
---------------

**get_associated_documents_metadata** (Type: USPTOOpenDataPortalTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Obtains metadata for documents associated with an application, such as publications and grants.

.. dropdown:: get_associated_documents_metadata tool specification

   **Tool Information:**

   * **Name**: ``get_associated_documents_metadata``
   * **Type**: ``USPTOOpenDataPortalTool``
   * **Description**: Obtains metadata for documents associated with an application, such as publications and grants.

   **Parameters:**

   * ``applicationNumberText`` (string) (optional)
     The application number of the patent.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_associated_documents_metadata",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_patent_application_metadata** (Type: USPTOOpenDataPortalTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieves only the metadata for a specific patent application by its application number.

.. dropdown:: get_patent_application_metadata tool specification

   **Tool Information:**

   * **Name**: ``get_patent_application_metadata``
   * **Type**: ``USPTOOpenDataPortalTool``
   * **Description**: Retrieves only the metadata for a specific patent application by its application number.

   **Parameters:**

   * ``applicationNumberText`` (string) (optional)
     The application number of the patent.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_patent_application_metadata",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_patent_continuity_data** (Type: USPTOOpenDataPortalTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetches the parent and child continuity data for a patent application, showing its relationship t...

.. dropdown:: get_patent_continuity_data tool specification

   **Tool Information:**

   * **Name**: ``get_patent_continuity_data``
   * **Type**: ``USPTOOpenDataPortalTool``
   * **Description**: Fetches the parent and child continuity data for a patent application, showing its relationship to other applications.

   **Parameters:**

   * ``applicationNumberText`` (string) (optional)
     The application number of the patent.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_patent_continuity_data",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_patent_foreign_priority_data** (Type: USPTOOpenDataPortalTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accesses information about any foreign priority claims associated with an application.

.. dropdown:: get_patent_foreign_priority_data tool specification

   **Tool Information:**

   * **Name**: ``get_patent_foreign_priority_data``
   * **Type**: ``USPTOOpenDataPortalTool``
   * **Description**: Accesses information about any foreign priority claims associated with an application.

   **Parameters:**

   * ``applicationNumberText`` (string) (optional)
     The application number of the patent.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_patent_foreign_priority_data",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_patent_overview_by_text_query** (Type: USPTOOpenDataPortalTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search for patent application overviews using a query string of the format 'applicationMetaData.i...

.. dropdown:: get_patent_overview_by_text_query tool specification

   **Tool Information:**

   * **Name**: ``get_patent_overview_by_text_query``
   * **Type**: ``USPTOOpenDataPortalTool``
   * **Description**: Search for patent application overviews using a query string of the format 'applicationMetaData.inventionTitle:your_search_term', where your_search_term is the case-insensitive keyword or keyphrase you are searching for. If your_search_term is a multi-word phrase, it must be encased in escaped double quotation marks for exact matching. This tool allows for sorting, offsetting, and filterin of results. Returns a list of important metadata fields for each application, including application number, filing date, grant date, invention title, and CPC classifications.

   **Parameters:**

   * ``query`` (string) (optional)
     Keyword or keyphrase to search for in the patent application title. This field is required.

   * ``exact_match`` (boolean) (optional)
     If true, the search will only return results that exactly match the provided query. Default is false.

   * ``sort`` (string) (optional)
     Sorts results by one of the following fields: filingDate or grantDate. Follow the field name with a space and 'asc' or 'desc' for ascending or descending by date, respectively. For example: 'filingDate desc'

   * ``offset`` (integer) (optional)
     The starting position (zero-indexed) of the result set. Default is 0.

   * ``limit`` (integer) (optional)
     The maximum number of results to return. Default is 25.

   * ``rangeFilters`` (string) (optional)
     Limits results to the date range specified for one of the following fields: filingDate or grantDate. Provide the field name, a space, and a colon-separated start and end value in YYYY-MM-DD format. For example: 'grantDate 2010-01-01:2011-01-01'

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_patent_overview_by_text_query",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_patent_term_adjustment_data** (Type: USPTOOpenDataPortalTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Obtains the patent term adjustment details for a given application number.

.. dropdown:: get_patent_term_adjustment_data tool specification

   **Tool Information:**

   * **Name**: ``get_patent_term_adjustment_data``
   * **Type**: ``USPTOOpenDataPortalTool``
   * **Description**: Obtains the patent term adjustment details for a given application number.

   **Parameters:**

   * ``applicationNumberText`` (string) (optional)
     The application number of the patent.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_patent_term_adjustment_data",
          "arguments": {
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
