Odphp Tools
===========

**Configuration File**: ``odphp_tools.json``
**Tool Type**: Local
**Tools Count**: 4

This page contains all tools defined in the ``odphp_tools.json`` configuration file.

Available Tools
---------------

**odphp_itemlist** (Type: ODPHPItemList)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tools browses and returns available topics and categories and it is helpful to help narrow a...

.. dropdown:: odphp_itemlist tool specification

   **Tool Information:**

   * **Name**: ``odphp_itemlist``
   * **Type**: ``ODPHPItemList``
   * **Description**: This tools browses and returns available topics and categories and it is helpful to help narrow a broad request (e.g., “show me all topics”). For full topic content, `odphp_topicsearch` tool is helpful.

   **Parameters:**

   * ``lang`` (string) (required)
     Language code (en or es)

   * ``type`` (string) (required)
     topic or category

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "odphp_itemlist",
          "arguments": {
              "lang": "example_value",
              "type": "example_value"
          }
      }
      result = tu.run(query)


**odphp_myhealthfinder** (Type: ODPHPMyHealthfinder)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tool provides personalized preventive-care recommendations and it is helpful for different a...

.. dropdown:: odphp_myhealthfinder tool specification

   **Tool Information:**

   * **Name**: ``odphp_myhealthfinder``
   * **Type**: ``ODPHPMyHealthfinder``
   * **Description**: This tool provides personalized preventive-care recommendations and it is helpful for different ages, sexes, pregnancy status, gives age/sex/pregnancy. It retrieves metadata, plain-language sections, and dataset links to the full article (AccessibleVersion links). If the user wants the full text of a recommendation, the `odphp_outlink_fetch` tool is helpful.

   **Parameters:**

   * ``lang`` (string) (required)
     Language code (en or es)

   * ``age`` (integer) (required)
     Age in years (0–120)

   * ``sex`` (string) (required)
     Male or Female

   * ``pregnant`` (string) (required)
     "Yes" or "No"

   * ``strip_html`` (boolean) (required)
     If true, also return PlainSections[] with HTML removed for each topic

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "odphp_myhealthfinder",
          "arguments": {
              "lang": "example_value",
              "age": 10,
              "sex": "example_value",
              "pregnant": "example_value",
              "strip_html": true
          }
      }
      result = tu.run(query)


**odphp_outlink_fetch** (Type: ODPHPOutlinkFetch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tool retrieves readable text from ODPHP article links and information sources. This is helpf...

.. dropdown:: odphp_outlink_fetch tool specification

   **Tool Information:**

   * **Name**: ``odphp_outlink_fetch``
   * **Type**: ``ODPHPOutlinkFetch``
   * **Description**: This tool retrieves readable text from ODPHP article links and information sources. This is helpful after using the `odphp_myhealthfinder` or `odphp_topicsearch` tools or when the user wants to simply dive deeper into ODPHP data.

   **Parameters:**

   * ``urls`` (array) (required)
     1–3 absolute URLs from AccessibleVersion or RelatedItems.Url

   * ``max_chars`` (integer) (required)
     Optional hard cap on extracted text length (e.g., 5000)

   * ``return_html`` (boolean) (required)
     If true, also return minimally cleaned HTML

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "odphp_outlink_fetch",
          "arguments": {
              "urls": ["item1", "item2"],
              "max_chars": 10,
              "return_html": true
          }
      }
      result = tu.run(query)


**odphp_topicsearch** (Type: ODPHPTopicSearch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find specific health topics and get their full content. Use when the user mentions a keyword (e.g...

.. dropdown:: odphp_topicsearch tool specification

   **Tool Information:**

   * **Name**: ``odphp_topicsearch``
   * **Type**: ``ODPHPTopicSearch``
   * **Description**: Find specific health topics and get their full content. Use when the user mentions a keyword (e.g., “folic acid”, “blood pressure”) or when you already have topic/category IDs from `odphp_itemlist`. Returns detailed topic pages (Title, Sections, RelatedItems) and an AccessibleVersion link. Next: to quote or summarize the actual page text, pass the AccessibleVersion (or RelatedItems URLs) to `odphp_outlink_fetch`.

   **Parameters:**

   * ``lang`` (string) (required)
     Language code (en or es)

   * ``topicId`` (string) (required)
     Comma-separated topic IDs

   * ``categoryId`` (string) (required)
     Comma-separated category IDs

   * ``keyword`` (string) (required)
     Keyword search for topics

   * ``strip_html`` (boolean) (required)
     If true, also return PlainSections[] with HTML removed for each topic

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "odphp_topicsearch",
          "arguments": {
              "lang": "example_value",
              "topicId": "example_value",
              "categoryId": "example_value",
              "keyword": "example_value",
              "strip_html": true
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
