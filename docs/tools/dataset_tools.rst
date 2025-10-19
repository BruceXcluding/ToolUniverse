Dataset Tools
=============

**Configuration File**: ``dataset_tools.json``
**Tool Type**: Local
**Tools Count**: 7

This page contains all tools defined in the ``dataset_tools.json`` configuration file.

Available Tools
---------------

**dict_search** (Type: DatasetTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search the DICTrank dataset for drug-induced cardiotoxicity (DICT) risk information by trade name...

.. dropdown:: dict_search tool specification

   **Tool Information:**

   * **Name**: ``dict_search``
   * **Type**: ``DatasetTool``
   * **Description**: Search the DICTrank dataset for drug-induced cardiotoxicity (DICT) risk information by trade name, generic name, or active ingredient. Searching with exact match is not recommeded.

   **Parameters:**

   * ``query`` (string) (required)
     Free-text query (e.g. 'ZYPREXA', 'Olanzapine').

   * ``search_fields`` (array) (required)
     Columns to search. Choose from: 'Trade Name', 'Generic/Proper Name(s)', 'Active Ingredient(s)'.

   * ``case_sensitive`` (boolean) (required)
     Match text with exact case if true.

   * ``exact_match`` (boolean) (required)
     Field value must equal query exactly if true; otherwise substring match.

   * ``limit`` (integer) (required)
     Maximum number of rows to return.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "dict_search",
          "arguments": {
              "query": "example_value",
              "search_fields": ["item1", "item2"],
              "case_sensitive": true,
              "exact_match": true,
              "limit": 10
          }
      }
      result = tu.run(query)


**dili_search** (Type: DatasetTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search the DILIrank dataset for drug-induced liver-injury (DILI) risk information by compound nam...

.. dropdown:: dili_search tool specification

   **Tool Information:**

   * **Name**: ``dili_search``
   * **Type**: ``DatasetTool``
   * **Description**: Search the DILIrank dataset for drug-induced liver-injury (DILI) risk information by compound name. Searching with exact match is not recommeded.

   **Parameters:**

   * ``query`` (string) (required)
     Free-text query (e.g. 'acetaminophen').

   * ``search_fields`` (array) (required)
     Columns to search. Choose from: 'Compound Name'.

   * ``case_sensitive`` (boolean) (required)
     Match text with exact case if true.

   * ``exact_match`` (boolean) (required)
     Field value must equal query exactly if true; otherwise substring match.

   * ``limit`` (integer) (required)
     Maximum number of rows to return.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "dili_search",
          "arguments": {
              "query": "example_value",
              "search_fields": ["item1", "item2"],
              "case_sensitive": true,
              "exact_match": true,
              "limit": 10
          }
      }
      result = tu.run(query)


**diqt_search** (Type: DatasetTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search the DIQTA dataset for drug-induced QT-interval prolongation risk information by generic na...

.. dropdown:: diqt_search tool specification

   **Tool Information:**

   * **Name**: ``diqt_search``
   * **Type**: ``DatasetTool``
   * **Description**: Search the DIQTA dataset for drug-induced QT-interval prolongation risk information by generic name or DrugBank ID. Searching with exact match is not recommeded for generic name.

   **Parameters:**

   * ``query`` (string) (required)
     Free-text query (e.g. 'Astemizole', 'DB00637').

   * ``search_fields`` (array) (required)
     Columns to search. Choose from: 'Generic/Proper Name(s)', 'DrugBank ID'.

   * ``case_sensitive`` (boolean) (required)
     Match text with exact case if true.

   * ``exact_match`` (boolean) (required)
     Field value must equal query exactly if true; otherwise substring match.

   * ``limit`` (integer) (required)
     Maximum number of rows to return.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "diqt_search",
          "arguments": {
              "query": "example_value",
              "search_fields": ["item1", "item2"],
              "case_sensitive": true,
              "exact_match": true,
              "limit": 10
          }
      }
      result = tu.run(query)


**drugbank_full_search** (Type: DatasetTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search the cleaned DrugBank dataframe (one row per drug) by ID, common name, or synonym. Returns ...

.. dropdown:: drugbank_full_search tool specification

   **Tool Information:**

   * **Name**: ``drugbank_full_search``
   * **Type**: ``DatasetTool``
   * **Description**: Search the cleaned DrugBank dataframe (one row per drug) by ID, common name, or synonym. Returns identifiers, ATC, main pharmacology text fields, and protein partners. For best results, it is recommended that one uses `drugbank_vocab_search` to obtain DrugBank ID from other keywords first, and use this tool with DrugBank ID.

   **Parameters:**

   * ``query`` (string) (required)
     Free-text query (e.g. 'DB00945', 'acetylsalicylic', 'Acarbosa').

   * ``search_fields`` (array) (required)
     Columns to search in. Choose from: 'drugbank_id', 'name', 'synonyms'.

   * ``case_sensitive`` (boolean) (required)
     Match text with exact case if true.

   * ``exact_match`` (boolean) (required)
     Field value must equal query exactly if true; otherwise substring match.

   * ``limit`` (integer) (required)
     Max number of rows to return.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "drugbank_full_search",
          "arguments": {
              "query": "example_value",
              "search_fields": ["item1", "item2"],
              "case_sensitive": true,
              "exact_match": true,
              "limit": 10
          }
      }
      result = tu.run(query)


**drugbank_links_search** (Type: DatasetTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search the cross-reference table linking DrugBank IDs to external identifiers (CAS, KEGG, PubChem...

.. dropdown:: drugbank_links_search tool specification

   **Tool Information:**

   * **Name**: ``drugbank_links_search``
   * **Type**: ``DatasetTool``
   * **Description**: Search the cross-reference table linking DrugBank IDs to external identifiers (CAS, KEGG, PubChem, ChEBI, PharmGKB, UniProt, etc.) and web resources.

   **Parameters:**

   * ``query`` (string) (required)
     Free-text query (e.g. 'DB00002', 'Cetuximab').

   * ``search_fields`` (array) (required)
     Columns to search. Choose from: 'DrugBank ID', 'Name', 'CAS Number', 'Drug Type', 'KEGG Compound ID', 'KEGG Drug ID', 'PubChem Compound ID', 'PubChem Substance ID', 'ChEBI ID', 'PharmGKB ID', 'HET ID', 'UniProt ID', 'Wikipedia ID', 'Drugs.com Link', 'NDC ID', 'ChemSpider ID', 'BindingDB ID', 'TTD ID'.

   * ``case_sensitive`` (boolean) (required)
     Match text with exact case if true.

   * ``exact_match`` (boolean) (required)
     Field value must equal query exactly if true; otherwise substring match.

   * ``limit`` (integer) (required)
     Maximum number of rows to return.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "drugbank_links_search",
          "arguments": {
              "query": "example_value",
              "search_fields": ["item1", "item2"],
              "case_sensitive": true,
              "exact_match": true,
              "limit": 10
          }
      }
      result = tu.run(query)


**drugbank_vocab_filter** (Type: DatasetTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Filter the DrugBank vocabulary dataset based on specific field criteria. Use simple field-value p...

.. dropdown:: drugbank_vocab_filter tool specification

   **Tool Information:**

   * **Name**: ``drugbank_vocab_filter``
   * **Type**: ``DatasetTool``
   * **Description**: Filter the DrugBank vocabulary dataset based on specific field criteria. Use simple field-value pairs to filter drugs by properties like names, IDs, and chemical identifiers.

   **Parameters:**

   * ``field`` (string) (required)
     The field to filter on

   * ``condition`` (string) (required)
     The type of filtering condition to apply. Filter is case-insensitive.

   * ``value`` (string) (optional)
     The value to filter by. Not required when condition is 'not_empty'. Examples: 'insulin' (for contains), 'DB00' (for starts_with), 'acid' (for ends_with), 'Aspirin' (for exact)

   * ``limit`` (integer) (required)
     Maximum number of results to return.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "drugbank_vocab_filter",
          "arguments": {
              "field": "example_value",
              "condition": "example_value",
              "limit": 10
          }
      }
      result = tu.run(query)


**drugbank_vocab_search** (Type: DatasetTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search the DrugBank vocabulary dataset for drugs by name, ID, synonyms, or other fields using tex...

.. dropdown:: drugbank_vocab_search tool specification

   **Tool Information:**

   * **Name**: ``drugbank_vocab_search``
   * **Type**: ``DatasetTool``
   * **Description**: Search the DrugBank vocabulary dataset for drugs by name, ID, synonyms, or other fields using text-based queries. Returns detailed drug information including DrugBank ID, common name, CAS number, UNII, and synonyms.

   **Parameters:**

   * ``query`` (string) (required)
     Search query string. Can be drug name, synonym, DrugBank ID, or any text to search for.

   * ``search_fields`` (array) (required)
     Fields to search in. Available fields: 'DrugBank ID', 'Accession Numbers', 'Common name', 'CAS', 'UNII', 'Synonyms', 'Standard InChI Key'.

   * ``case_sensitive`` (boolean) (required)
     Whether the search should be case sensitive.

   * ``exact_match`` (boolean) (required)
     Whether to perform exact matching instead of substring matching.

   * ``limit`` (integer) (required)
     Maximum number of results to return.

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "drugbank_vocab_search",
          "arguments": {
              "query": "example_value",
              "search_fields": ["item1", "item2"],
              "case_sensitive": true,
              "exact_match": true,
              "limit": 10
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
