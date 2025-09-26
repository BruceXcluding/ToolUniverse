Rcsb Pdb Tools
==============

**Configuration File**: ``rcsb_pdb_tools.json``
**Tool Type**: Local
**Tools Count**: 39

This page contains all tools defined in the ``rcsb_pdb_tools.json`` configuration file.

Available Tools
---------------

**get_assembly_info_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve all associated biological assembly details for a given PDB structure.

.. dropdown:: get_assembly_info_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_assembly_info_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve all associated biological assembly details for a given PDB structure.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character RCSB PDB ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_assembly_info_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_assembly_summary** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get key assembly composition and symmetry summary for an assembly associated with a PDB entry.

.. dropdown:: get_assembly_summary tool specification

   **Tool Information:**

   * **Name**: ``get_assembly_summary``
   * **Type**: ``RCSBTool``
   * **Description**: Get key assembly composition and symmetry summary for an assembly associated with a PDB entry.

   **Parameters:**

   * ``assembly_id`` (string) (optional)
     Assembly ID in format 'PDBID-assemblyNumber' (e.g., '1A8M-1')

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_assembly_summary",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_binding_affinity_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve binding affinity constants (Kd, Ki, IC50) associated with ligands in a PDB entry.

.. dropdown:: get_binding_affinity_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_binding_affinity_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve binding affinity constants (Kd, Ki, IC50) associated with ligands in a PDB entry.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     RCSB PDB ID (e.g., 1A8M)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_binding_affinity_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_chem_comp_audit_info** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetch audit history for a chemical component: action type, date, details, ordinal, and processing...

.. dropdown:: get_chem_comp_audit_info tool specification

   **Tool Information:**

   * **Name**: ``get_chem_comp_audit_info``
   * **Type**: ``RCSBTool``
   * **Description**: Fetch audit history for a chemical component: action type, date, details, ordinal, and processing site.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     Chemical component ID to retrieve audit info for

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_chem_comp_audit_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_chem_comp_charge_and_ambiguity** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve the formal charge and ambiguity flag of a chemical component.

.. dropdown:: get_chem_comp_charge_and_ambiguity tool specification

   **Tool Information:**

   * **Name**: ``get_chem_comp_charge_and_ambiguity``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve the formal charge and ambiguity flag of a chemical component.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     Chemical component ID to query charge and ambiguity

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_chem_comp_charge_and_ambiguity",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_citation_info_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve citation information (authors, journal, year) for a given PDB structure.

.. dropdown:: get_citation_info_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_citation_info_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve citation information (authors, journal, year) for a given PDB structure.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character RCSB PDB ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_citation_info_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_core_refinement_statistics** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve essential refinement statistics for a given PDB structure including R-factors, occupancy...

.. dropdown:: get_core_refinement_statistics tool specification

   **Tool Information:**

   * **Name**: ``get_core_refinement_statistics``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve essential refinement statistics for a given PDB structure including R-factors, occupancy, phase errors, and solvent model parameters.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     PDB entry ID (e.g., '1ABC')

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_core_refinement_statistics",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_crystal_growth_conditions_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get information about the crystallization method and conditions for a structure.

.. dropdown:: get_crystal_growth_conditions_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_crystal_growth_conditions_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Get information about the crystallization method and conditions for a structure.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     PDB ID of the structure

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_crystal_growth_conditions_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_crystallization_ph_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetch the pH used during crystallization of the sample.

.. dropdown:: get_crystallization_ph_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_crystallization_ph_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Fetch the pH used during crystallization of the sample.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     RCSB PDB ID of the structure

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_crystallization_ph_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_crystallographic_properties_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve crystallographic properties such as unit cell dimensions and space group for a PDB entry.

.. dropdown:: get_crystallographic_properties_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_crystallographic_properties_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve crystallographic properties such as unit cell dimensions and space group for a PDB entry.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     PDB ID of the structure

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_crystallographic_properties_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_ec_number_by_entity_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve the Enzyme Commission (EC) number(s) for an entity.

.. dropdown:: get_ec_number_by_entity_id tool specification

   **Tool Information:**

   * **Name**: ``get_ec_number_by_entity_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve the Enzyme Commission (EC) number(s) for an entity.

   **Parameters:**

   * ``entity_id`` (string) (optional)
     Polymer entity ID (e.g., '1A8M_1')

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_ec_number_by_entity_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_em_3d_fitting_and_reconstruction_details** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve EM 3D fitting model details and associated 3D reconstruction info for a given PDB entry.

.. dropdown:: get_em_3d_fitting_and_reconstruction_details tool specification

   **Tool Information:**

   * **Name**: ``get_em_3d_fitting_and_reconstruction_details``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve EM 3D fitting model details and associated 3D reconstruction info for a given PDB entry.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character RCSB PDB ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_em_3d_fitting_and_reconstruction_details",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_emdb_ids_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve Electron Microscopy Data Bank (EMDB) identifiers linked to a PDB entry.

.. dropdown:: get_emdb_ids_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_emdb_ids_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve Electron Microscopy Data Bank (EMDB) identifiers linked to a PDB entry.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character PDB ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_emdb_ids_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_gene_name_by_entity_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve gene name(s) associated with a polymer entity.

.. dropdown:: get_gene_name_by_entity_id tool specification

   **Tool Information:**

   * **Name**: ``get_gene_name_by_entity_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve gene name(s) associated with a polymer entity.

   **Parameters:**

   * ``entity_id`` (string) (optional)
     Entity ID like '1A8M_1'

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_gene_name_by_entity_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_host_organism_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the host organism used for protein expression in a PDB entry.

.. dropdown:: get_host_organism_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_host_organism_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Get the host organism used for protein expression in a PDB entry.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character PDB ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_host_organism_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_ligand_bond_count_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the number of bonds for each ligand in a given PDB structure.

.. dropdown:: get_ligand_bond_count_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_ligand_bond_count_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Get the number of bonds for each ligand in a given PDB structure.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     PDB ID of the entry

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_ligand_bond_count_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_ligand_smiles_by_chem_comp_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve the SMILES chemical structure string for a given chemical component (ligand) ID.

.. dropdown:: get_ligand_smiles_by_chem_comp_id tool specification

   **Tool Information:**

   * **Name**: ``get_ligand_smiles_by_chem_comp_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve the SMILES chemical structure string for a given chemical component (ligand) ID.

   **Parameters:**

   * ``chem_comp_id`` (string) (optional)
     Chemical component ID (e.g., 'ATP')

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_ligand_smiles_by_chem_comp_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_mutation_annotations_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve mutation annotations for a given PDB structure.

.. dropdown:: get_mutation_annotations_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_mutation_annotations_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve mutation annotations for a given PDB structure.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character RCSB PDB ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_mutation_annotations_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_oligosaccharide_descriptors_by_entity_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve structural descriptors for branched entities (e.g., oligosaccharides) in a PDB entry.

.. dropdown:: get_oligosaccharide_descriptors_by_entity_id tool specification

   **Tool Information:**

   * **Name**: ``get_oligosaccharide_descriptors_by_entity_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve structural descriptors for branched entities (e.g., oligosaccharides) in a PDB entry.

   **Parameters:**

   * ``entity_id`` (string) (optional)
     Branched entity ID like '5FMB_2'

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_oligosaccharide_descriptors_by_entity_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_polymer_entity_annotations** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve functional annotations (Pfam domains, GO terms) and associated UniProt accession IDs for...

.. dropdown:: get_polymer_entity_annotations tool specification

   **Tool Information:**

   * **Name**: ``get_polymer_entity_annotations``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve functional annotations (Pfam domains, GO terms) and associated UniProt accession IDs for a polymer entity.

   **Parameters:**

   * ``entity_id`` (string) (optional)
     Polymer entity ID like '1A8M_1'

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_polymer_entity_annotations",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_polymer_entity_count_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the number of distinct polymer entities (chains) in a structure.

.. dropdown:: get_polymer_entity_count_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_polymer_entity_count_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Get the number of distinct polymer entities (chains) in a structure.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character PDB ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_polymer_entity_count_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_polymer_entity_ids_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List polymer entity IDs for a given PDB ID. Useful for building further queries on individual pol...

.. dropdown:: get_polymer_entity_ids_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_polymer_entity_ids_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: List polymer entity IDs for a given PDB ID. Useful for building further queries on individual polymer entities.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character RCSB PDB ID of the protein

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_polymer_entity_ids_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_polymer_entity_type_by_entity_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the polymer entity type (e.g., Protein, DNA) using the polymer entity ID.

.. dropdown:: get_polymer_entity_type_by_entity_id tool specification

   **Tool Information:**

   * **Name**: ``get_polymer_entity_type_by_entity_id``
   * **Type**: ``RCSBTool``
   * **Description**: Get the polymer entity type (e.g., Protein, DNA) using the polymer entity ID.

   **Parameters:**

   * ``entity_id`` (string) (optional)
     Polymer entity ID like '1A8M_1'

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_polymer_entity_type_by_entity_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_polymer_molecular_weight_by_entity_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve the molecular weight of a polymer entity.

.. dropdown:: get_polymer_molecular_weight_by_entity_id tool specification

   **Tool Information:**

   * **Name**: ``get_polymer_molecular_weight_by_entity_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve the molecular weight of a polymer entity.

   **Parameters:**

   * ``entity_id`` (string) (optional)
     Polymer entity ID like '1A8M_1'

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_polymer_molecular_weight_by_entity_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_protein_classification_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the classification of the protein structure (e.g., transferase, oxidoreductase).

.. dropdown:: get_protein_classification_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_protein_classification_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Get the classification of the protein structure (e.g., transferase, oxidoreductase).

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     PDB ID of the entry

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_protein_classification_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_protein_metadata_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve basic protein structure metadata, including structure title, experimental method, resolu...

.. dropdown:: get_protein_metadata_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_protein_metadata_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve basic protein structure metadata, including structure title, experimental method, resolution, and initial release date.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character RCSB PDB ID of the protein

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_protein_metadata_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_refinement_resolution_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve the reported resolution from refinement data for X-ray structures.

.. dropdown:: get_refinement_resolution_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_refinement_resolution_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve the reported resolution from refinement data for X-ray structures.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     PDB entry ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_refinement_resolution_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_release_deposit_dates_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the release and deposition dates for a PDB entry.

.. dropdown:: get_release_deposit_dates_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_release_deposit_dates_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Get the release and deposition dates for a PDB entry.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character RCSB PDB ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_release_deposit_dates_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_sequence_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve amino acid or nucleotide sequence of polymer entities for a given PDB structure.

.. dropdown:: get_sequence_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_sequence_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve amino acid or nucleotide sequence of polymer entities for a given PDB structure.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character RCSB PDB ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_sequence_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_sequence_lengths_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve the sequence lengths of polymer entities for a given PDB structure.

.. dropdown:: get_sequence_lengths_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_sequence_lengths_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve the sequence lengths of polymer entities for a given PDB structure.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character RCSB PDB ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_sequence_lengths_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_sequence_positional_features_by_instance_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve sequence positional features (e.g., binding sites, motifs) for a polymer entity instance.

.. dropdown:: get_sequence_positional_features_by_instance_id tool specification

   **Tool Information:**

   * **Name**: ``get_sequence_positional_features_by_instance_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve sequence positional features (e.g., binding sites, motifs) for a polymer entity instance.

   **Parameters:**

   * ``instance_id`` (string) (optional)
     Polymer entity instance ID like '1NDO.A'

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_sequence_positional_features_by_instance_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_source_organism_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve the scientific name of the source organism for a given PDB structure.

.. dropdown:: get_source_organism_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_source_organism_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve the scientific name of the source organism for a given PDB structure.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character RCSB PDB ID of the structure

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_source_organism_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_space_group_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the crystallographic space group of the structure.

.. dropdown:: get_space_group_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_space_group_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Get the crystallographic space group of the structure.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character RCSB PDB ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_space_group_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_structure_determination_software_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve names of software used during structure determination.

.. dropdown:: get_structure_determination_software_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_structure_determination_software_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve names of software used during structure determination.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     RCSB PDB entry ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_structure_determination_software_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_structure_title_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve the structure title for a given PDB entry.

.. dropdown:: get_structure_title_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_structure_title_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve the structure title for a given PDB entry.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character PDB ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_structure_title_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_structure_validation_metrics_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve structure validation metrics such as R-free, R-work, and clashscore for a PDB entry.

.. dropdown:: get_structure_validation_metrics_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_structure_validation_metrics_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve structure validation metrics such as R-free, R-work, and clashscore for a PDB entry.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     PDB ID of the structure

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_structure_validation_metrics_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_target_cofactor_info** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve essential cofactor information for a given target including cofactor IDs, mechanism of a...

.. dropdown:: get_target_cofactor_info tool specification

   **Tool Information:**

   * **Name**: ``get_target_cofactor_info``
   * **Type**: ``RCSBTool``
   * **Description**: Retrieve essential cofactor information for a given target including cofactor IDs, mechanism of action, literature references, and resource metadata.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     Target ID or entity identifier (e.g., UniProt ID or internal target id)

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_target_cofactor_info",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_taxonomy_by_pdb_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the scientific name and taxonomy of the organism(s) associated with a PDB entry.

.. dropdown:: get_taxonomy_by_pdb_id tool specification

   **Tool Information:**

   * **Name**: ``get_taxonomy_by_pdb_id``
   * **Type**: ``RCSBTool``
   * **Description**: Get the scientific name and taxonomy of the organism(s) associated with a PDB entry.

   **Parameters:**

   * ``pdb_id`` (string) (optional)
     4-character RCSB PDB ID

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_taxonomy_by_pdb_id",
          "arguments": {
          }
      }
      result = tu.run(query)


**get_uniprot_accession_by_entity_id** (Type: RCSBTool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetch UniProt accession numbers associated with a specific polymer entity.

.. dropdown:: get_uniprot_accession_by_entity_id tool specification

   **Tool Information:**

   * **Name**: ``get_uniprot_accession_by_entity_id``
   * **Type**: ``RCSBTool``
   * **Description**: Fetch UniProt accession numbers associated with a specific polymer entity.

   **Parameters:**

   * ``entity_id`` (string) (optional)
     Polymer entity ID (e.g., '1A8M_1')

   **Example Usage:**

   .. code-block:: python

      query = {
          "name": "get_uniprot_accession_by_entity_id",
          "arguments": {
          }
      }
      result = tu.run(query)


Navigation
----------

* :doc:`tools_config_index` - Back to Tools Overview
* :doc:`../guide/loading_tools` - Loading Local Tools
