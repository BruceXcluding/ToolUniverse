#!/usr/bin/env python3
"""
Automatically generate tool reference documentation from JSON configuration files.

This script scans all tool configuration files in the ToolUniverse data directory
and generates comprehensive RST documentation.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict


def load_tool_configs(data_dir):
    """Load all tool configurations from JSON files."""
    tool_configs = []
    config_files = []
    
    # Find all JSON files in data directory
    for json_file in Path(data_dir).rglob("*.json"):
        if json_file.name.startswith('.') or 'hook_config' in json_file.name:
            continue
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle both list and dict formats
            if isinstance(data, list):
                for tool in data:
                    if isinstance(tool, dict) and 'name' in tool:
                        tool['source_file'] = str(json_file.relative_to(data_dir))
                        tool_configs.append(tool)
            elif isinstance(data, dict):
                # Some files might have nested structure
                for key, value in data.items():
                    if isinstance(value, list):
                        for tool in value:
                            if isinstance(tool, dict) and 'name' in tool:
                                tool['source_file'] = str(json_file.relative_to(data_dir))
                                tool_configs.append(tool)
                                
            config_files.append(str(json_file.relative_to(data_dir)))
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue
    
    return tool_configs, config_files


def categorize_tools(tools):
    """Categorize tools by domain and type."""
    categories = defaultdict(list)
    
    # Define category mappings based on tool names, types, and labels
    category_mappings = {
        'Molecular & Genetic Data': ['uniprot', 'gene_ontology', 'enrichr', 'alphafold'],
        'Disease & Target Data': ['opentarget', 'efo', 'monarch', 'disease_target'],
        'Drug & Chemical Data': ['pubchem', 'chembl', 'admetai', 'dailymed'],
        'Drug Safety & Regulatory': ['fda_drug', 'adverse_event', 'openfda'],
        'Clinical Research': ['clinicaltrials', 'clinical'],
        'Literature & Publications': ['pubtator', 'europe_pmc', 'semantic_scholar', 'openalex'],
        'Specialized Databases': ['hpa', 'reactome', 'humanbase', 'medlineplus', 'gwas'],
        'AI & ML Models': ['agentic', 'boltz', 'machine_learning', 'txagent'],
        'Data Integration': ['embedding', 'finder', 'compose', 'special'],
        'Software Packages': ['packages/', 'bioinformatics', 'visualization', 'computing'],
        'External Services': ['uspto', 'remote_tools/']
    }
    
    for tool in tools:
        tool_name = tool.get('name', '').lower()
        tool_type = tool.get('type', '').lower()
        source_file = tool.get('source_file', '').lower()
        labels = [label.lower() for label in tool.get('label', [])]
        
        categorized = False
        
        for category, keywords in category_mappings.items():
            if any(keyword in source_file or keyword in tool_name or keyword in tool_type 
                   for keyword in keywords):
                categories[category].append(tool)
                categorized = True
                break
        
        if not categorized:
            categories['Other Tools'].append(tool)
    
    return dict(categories)


def format_parameter_docs(parameter_schema):
    """Format parameter documentation from JSON schema."""
    if not parameter_schema or 'properties' not in parameter_schema:
        return "No parameters required."
    
    docs = []
    properties = parameter_schema.get('properties')
    if not properties:
        return "No parameters required."
        
    required = parameter_schema.get('required', [])
    
    for param_name, param_info in properties.items():
        param_type = param_info.get('type', 'unknown')
        description = param_info.get('description', 'No description provided')
        is_required = param_name in required
        
        req_marker = " (required)" if is_required else " (optional)"
        docs.append(f"* ``{param_name}`` ({param_type}){req_marker} - {description}")
    
    return "\n".join(docs)


def generate_tool_section(tool):
    """Generate RST documentation for a single tool."""
    name = tool.get('name', 'Unknown Tool')
    description = tool.get('description', 'No description provided')
    parameter_schema = tool.get('parameter', {})
    tool_type = tool.get('type', 'Unknown')
    
    # Format the tool name as a subsection
    section = f"\n**{name}**\n"
    section += "~" * (len(name) + 4) + "\n\n"
    
    # Add description
    section += f"{description}\n\n"
    
    # Add parameters
    section += "**Parameters:**\n\n"
    param_docs = format_parameter_docs(parameter_schema)
    section += param_docs + "\n\n"
    
    # Add tool type
    section += f"**Type:** {tool_type}\n\n"
    
    # Add example usage
    section += "**Example:**\n\n"
    section += ".. code-block:: python\n\n"
    section += f"   query = {{\n"
    section += f"       \"name\": \"{name}\",\n"
    section += f"       \"arguments\": {{\n"
    
    # Generate example parameters
    if parameter_schema and 'properties' in parameter_schema:
        properties = parameter_schema.get('properties')
        if properties:
            required = parameter_schema.get('required', [])
            examples = []
            
            for param_name, param_info in properties.items():
                if param_name in required:
                    param_type = param_info.get('type', 'string')
                    if param_type == 'string':
                        examples.append(f'           "{param_name}": "example_value"')
                    elif param_type == 'integer':
                        examples.append(f'           "{param_name}": 10')
                    elif param_type == 'boolean':
                        examples.append(f'           "{param_name}": true')
                    elif param_type == 'array':
                        examples.append(f'           "{param_name}": ["item1", "item2"]')
                    else:
                        examples.append(f'           "{param_name}": "example_value"')
            
            if examples:
                section += ",\n".join(examples) + "\n"
    
    section += "       }\n"
    section += "   }\n"
    section += "   result = tu.run(query)\n\n"
    
    return section


def generate_rst_content(categories, config_files):
    """Generate complete RST content."""
    
    content = """ToolUniverse API Reference
==========================

**Complete API reference for all ToolUniverse tools with automatic configuration extraction.**

This comprehensive reference is automatically generated from tool configuration files, ensuring accuracy and completeness.

Total Tools Available
---------------------

"""
    
    total_tools = sum(len(tools) for tools in categories.values())
    content += f"ToolUniverse provides **{total_tools} tools** across {len(categories)} major categories.\n\n"
    
    # Add overview table
    content += ".. list-table:: Tool Categories Overview\n"
    content += "   :header-rows: 1\n"
    content += "   :widths: 30 15 55\n\n"
    content += "   * - Category\n"
    content += "     - Count\n"
    content += "     - Description\n"
    
    category_descriptions = {
        'Molecular & Genetic Data': 'Protein, gene, and molecular information databases',
        'Disease & Target Data': 'Disease-target associations and biomedical ontologies',
        'Drug & Chemical Data': 'Chemical compounds, bioactivity, and drug information',
        'Drug Safety & Regulatory': 'FDA databases, adverse events, and safety data',
        'Clinical Research': 'Clinical trials and research study information',
        'Literature & Publications': 'Scientific literature and publication databases',
        'Specialized Databases': 'Domain-specific research databases',
        'AI & ML Models': 'Machine learning models and AI-powered analysis',
        'Data Integration': 'Tool discovery, composition, and data integration',
        'Software Packages': 'Scientific computing and analysis packages',
        'External Services': 'External APIs and specialized services'
    }
    
    for category, tools in sorted(categories.items()):
        desc = category_descriptions.get(category, 'Specialized tools and services')
        content += f"   * - {category}\n"
        content += f"     - {len(tools)}\n"
        content += f"     - {desc}\n"
    
    content += "\n\n"
    
    # Add detailed sections for each category
    for category, tools in sorted(categories.items()):
        if not tools:
            continue
            
        content += f"{category}\n"
        content += "-" * len(category) + "\n\n"
        
        # Sort tools by name
        sorted_tools = sorted(tools, key=lambda x: x.get('name', ''))
        
        for tool in sorted_tools:
            content += generate_tool_section(tool)
    
    # Add configuration files reference
    content += "\nConfiguration Files\n"
    content += "-------------------\n\n"
    content += "This documentation is automatically generated from the following configuration files:\n\n"
    
    for config_file in sorted(config_files):
        content += f"* ``{config_file}``\n"
    
    content += "\n"
    content += ".. note::\n"
    content += "   This reference is automatically generated from tool configuration files.\n"
    content += "   For the most up-to-date information, regenerate using ``python docs/generate_tool_reference.py``\n\n"
    
    return content


def main():
    """Main function to generate tool reference documentation."""
    # Get the script directory and find the data directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "src" / "tooluniverse" / "data"
    
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        sys.exit(1)
    
    print(f"Loading tool configurations from {data_dir}")
    
    # Load all tool configurations
    tools, config_files = load_tool_configs(data_dir)
    print(f"Loaded {len(tools)} tools from {len(config_files)} configuration files")
    
    # Categorize tools
    categories = categorize_tools(tools)
    
    # Generate RST content
    rst_content = generate_rst_content(categories, config_files)
    
    # Write to API directory
    api_dir = script_dir / "api"
    api_dir.mkdir(exist_ok=True)
    
    output_file = api_dir / "tools_complete_reference.rst"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(rst_content)
    
    print(f"Generated tool reference documentation: {output_file}")
    print(f"Total categories: {len(categories)}")
    print(f"Total tools documented: {sum(len(tools) for tools in categories.values())}")
    
    # Print category breakdown
    print("\nCategory breakdown:")
    for category, tools in sorted(categories.items()):
        print(f"  {category}: {len(tools)} tools")


if __name__ == "__main__":
    main()
