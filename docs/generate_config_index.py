#!/usr/bin/env python3
"""
Generate separate pages for each tool configuration file.

This script creates individual RST pages for each configuration file,
making it easier to navigate and maintain tool documentation.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_tool_configs_by_file(data_dir):
    """Load all tool configurations grouped by source file."""
    config_by_file = defaultdict(list)

    # Find all JSON files in data directory
    for json_file in Path(data_dir).rglob("*.json"):
        if json_file.name.startswith(".") or "hook_config" in json_file.name:
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            relative_path = str(json_file.relative_to(data_dir))

            # Handle both list and dict formats
            if isinstance(data, list):
                for tool in data:
                    if isinstance(tool, dict) and "name" in tool:
                        config_by_file[relative_path].append(tool)
            elif isinstance(data, dict):
                # Some files might have nested structure
                for key, value in data.items():
                    if isinstance(value, list):
                        for tool in value:
                            if isinstance(tool, dict) and "name" in tool:
                                config_by_file[relative_path].append(tool)

        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue

    return dict(config_by_file)


def format_tool_details(tool):
    """Format detailed tool information in collapsible sections."""
    tool_name = tool.get("name", "Unknown")
    tool_desc = tool.get("description", "No description provided")
    tool_type = tool.get("type", "Unknown")
    parameters = tool.get("parameter", {})
    
    # Create collapsible section using dropdown directive
    details = f".. dropdown:: {tool_name} tool specification\n\n"
    
    # Tool Information
    details += f"   **Tool Information:**\n\n"
    details += f"   * **Name**: ``{tool_name}``\n"
    details += f"   * **Type**: ``{tool_type}``\n"
    details += f"   * **Description**: {tool_desc}\n\n"
    
    # Parameters section
    if parameters and 'properties' in parameters:
        details += f"   **Parameters:**\n\n"
        properties = parameters.get('properties', {})
        required = parameters.get('required', [])
        
        if properties:
            for param_name, param_info in properties.items():
                param_type = param_info.get('type', 'unknown')
                param_desc = param_info.get('description', 'No description')
                is_required = param_name in required
                req_marker = " (required)" if is_required else " (optional)"
                
                details += f"   * ``{param_name}`` ({param_type}){req_marker}\n"
                details += f"     {param_desc}\n\n"
        else:
            details += f"   No parameters required.\n\n"
    else:
        details += f"   **Parameters:** No parameters required.\n\n"
    
    # Example usage
    details += f"   **Example Usage:**\n\n"
    details += f"   .. code-block:: python\n\n"
    details += f"      query = {{\n"
    details += f'          "name": "{tool_name}",\n'
    details += f'          "arguments": {{\n'
    
    # Generate example parameters
    if parameters and 'properties' in parameters:
        properties = parameters.get('properties')
        if properties:
            required = parameters.get('required', [])
            examples = []
            
            for param_name, param_info in properties.items():
                if param_name in required:
                    param_type = param_info.get('type', 'string')
                    if param_type == 'string':
                        examples.append(f'              "{param_name}": "example_value"')
                    elif param_type == 'integer':
                        examples.append(f'              "{param_name}": 10')
                    elif param_type == 'boolean':
                        examples.append(f'              "{param_name}": true')
                    elif param_type == 'array':
                        examples.append(f'              "{param_name}": ["item1", "item2"]')
                    else:
                        examples.append(f'              "{param_name}": "example_value"')
            
            if examples:
                details += ",\n".join(examples) + "\n"
    
    details += f"          }}\n"
    details += f"      }}\n"
    details += f"      result = tu.run(query)\n\n"
    
    return details


def generate_file_page(file_path, tools, is_remote=False):
    """Generate RST content for a single configuration file."""
    file_name = Path(file_path).name
    file_stem = Path(file_path).stem
    
    # Create title based on file name
    title = file_stem.replace("_", " ").replace("-", " ").title()
    
    # Determine if this is a remote tool
    tool_type_label = "Remote" if is_remote else "Local"
    
    content = f"{title}\n"
    content += "=" * len(title) + "\n\n"
    
    content += f"**Configuration File**: ``{file_path}``\n"
    content += f"**Tool Type**: {tool_type_label}\n"
    content += f"**Tools Count**: {len(tools)}\n\n"
    
    content += f"This page contains all tools defined in the ``{file_name}`` configuration file.\n\n"
    
    # List all tools with collapsible details
    content += "Available Tools\n"
    content += "---------------\n\n"
    
    for tool in sorted(tools, key=lambda x: x.get("name", "")):
        tool_name = tool.get("name", "Unknown")
        tool_type = tool.get("type", "Unknown")
        tool_desc = tool.get("description", "No description")
        
        # Truncate description if too long for summary
        if len(tool_desc) > 100:
            tool_desc = tool_desc[:97] + "..."
        
        content += f"**{tool_name}** (Type: {tool_type})\n"
        content += f"~" * (len(tool_name) + len(tool_type) + 15) + "\n\n"
        content += f"{tool_desc}\n\n"
        
        # Add collapsible details
        content += format_tool_details(tool)
        content += "\n"
    
    # Add navigation
    content += "Navigation\n"
    content += "----------\n\n"
    content += "* :doc:`tools_config_index` - Back to Tools Overview\n"
    if is_remote:
        content += "* :doc:`remote_tools` - Remote Tools Setup\n"
    else:
        content += "* :doc:`../guide/loading_tools` - Loading Local Tools\n"
    
    return content


def generate_main_index(config_by_file):
    """Generate main index page that links to individual file pages."""
    
    content = """Available Tools
==========================

ToolUniverse provides over 600 scientific tools organized into local and remote categories. This overview helps you discover and navigate the available tools for your research needs.

Tool Registry
---------------

Local Tools
~~~~~~~~~~~

Local tools are Python classes that run within the ToolUniverse process, providing high performance and seamless integration.

**Key Features:**
- High performance with no network overhead
- Automatic discovery and registration
- Full access to ToolUniverse features
- Easy development and testing


Remote Tools
~~~~~~~~~~~~

Remote tools integrate external services, APIs, and specialized systems running on different servers.

**Key Features:**
- Integration with external APIs and services
- Scalable computation offloading
- Support for different programming languages
- Secure isolation of sensitive operations

.. tip::
   **Discover Available Tools:**

   * :doc:`../guide/listing_tools` - Browse tools by category and explore the full catalog
   * :doc:`../tutorials/finding_tools` - Learn advanced search techniques and filters
   * :doc:`../tools/remote_tools` - Set up remote tool integrations
   * :doc:`../guide/loading_tools` - Load and configure local tools

Overview
-----------------------------

"""

    total_files = len(config_by_file)
    total_tools = sum(len(tools) for tools in config_by_file.values())

    content += f"* **Total Configuration Files**: {total_files}\n"
    content += f"* **Total Tools**: {total_tools}\n\n"

    # Separate local and remote tools
    local_tools = {}
    remote_tools = {}

    for file_path, tools in config_by_file.items():
        if file_path.startswith("remote_tools/"):
            remote_tools[file_path] = tools
        else:
            local_tools[file_path] = tools

    # Local Tools Section
    if local_tools:
        content += "Local Tools\n"
        content += "-----------\n\n"
        content += (
            ".. note::\n"
            "   Tools running within the ToolUniverse process\n\n"
        )
        
        # Group local tools by directory
        by_directory = defaultdict(list)
        for file_path, tools in local_tools.items():
            directory = (
                str(Path(file_path).parent) if "/" in file_path else "root"
            )
            by_directory[directory].append((file_path, tools))

        # Generate sections for each directory
        for directory in sorted(by_directory.keys()):
            if directory == "root":
                content += "Builtin Tools\n"
                content += "~" * 28 + "\n\n"
            else:
                content += f"{directory.title()} Tools\n"
                content += "~" * (len(directory) + 21) + "\n\n"

            files_in_dir = sorted(by_directory[directory], key=lambda x: x[0])

            for file_path, tools in files_in_dir:
                file_name = Path(file_path).name
                file_stem = Path(file_path).stem
                title = file_stem.replace("_", " ").replace("-", " ").title()
                
                # Add title for each file
                # content += f"\n\n"
                content += f".. toctree::\n"
                content += f"   :maxdepth: 1\n\n"
                content += f"   {title} <{file_stem}>\n\n"

    # Remote Tools Section
    if remote_tools:
        content += "\nRemote Tools\n"
        content += "------------\n\n"
        content += (
            ".. note::\n"
            "   Tools registered in with remote registry\n\n"
        )
        
        # Group remote tools by directory
        by_directory = defaultdict(list)
        for file_path, tools in remote_tools.items():
            directory = (
                str(Path(file_path).parent) if "/" in file_path else "remote_tools"
            )
            by_directory[directory].append((file_path, tools))

        # Generate sections for each directory
        for directory in sorted(by_directory.keys()):
            if directory == "remote_tools":
                content += "^" * 28 + "\n\n"
            else:
                content += "^" * (len(directory) + 21) + "\n\n"

            files_in_dir = sorted(by_directory[directory], key=lambda x: x[0])

            # Add unified toctree for this directory
            content += ".. toctree::\n"
            content += "   :maxdepth: 1\n\n"
            
            for file_path, tools in files_in_dir:
                file_name = Path(file_path).name
                file_stem = Path(file_path).stem
                title = file_stem.replace("_", " ").replace("-", " ").title()
                
                content += f"   {title} <{file_stem}>\n"
            
            content += "\n"

    # Add tools organized by type
    content += "\nTools by Type\n"
    content += "-------------\n\n"

    tools_by_type = defaultdict(list)
    for file_path, tools in config_by_file.items():
        for tool in tools:
            tool_type = tool.get("type", "Unknown")
            is_remote = file_path.startswith("remote_tools/")
            tools_by_type[tool_type].append((tool, file_path, is_remote))

    for tool_type in sorted(tools_by_type.keys()):
        content += f"**{tool_type}** ({len(tools_by_type[tool_type])} tools)\n"
        content += "~" * (len(tool_type) + 20) + "\n\n"

        # Separate local and remote tools of this type
        local_tools_of_type = [
            (tool, file_path) for tool, file_path, is_remote in tools_by_type[tool_type] 
            if not is_remote
        ]
        remote_tools_of_type = [
            (tool, file_path) for tool, file_path, is_remote in tools_by_type[tool_type] 
            if is_remote
        ]

        if local_tools_of_type:
            content += "*Local Tools:*\n"
            for tool, file_path in sorted(
                local_tools_of_type, key=lambda x: x[0].get("name", "")
            ):
                tool_name = tool.get("name", "Unknown")
                file_stem = Path(file_path).stem
                content += f"  * {tool_name} (from :doc:`{file_stem}`)\n"
            content += "\n"

        if remote_tools_of_type:
            content += "*Remote Tools:*\n"
            for tool, file_path in sorted(
                remote_tools_of_type, key=lambda x: x[0].get("name", "")
            ):
                tool_name = tool.get("name", "Unknown")
                file_stem = Path(file_path).stem
                content += f"  * {tool_name} (from :doc:`{file_stem}`)\n"
            content += "\n"

    # Quick search section
    content += "\nQuick Tool Search\n"
    content += "-----------------\n\n"
    content += (
        "Use your browser's search function (Ctrl+F / Cmd+F) to quickly find:\n\n"
    )
    content += "* **Tool Name**: Search for the exact tool name\n"
    content += (
        "* **File Path**: Search for ``filename.json`` to find all tools "
        "in that file\n"
    )
    content += (
        '* **Tool Type**: Search for specific tool types like "AgenticTool", '
        '"OpenTarget", etc.\n\n'
    )

    # Add navigation note
    content += ".. note::\n"
    content += (
        "   Each configuration file has its own dedicated page with detailed "
        "tool information and collapsible sections.\n\n"
    )

    return content


def main():
    """Main function to generate configuration index and individual file pages."""
    # Get the script directory and find the data directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "src" / "tooluniverse" / "data"

    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        sys.exit(1)

    print(f"Loading tool configurations from {data_dir}")

    # Load all tool configurations by file
    config_by_file = load_tool_configs_by_file(data_dir)
    print(f"Loaded configurations from {len(config_by_file)} files")

    # Create directories
    api_dir = script_dir / "api"
    api_dir.mkdir(exist_ok=True)
    tools_dir = script_dir / "tools"
    tools_dir.mkdir(exist_ok=True)

    # Generate main index page
    main_index_content = generate_main_index(config_by_file)
    main_index_file = tools_dir / "tools_config_index.rst"
    with open(main_index_file, "w", encoding="utf-8") as f:
        f.write(main_index_content)
    print(f"Generated main index: {main_index_file}")

    # Generate individual pages for each configuration file
    for file_path, tools in config_by_file.items():
        is_remote = file_path.startswith("remote_tools/")
        file_stem = Path(file_path).stem
        
        # Generate page content
        page_content = generate_file_page(file_path, tools, is_remote)
        
        # Write individual page
        page_file = tools_dir / f"{file_stem}.rst"
        with open(page_file, "w", encoding="utf-8") as f:
            f.write(page_content)
        print(f"Generated page: {page_file} ({len(tools)} tools)")

    print(f"\nGenerated {len(config_by_file)} individual pages")
    total_tools_indexed = sum(len(tools) for tools in config_by_file.values())
    print(f"Total tools indexed: {total_tools_indexed}")


if __name__ == "__main__":
    main()