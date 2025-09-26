#!/usr/bin/env python3
"""
Auto-generate Remote Tools documentation script
Scans README files in src/tooluniverse/remote directory and generates
corresponding RST documentation
"""

import sys
from pathlib import Path
from typing import List, Dict


class RemoteToolsDocGenerator:
    """Remote Tools documentation generator"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = project_root / "docs"
        self.remote_dir = project_root / "src" / "tooluniverse" / "remote"
        self.remote_docs_dir = self.docs_dir / "tools" / "remote"
        self.tools_dir = self.docs_dir / "tools"

    def scan_remote_tools(self) -> List[Dict[str, str]]:
        """Scan remote tools directory and return tool information"""
        tools = []

        if not self.remote_dir.exists():
            print(f"❌ Remote directory does not exist: {self.remote_dir}")
            return tools

        for tool_dir in sorted(self.remote_dir.iterdir()):
            if not tool_dir.is_dir():
                continue

            tool_name = tool_dir.name
            readme_files = []

            # Look for README files
            for readme_pattern in ["README.md", "README.rst", "README.txt"]:
                readme_path = tool_dir / readme_pattern
                if readme_path.exists():
                    readme_files.append(readme_path)

            if readme_files:
                # Select the most appropriate README file
                readme_file = self._select_best_readme(readme_files, tool_name)
                tools.append({
                    'name': tool_name,
                    'dir': tool_dir,
                    'readme': readme_file,
                    'relative_path': readme_file.relative_to(self.project_root)
                })
                print(f"✅ Found tool: {tool_name} -> {readme_file.name}")
            else:
                print(f"⚠️  Tool {tool_name} has no README file found")

        return tools

    def _select_best_readme(self, readme_files: List[Path],
                            tool_name: str) -> Path:
        """Select the most appropriate README file"""
        # Priority order
        priority_patterns = [
            "README.md",
            "README.rst",
            "README.txt"
        ]

        for pattern in priority_patterns:
            for readme_file in readme_files:
                if readme_file.name == pattern:
                    return readme_file

        # If no match found, return the first one
        return readme_files[0]

    def generate_individual_md(self, tool: Dict[str, str]) -> str:
        """Generate individual tool Markdown file content"""
        readme_path = tool['relative_path']

        # Directly read and include README file content, using original title
        readme_file = self.project_root / readme_path
        if readme_file.exists():
            with open(readme_file, 'r', encoding='utf-8') as f:
                readme_content = f.read()

            # Directly return README content without adding extra title
            return readme_content
        else:
            return f"**Note:** README file not found: {readme_path}\n"

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Simple Markdown to HTML conversion"""
        import re

        html = markdown_text

        # Convert headers
        html = re.sub(r'^# (.+)$', r'   <h1>\1</h1>',
                      html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'   <h2>\1</h2>',
                      html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'   <h3>\1</h3>',
                      html, flags=re.MULTILINE)
        html = re.sub(r'^#### (.+)$', r'   <h4>\1</h4>',
                      html, flags=re.MULTILINE)

        # Convert bold
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

        # Convert italic
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Convert code blocks
        html = re.sub(r'```(.+?)```', r'<pre><code>\1</code></pre>',
                      html, flags=re.DOTALL)

        # Convert inline code
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

        # Convert lists
        html = re.sub(r'^- (.+)$', r'   <li>\1</li>', html, flags=re.MULTILINE)

        # Convert paragraphs
        lines = html.split('\n')
        result_lines = []
        in_list = False

        for line in lines:
            if (line.strip().startswith('<h') or
                    line.strip().startswith('<li>') or
                    line.strip().startswith('<pre>')):
                if in_list and not line.strip().startswith('<li>'):
                    result_lines.append('   </ul>')
                    in_list = False
                if line.strip().startswith('<li>') and not in_list:
                    result_lines.append('   <ul>')
                    in_list = True
                result_lines.append(line)
            elif line.strip() == '':
                if in_list:
                    result_lines.append('   </ul>')
                    in_list = False
                result_lines.append(line)
            else:
                if in_list:
                    result_lines.append('   </ul>')
                    in_list = False
                if line.strip():
                    result_lines.append(f'   <p>{line}</p>')
                else:
                    result_lines.append(line)

        if in_list:
            result_lines.append('   </ul>')

        return '\n'.join(result_lines)

    def generate_main_remote_tools_rst(self,
                                       tools: List[Dict[str, str]]) -> str:
        """Generate main remote_tools.rst file"""
        content = "Remote Tools Setup\n"
        content += "=========================\n\n"
        content += ("This section aggregates setup guides for all "
                    "integrations under ``src/tooluniverse/remote``.\n\n")

        content += ".. toctree::\n"
        content += "   :maxdepth: 1\n\n"

        for tool in tools:
            tool_name = tool['name']
            content += f"   remote/{tool_name}.md\n"

        content += "\n"
        return content

    def create_directories(self):
        """Create necessary directories"""
        self.remote_docs_dir.mkdir(parents=True, exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {self.remote_docs_dir}")
        print(f"✅ Created directory: {self.tools_dir}")

    def generate_all_docs(self):
        """Generate all documentation"""
        print("🔍 Scanning Remote Tools...")
        tools = self.scan_remote_tools()

        if not tools:
            print("❌ No remote tools found")
            return False

        print(f"📊 Found {len(tools)} remote tools")

        # Create directories
        self.create_directories()

        # Generate individual tool Markdown files
        print("\n📝 Generating individual tool Markdown files...")
        for tool in tools:
            tool_name = tool['name']
            md_content = self.generate_individual_md(tool)

            # Write to tools/remote/ directory
            md_file = self.remote_docs_dir / f"{tool_name}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
            print(f"✅ Generated: {md_file}")

        # Generate main remote_tools.rst file
        print("\n📝 Generating main remote_tools.rst file...")
        main_content = self.generate_main_remote_tools_rst(tools)
        main_file = self.tools_dir / "remote_tools.rst"
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(main_content)
        print(f"✅ Generated: {main_file}")

        print(f"\n🎉 Successfully generated documentation for "
              f"{len(tools)} remote tools!")
        return True


def main():
    """Main function"""
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print("🚀 Remote Tools Documentation Generator")
    print("=" * 40)
    print(f"📁 Project root directory: {project_root}")
    print(f"📁 Documentation directory: {script_dir}")

    # Create generator
    generator = RemoteToolsDocGenerator(project_root)

    # Generate documentation
    success = generator.generate_all_docs()

    if success:
        print("\n✅ All documentation generation completed!")
        print("\n📋 Generated files:")
        print("   - docs/tools/remote_tools.rst")
        print("   - docs/tools/remote/*.md")
    else:
        print("\n❌ Documentation generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
