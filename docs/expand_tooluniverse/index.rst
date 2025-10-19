Extending ToolUniverse
======================

Learn how to extend ToolUniverse with your own custom tools. This section provides comprehensive guides for creating, registering, and contributing tools to the ToolUniverse ecosystem.

What You'll Learn
-----------------

- 🏠 **Local Tool Development**: Create tools that run within ToolUniverse
- 🔗 **Remote Tool Integration**: Connect with external services and APIs
- 📤 **Community Contributions**: Submit your tools to the ToolUniverse repository
- 🔧 **Advanced Patterns**: Best practices and advanced development techniques

Quick Start
-----------

**I want to quickly try creating a tool:**
→ :doc:`quick_start` - 5-minute tutorial to create your first tool

**I want to learn tool development systematically:**
→ Choose your path:
   - **Local tools**: :doc:`local_tools/tutorial` - Python classes for API wrappers, data processing
   - **Remote tools**: :doc:`remote_tools/tutorial` - MCP servers for microservices, heavy computation

**I want to contribute tools to the community:**
→ Choose your tool type:
   - **Local tools**: :doc:`contributing/local_tools` - Submit Python tools (requires modifying ``__init__.py``)
   - **Remote tools**: :doc:`contributing/remote_tools` - Submit MCP servers (requires deployment)

**I want to understand the differences:**
→ :doc:`reference/comparison` - Compare tool types and usage scenarios

Tool Types Overview
-------------------

Local Tools
~~~~~~~~~~~

Local tools are Python classes that run within the same process as ToolUniverse. They provide:

- **High Performance**: No network overhead
- **Easy Development**: Simple Python classes
- **Automatic Discovery**: Tools auto-register with decorators
- **Full Integration**: Access to all ToolUniverse features

**Best for:**
- API wrappers and data processing
- File manipulation utilities
- Lightweight computational tools

**Key Point**: Contributing local tools requires modifying ``__init__.py`` in 4 specific locations.

Remote Tools
~~~~~~~~~~~~

Remote tools allow you to integrate external services, APIs, or tools running on different servers. They provide:

- **Scalability**: Offload heavy computation to dedicated servers
- **Integration**: Connect with existing systems and services
- **Flexibility**: Use tools in different programming languages
- **Isolation**: Keep sensitive operations separate

**Best for:**
- External API integrations
- Microservice connections
- Cloud-based AI services
- Proprietary system connections

**Key Point**: Contributing remote tools requires providing a publicly accessible server or detailed deployment documentation.

Development Workflow
--------------------

1. **Plan Your Tool**
   - Define functionality and requirements
   - Choose between local or remote implementation
   - Design API and parameter structure

2. **Develop Your Tool**
   - Implement core functionality
   - Add proper error handling
   - Write comprehensive tests

3. **Document Your Tool**
   - Create clear documentation
   - Provide usage examples
   - Document all parameters and outputs

4. **Test Thoroughly**
   - Unit tests for all functionality
   - Integration tests with ToolUniverse
   - Test edge cases and error conditions

5. **Submit for Review** (if contributing)
   - Follow contribution guidelines
   - Create pull request
   - Address review feedback

Examples
--------

For detailed examples and code samples, see:

- **Local Tools**: :doc:`local_tools/tutorial` - Complete examples with step-by-step instructions
- **Remote Tools**: :doc:`remote_tools/tutorial` - MCP integration and API examples
- **Quick Start**: :doc:`quick_start` - Simple 5-minute example to get started

Getting Help
------------

If you need help with tool development:

- **Documentation**: Check the specific guides for detailed information
- **Examples**: Look at existing tools in the codebase
- **Community**: Ask questions in GitHub discussions
- **Issues**: Report bugs or request features

Resources
---------

- **ToolUniverse Repository**: https://github.com/mims-harvard/ToolUniverse
- **Issue Tracker**: https://github.com/mims-harvard/ToolUniverse/issues
- **Discussions**: https://github.com/mims-harvard/ToolUniverse/discussions
- **Documentation**: https://tooluniverse.readthedocs.io

Next Steps
----------

Ready to start? Choose your path:

* 🚀 **Quick Start**: :doc:`quick_start` - Create your first tool in 5 minutes
* 🏠 **Local Tools**: :doc:`local_tools/tutorial` - Learn local tool development
* 🔗 **Remote Tools**: :doc:`remote_tools/tutorial` - Learn remote tool integration
* 🎁 **Contributing**: :doc:`contributing/index` - Submit tools to the community
* 🔍 **Understanding**: :doc:`reference/comparison` - Compare different approaches

.. tip::
   **Getting Started**: We recommend starting with the quick start tutorial to understand the basics, then choosing the appropriate detailed guide based on your needs. The community is here to help you succeed!