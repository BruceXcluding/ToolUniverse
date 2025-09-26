# ToolUniverse MCP Integration with Gemini CLI

This tutorial will Tutorial you through setting up ToolUniverse as an MCP (Model Context Protocol) server for the Gemini CLI.

## Prerequisites

- Gemini CLI installed on your system ([Install guideline](https://github.com/google-gemini/gemini-cli))
- `tooluniverse>=0.2.0` installed
- `uv` package manager installed
- Valid API keys for external services (if required by specific tools)

## Configuration Steps

### 1. Locate Gemini CLI Configuration

The Gemini CLI uses configuration files to manage MCP servers. You can configure them:

- **Globally**: In the `~/.gemini/settings.json` file.
- **Per project**: In your project's `.gemini/settings.json` file.

### 2. Add ToolUniverse MCP Server

Create or open your `settings.json` file and add the ToolUniverse MCP server configuration:

```json
{
  "mcpServers": {
    "tooluniverse": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/your/gemini_running_env",
        "run",
        "tooluniverse-smcp-stdio"
      ]
    }
  },
}
```

**Important**: Replace `/path/to/your/gemini_running_env` with the actual absolute path to your gemini running folder.

### 3. Configuration Explanation

- **mcpServers**: The main container for all MCP server configurations
- **tooluniverse**: The name identifier for your ToolUniverse MCP server
- **command**: Uses `uv` package manager to run the MCP server
- **args**: Command line arguments passed to uv:
  - `--directory`: Specifies the working directory for the ToolUniverse project
  - `run`: Tells uv to run a command
  - `tooluniverse-smcp-stdio`: The specific MCP command to execute


### 4. Optional Configuration Properties

You can customize your ToolUniverse MCP server with additional properties:

#### Environment Variables
To provide API keys or other secrets to the server, use the `env` block.

**Basic Configuration:**
```json
{
  "mcpServers": {
    "tooluniverse": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/your/gemini_running_env",
        "run",
        "tooluniverse-smcp-stdio"
      ],
      "cwd": "/path/to/your/ToolUniverse",
      "env": {
        "PUBMED_API_KEY": "$PUBMED_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY": "$SEMANTIC_SCHOLAR_API_KEY"
      },
      "timeout": 45000,
      "trust": false
    }
  }
}
```

**With SummarizationHook and Optimized for Non-Coding Users:**
```json
{
  "mcpServers": {
    "tooluniverse": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/your/gemini_running_env",
        "run",
        "tooluniverse-smcp-stdio",
        "--exclude-tool-types",
        "PackageTool",
        "--hook-type",
        "SummarizationHook"
      ],
      "env": {
        "AZURE_OPENAI_API_KEY": "your-azure-openai-api-key",
        "AZURE_OPENAI_ENDPOINT": "https://your-resource.openai.azure.com",
        "PUBMED_API_KEY": "$PUBMED_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY": "$SEMANTIC_SCHOLAR_API_KEY"
      },
      "timeout": 45000,
      "trust": false
    }
  }
}
```

**Configuration Notes:**
- `--exclude-tool-types PackageTool`: Excludes package management tools to save context window space (recommended for non-coding users)
- `--hook-type SummarizationHook`: Enables output summarization for better readability
- `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`: Required for SummarizationHook functionality

#### Tool Filtering
If you want to use only specific ToolUniverse tools to reduce overhead:

```json
{
  "mcpServers": {
    "tooluniverse": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/your/gemini_running_env",
        "run",
        "tooluniverse-smcp-stdio"
      ],
      "includeTools": [
        "EuropePMC_search_articles",
        "ChEMBL_search_similar_molecules",
        "openalex_literature_search",
        "search_clinical_trials"
      ]
    }
  }
}
```

#### Excluding Specific Tools
To exclude certain tools while keeping the rest:

```json
{
  "mcpServers": {
    "tooluniverse": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/your/gemini_running_env",
        "run",
        "tooluniverse-smcp-stdio"
      ],
      "excludeTools": [
        "EuropePMC_search_articles",
        "ChEMBL_search_similar_molecules"
      ]
    }
  }
}
```

### 5. Start the Gemini CLI

After saving the configuration file, start or restart the Gemini CLI. The MCP server will be automatically discovered and connected.

### 6. Verify Integration

Use the `/mcp` command to check the status of your ToolUniverse MCP server:

```bash
/mcp
```

This will display:
- Server connection status (`CONNECTED` or `DISCONNECTED`)
- Available ToolUniverse tools
- Configuration details
- Any connection errors

## Example Usage

Once configured, you can use ToolUniverse tools naturally in your conversations:

### Scientific Literature Search
```
Find recent papers about CRISPR gene editing in cancer therapy published in the last 2 years
```

### Chemical Compound Research
```
Search for FDA-approved drugs that target the EGFR protein and show me their chemical structures
```

### Clinical Trial Analysis
```
Find ongoing clinical trials for Alzheimer's disease treatments in the United States
```

### Multi-Tool Research Workflow
```
I'm researching potential drug targets for Parkinson's disease. Can you:
1. Find recent papers on Parkinson's disease drug targets
2. Search for compounds that interact with those targets
3. Check if there are any ongoing clinical trials for those compounds
```

## Tool Discovery and Execution

### Discovery Process

When the Gemini CLI starts, it:

1.  **Connects to ToolUniverse**: Establishes a connection using stdio transport based on your `settings.json`.
2.  **Discovers tools**: Fetches all available ToolUniverse tools and their schemas.
3.  **Validates schemas**: Ensures compatibility with Gemini API requirements.
4.  **Registers tools**: Makes the tools available for use in conversations.

### Tool Execution Flow

When you request a scientific research task:

1.  **Tool selection**: Gemini automatically selects the most appropriate ToolUniverse tools.
2.  **Confirmation**: If `trust: false`, you'll be prompted to confirm the tool execution.
3.  **Execution**: The tool is called with the necessary parameters.
4.  **Response**: Results are formatted and displayed in a user-friendly manner.

### Trust and Security

For enhanced security, keep `trust: false` to:
- Review each tool call before it runs.
- Understand what data is being accessed or sent.
- Prevent unintended API calls to external services.

You can selectively trust tools by choosing "Always allow this tool" during confirmation prompts.

## Troubleshooting

### Common Issues and Solutions

#### Server Won't Connect
**Symptoms**: ToolUniverse server shows `DISCONNECTED` status in `/mcp`.

**Solutions**:
1.  Verify ToolUniverse is properly installed and the `tooluniverse-mcp-claude` command is runnable. Test it directly in your terminal:
    `cd /path/to/your/ToolUniverse && uv run tooluniverse-mcp-claude --help`
2.  Check that the `cwd` path in your configuration is correct and absolute.
3.  Ensure `uv` is installed and accessible in your system's PATH.
4.  Review any error messages in the CLI output, possibly with debug mode.

#### No Tools Discovered
**Symptoms**: Server connects but no tools are available.

**Solutions**:
1.  Verify the ToolUniverse MCP server is working by running it manually.
2.  Check if your `includeTools` or `excludeTools` filter is too restrictive or has typos.
3.  Ensure all ToolUniverse dependencies are installed (`uv sync`).
4.  Review your ToolUniverse installation and configuration.

#### Tools Not Executing
**Symptoms**: Tools are discovered but fail during execution.

**Solutions**:
1.  Check your `env` configuration for API keys. Ensure they are correct and have the necessary permissions.
2.  Verify network connectivity to the required scientific APIs.
3.  Increase the `timeout` value in your configuration for slow API responses.
4.  Review parameter validation errors in the CLI output.

#### Performance Issues
**Symptoms**: Tools are slow or time out.

**Solutions**:
1.  Increase the `timeout` value in the configuration.
2.  Use `includeTools` to load only the necessary tools, reducing startup overhead.
3.  Check network connectivity and latency to external APIs.
4.  Be mindful of potential rate limits for the external services you are calling.

### Debug Mode

Run the Gemini CLI with the `--debug` flag for detailed information:

```bash
gemini --debug
```

This provides verbose output about:
- MCP server connection attempts
- The tool discovery process
- Tool execution details and errors

## Advanced Configuration

### Multiple ToolUniverse Instances

You can configure multiple ToolUniverse instances, for example, to separate concerns or use different toolsets.

```json
{
  "mcpServers": {
    "tooluniverse-literature": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/your/gemini_running_env",
        "run",
        "tooluniverse-smcp-stdio"
      ],
      "cwd": "/path/to/your/ToolUniverse",
      "includeTools": ["EuropePMC_search_articles", "openalex_literature_search", "PubTator3_LiteratureSearch"],
      "timeout": 30000
    },
    "tooluniverse-compounds": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/your/gemini_running_env",
        "run",
        "tooluniverse-smcp-stdio"
      ],
      "cwd": "/path/to/your/ToolUniverse",
      "includeTools": ["ChEMBL_search_similar_molecules", "FDA_get_drug_names_by_indication", "drugbank_search"],
      "timeout": 45000
    }
  }
}
```

### Docker-based Setup

For containerized environments, you can configure the server to run inside Docker.

```json
{
  "mcpServers": {
    "tooluniverse-docker": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e", "PUBMED_API_KEY",
        "-e", "SEMANTIC_SCHOLAR_API_KEY",
        "your-tooluniverse-image:latest",
        "tooluniverse-smcp-stdio"
      ],
      "env": {
        "PUBMED_API_KEY": "$PUBMED_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY": "$SEMANTIC_SCHOLAR_API_KEY"
      }
    }
  }
}
```

## Security Considerations

- **API Keys**: Store sensitive API keys as environment variables and pass them to the server using the `env` block. Avoid hardcoding them in configuration files.
- **Trust Settings**: Keep `trust: false` for production use to maintain control over tool executions.
- **Network Access**: Ensure your firewall allows outbound connections to the necessary scientific APIs.
- **Rate Limits**: Be aware of and respect the API rate limits for all external services.

## Performance Tips

- **Tool Selection**: Use `includeTools` to load only the tools you need for a specific task.
- **Timeout Configuration**: Set appropriate timeouts based on expected API response times.
- **Connection Persistence**: The CLI maintains persistent connections to reduce startup overhead.
- **Caching**: Some ToolUniverse tools may implement caching to improve performance on repeated queries.

## Optimization for Non-Coding Users

If you're primarily using ToolUniverse for scientific research without coding needs, consider these optimizations:

### Recommended Configuration for Research-Only Users

```json
{
  "mcpServers": {
    "tooluniverse": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/your/gemini_running_env",
        "run",
        "tooluniverse-smcp-stdio",
        "--exclude-tool-types",
        "PackageTool",
        "--hook-type",
        "SummarizationHook"
      ],
      "env": {
        "AZURE_OPENAI_API_KEY": "your-azure-openai-api-key",
        "AZURE_OPENAI_ENDPOINT": "https://your-resource.openai.azure.com"
      }
    }
  }
}
```

### Benefits of This Configuration

1. **Context Window Optimization**:
   - `--exclude-tool-types PackageTool` removes package management tools
   - Saves significant context window space for more relevant scientific tools
   - Reduces tool discovery overhead

2. **Enhanced Readability**:
   - `--hook-type SummarizationHook` provides cleaner, more readable output
   - Automatically summarizes complex tool results
   - Improves user experience for research tasks

3. **Focused Tool Set**:
   - Prioritizes scientific research tools over development tools
   - Better performance for literature search, chemical analysis, and clinical data
   - Reduced cognitive load with fewer irrelevant tools

### When to Use PackageTool Exclusion

**Exclude PackageTool if you:**
- Primarily conduct scientific research
- Don't need package management capabilities
- Want to maximize context window for research tools
- Prefer cleaner, more focused tool interfaces

**Keep PackageTool if you:**
- Develop scientific software
- Need to manage Python packages
- Work with custom tool development
- Require full ToolUniverse functionality

## Getting Help

- **Status Check**: Use `/mcp` to monitor server status and available tools.
- **Tool Documentation**: Each ToolUniverse tool includes detailed documentation and examples, which can be inspected.
- **Debug Mode**: Use the `--debug` flag for verbose troubleshooting information.
- **ToolUniverse Issues**: Report ToolUniverse-specific issues on the project's official repository.

## Conclusion

With ToolUniverse integrated as an MCP server, the Gemini CLI becomes a powerful, interactive platform for scientific research. It enables seamless access to literature databases, chemical compound information, clinical trial data, and advanced research tools through natural language.
