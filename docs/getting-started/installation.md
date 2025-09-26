# Installation

This guide covers how to install Hayhooks and its dependencies.

## System Requirements

- Python 3.9+
- Operating System: Linux, macOS, or Windows
- Memory: Minimum 512MB RAM, 2GB+ recommended
- Storage: Minimum 100MB free space

## Install from PyPI

### Standard Installation

```bash
pip install hayhooks
```

### Installation with MCP Support

If you want to use the [MCP Server](../features/mcp-support.md), you need to install the `hayhooks[mcp]` package:

```bash
pip install hayhooks[mcp]
```

**NOTE: You'll need to run at least Python 3.10+ to use the MCP Server.**

## Verify Installation

After installation, verify that Hayhooks is installed correctly:

```bash
# Check version
hayhooks --version

# Show help
hayhooks --help

# Show available commands
hayhooks --help
```

## Optional Dependencies

Depending on your use case, you may need additional dependencies:

### For OpenAI Integration

```bash
pip install openai
```

### For Web Content Processing

```bash
pip install trafilatura beautifulsoup4
```

### For Document Processing

```bash
pip install PyPDF2 python-docx
```

### For Async Operations

```bash
pip install aiohttp
```

## Development Installation

If you want to contribute to Hayhooks or develop with the latest features:

```bash
# Clone the repository
git clone https://github.com/deepset-ai/hayhooks.git
cd hayhooks

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Docker Installation

### Using Docker Hub

```bash
# Pull the latest image
docker pull deepset/hayhooks:latest

# Run Hayhooks
docker run -p 1416:1416 deepset/hayhooks:latest
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/deepset-ai/hayhooks.git
cd hayhooks

# Build the image
docker build -t hayhooks:latest .

# Run the image
docker run -p 1416:1416 hayhooks:latest
```

## Troubleshooting

### Common Issues

#### Permission Errors

If you encounter permission errors:

```bash
# Install with user permissions
pip install --user hayhooks

# Or use a virtual environment
python -m venv venv
source venv/bin/activate
pip install hayhooks
```

#### Module Not Found Errors

If you get "Module not found" errors:

```bash
# Install with all extras
pip install hayhooks[mcp,dev]

# Or install missing dependencies manually
pip install <missing-package>
```

#### Version Conflicts

If you encounter version conflicts:

```bash
# Use a virtual environment
python -m venv venv
source venv/bin/activate
pip install hayhooks

# Or upgrade pip first
pip install --upgrade pip
pip install hayhooks
```

### Getting Help

If you encounter issues during installation:

1. Check the [GitHub Issues](https://github.com/deepset-ai/hayhooks/issues)
2. Search existing issues for similar problems
3. Create a new issue with detailed information about your environment

## Next Steps

After successful installation:

- [Quick Start](quick-start.md) - Get started with basic usage
- [Configuration](configuration.md) - Configure Hayhooks for your needs
- [Examples](../examples/overview.md) - Explore example implementations
