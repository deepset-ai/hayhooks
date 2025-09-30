# Installation

This guide covers how to install Hayhooks and its dependencies.

## System Requirements

- Python 3.9+
- Operating System: Linux, macOS, or Windows
- Memory: Minimum 512MB RAM, 2GB+ recommended
- Storage: Minimum 100MB free space

## Install from PyPI

=== "Standard Installation"

    ```bash
    pip install hayhooks
    ```

    This includes all core features for deploying and running pipelines.

=== "With MCP Support"

    ```bash
    pip install hayhooks[mcp]
    ```

    Includes all standard features plus [MCP Server](../features/mcp-support.md) support for integration with AI development tools like Cursor and Claude Desktop.

    !!! warning "Python 3.10+ Required"
        You'll need to run at least Python 3.10+ to use the MCP Server.

=== "From Source"

    ```bash
    git clone https://github.com/deepset-ai/hayhooks.git
    cd hayhooks
    pip install -e .
    ```

    Useful for development or testing the latest unreleased features.

## Verify Installation

After installation, verify that Hayhooks is installed correctly:

```bash
# Check version
hayhooks --version

# Show help
hayhooks --help
```

## Development Installation

If you want to contribute to Hayhooks, clone the repository and install in editable mode:

```bash
# Clone the repository
git clone https://github.com/deepset-ai/hayhooks.git
cd hayhooks

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
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

# Build with Docker Buildx Bake (current platform) and load into Docker
cd docker
IMAGE_NAME=hayhooks IMAGE_TAG_SUFFIX=local docker buildx bake --load

# Run the image
docker run -p 1416:1416 hayhooks:local
```

Optional (multi-arch + push to registry):

```bash
# Build and push multi-platform image (amd64, arm64)
# Replace <your-user> and <tag> accordingly
cd docker
IMAGE_NAME=<your-user>/hayhooks IMAGE_TAG_SUFFIX=<tag> HAYHOOKS_VERSION=<tag> docker buildx bake --push
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
