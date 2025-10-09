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

If you want to contribute to Hayhooks, we recommend using [Hatch](https://hatch.pypa.io/), the project's build and environment management tool:

```bash
# Clone the repository
git clone https://github.com/deepset-ai/hayhooks.git
cd hayhooks

# Install Hatch (if not already installed)
pip install hatch

# Run unit tests
hatch run test:unit

# Run integration tests
hatch run test:integration

# Run tests
hatch run test:all

# Format code
hatch run fmt

# Serve documentation locally
hatch run docs:serve
```

Hatch automatically manages virtual environments and dependencies for you. See available commands in `pyproject.toml`.

### Alternative: Manual Installation

If you prefer manual setup:

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
# Pull the image corresponding to Hayhooks main branch
docker pull deepset/hayhooks:main

# Run Hayhooks
docker run -p 1416:1416 deepset/hayhooks:main
```

You can inspect all available images on [Docker Hub](https://hub.docker.com/r/deepset/hayhooks/tags).

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

## Next Steps

After successful installation:

- [Quick Start](quick-start.md) - Get started with basic usage
- [Configuration](configuration.md) - Configure Hayhooks for your needs
- [Examples](../examples/overview.md) - Explore example implementations
