# Documentation Development

This guide covers how to work with and contribute to the Hayhooks documentation.

## Prerequisites

- Python 3.9+
- Hatch installed (`pip install hatch`)

## Building Documentation

### Using Hatch (Recommended)

The documentation can be built and served using Hatch commands:

#### Development Server (Recommended)

Start a local development server with live reloading using the dedicated docs environment:

```bash
# Using the dedicated docs environment (auto-installs all dependencies)
hatch run docs:serve

# With custom options
hatch run docs:serve --open --dirty
```

The server will be available at `http://localhost:8000/hayhooks/`

#### Build Production Site

Build the documentation for deployment:

```bash
# Using the docs environment (recommended)
hatch run docs:build

# Clean build (removes old files first)
hatch run docs:build --clean

# Strict build (fails on warnings)
hatch run docs:build --strict
```

The built site will be available in the `site/` directory.

#### Deploy Documentation

Deploy the documentation to GitHub Pages:

```bash
# Deploy to GitHub Pages
hatch run docs:deploy

# With custom options
hatch run docs:deploy --force
```

### Direct MkDocs Usage

If you prefer to use MkDocs directly:

```bash
# Install dependencies
pip install mkdocs-material

# Serve documentation
mkdocs serve

# Build documentation
mkdocs build
```

## Documentation Structure

```text
docs/
├── mkdocs.yml                 # Main configuration
├── index.md                   # Homepage
├── getting-started/           # Getting started guides
├── concepts/                  # Core concepts
├── features/                  # Feature documentation
├── advanced/                  # Advanced topics
├── deployment/                # Deployment guides
├── examples/                  # Example documentation
├── reference/                 # Reference documentation
├── about/                     # About information
├── assets/                    # Images and static assets
├── stylesheets/               # Custom CSS
└── javascripts/               # Custom JavaScript
```

## Adding New Documentation

### 1. Create New Files

Add new Markdown files in the appropriate directory:

```bash
# Create a new feature document
touch docs/features/new-feature.md

# Create a new example
touch docs/examples/new-example.md
```

### 2. Update Navigation

Update `mkdocs.yml` to include your new documentation in the navigation:

```yaml
nav:
  - Features:
    - New Feature: features/new-feature.md  # Add this line
    - OpenAI Compatibility: features/openai-compatibility.md
    - MCP Support: features/mcp-support.md
```

### 3. Add Images

Place images in the `docs/assets/` directory:

```bash
# Add an image
mv screenshot.png docs/assets/

# Reference in Markdown
![Screenshot](../assets/screenshot.png)
```

## Documentation Style Guide

### Writing Style

- Use clear, concise language
- Include practical examples
- Provide step-by-step instructions
- Use proper Markdown formatting

### Avoiding Redundancy

To maintain consistency and reduce maintenance burden:

- **Single Source of Truth**: Each topic should have one canonical location
  - README.md: Quick overview and getting started examples
  - docs/concepts/: Detailed conceptual explanations
  - docs/reference/: Complete API and configuration references

- **Cross-Referencing**: Link to the canonical source instead of duplicating content
  - Good: "For complete configuration options, see [Environment Variables Reference](../reference/environment-variables.md)"
  - Bad: Copying all environment variables into multiple pages

- **Next Steps**: Keep to 2-3 most relevant links per page
  - Focus on logical next actions for the reader
  - Avoid circular references (page A → page B → page A)

### Code Examples

Use proper code block formatting:

```python
# Python code
def example_function():
    return "Hello, World!"
```

```bash
# Shell commands
hatch run docs-serve
```

### Links

- **Internal Documentation**: Use relative links (e.g., `../concepts/pipeline-wrapper.md`)
- **External Resources**: Use absolute links (e.g., `https://haystack.deepset.ai/`)
- **README Links**: When linking from docs to README, use absolute GitHub URLs
- **Test Links**: Always verify links work before committing

## Testing Documentation

### Build Verification

Always test the documentation builds successfully:

```bash
hatch run docs:build --strict
```

### Link Verification

Check for broken links:

```bash
# Test locally
hatch run docs:serve

# Or use a link checker tool
pip install linkchecker
linkchecker http://localhost:8000/hayhooks/
```

### Preview Changes

Preview your changes in the browser:

```bash
hatch run docs:serve --open
```

## Deployment

### GitHub Pages

> **Note:** Automatic deployment via GitHub Actions is not yet configured. For now, documentation must be deployed manually. A GitHub Actions workflow will be added in the future.

### Manual Deployment

To deploy manually:

```bash
# Deploy to GitHub Pages (builds and deploys in one command)
hatch run docs:deploy

# Or build only without deploying
hatch run docs:build
# The site is now ready in the site/ directory
```

## Troubleshooting

### Common Issues

1. **Build Fails**
   - Check for Markdown syntax errors
   - Verify all links are valid
   - Ensure image paths are correct

2. **Navigation Issues**
   - Check `mkdocs.yml` syntax
   - Verify all referenced files exist
   - Test navigation structure

3. **Style Issues**
   - Check CSS file paths
   - Verify JavaScript syntax
   - Test in multiple browsers

### Getting Help

- Check MkDocs documentation: <https://www.mkdocs.org/>
- Review Material theme docs: <https://squidfunk.github.io/mkdocs-material/>
- Open an issue on GitHub: <https://github.com/deepset-ai/hayhooks/issues>

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your documentation changes
4. Test locally with `hatch run docs:serve`
5. Submit a pull request

Thank you for contributing to Hayhooks documentation!
