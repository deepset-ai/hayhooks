import importlib.util
import functools
import sys
from hayhooks import log

MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None

MCP_ERROR_MSG = (
    "MCP functionality requires the 'mcp' package. "
    "Ensure you're running Python 3.10 or higher. "
    "Then install with 'pip install hayhooks[mcp]'"
)


def ensure_mcp_available():
    """Raise an informative error if MCP is not available"""
    if not MCP_AVAILABLE:
        log.error(MCP_ERROR_MSG)
        raise ImportError(MCP_ERROR_MSG)


def requires_mcp(func):
    """Decorator to ensure MCP is available before running the function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            ensure_mcp_available()
            return func(*args, **kwargs)
        except ImportError:
            # Check if we're in a CLI context (Typer/Click)
            # Exit more gracefully in this case
            if 'typer' in sys.modules:
                import typer

                raise typer.Exit(code=1)
            # Re-raise the exception for non-CLI contexts
            raise

    return wrapper
