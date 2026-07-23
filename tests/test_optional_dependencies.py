import subprocess
import sys
import textwrap


def test_base_import_and_cli_do_not_require_a2a_or_redis():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class OptionalDependencyBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if (
                    fullname == "a2a"
                    or fullname.startswith("a2a.")
                    or fullname == "redis"
                    or fullname.startswith("redis.")
                ):
                    raise ModuleNotFoundError(f"blocked optional dependency: {fullname}")
                return None

        sys.meta_path.insert(0, OptionalDependencyBlocker())

        import hayhooks
        from hayhooks.cli import hayhooks_cli
        from hayhooks.server.app import create_app

        assert hayhooks.__name__ == "hayhooks"
        assert callable(hayhooks_cli)
        assert callable(create_app)
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
