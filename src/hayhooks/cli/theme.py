from rich.theme import Theme

from hayhooks.colors import BRAND_COLOR, ERROR_COLOR, MUTED_COLOR, SUCCESS_COLOR, WARNING_COLOR

HAYHOOKS_STYLES = {
    "accent": BRAND_COLOR,
    "accent.bold": f"bold {BRAND_COLOR}",
    "success": SUCCESS_COLOR,
    "success.bold": f"bold {SUCCESS_COLOR}",
    "error": ERROR_COLOR,
    "error.bold": f"bold {ERROR_COLOR}",
    "warning": WARNING_COLOR,
    "warning.bold": f"bold {WARNING_COLOR}",
    "muted": MUTED_COLOR,
    "heading": f"bold {BRAND_COLOR}",
    "pipeline.name": f"bold {BRAND_COLOR}",
    "pipeline.status": SUCCESS_COLOR,
    "table.header": f"bold {BRAND_COLOR}",
    "progress.description": f"bold {BRAND_COLOR}",
}

hayhooks_theme = Theme(HAYHOOKS_STYLES)

SUCCESS_PREFIX = f"[{SUCCESS_COLOR}]✔[/{SUCCESS_COLOR}]"
ERROR_PREFIX = f"[{ERROR_COLOR}]✘ Error:[/{ERROR_COLOR}]"
WARNING_PREFIX = f"[{WARNING_COLOR}]![/{WARNING_COLOR}]"


def apply_typer_theme() -> None:
    """Override typer.rich_utils style constants to match the hayhooks theme."""
    import typer.rich_utils as ru

    ru.STYLE_OPTION = f"bold {BRAND_COLOR}"
    ru.STYLE_SWITCH = f"bold {SUCCESS_COLOR}"
    ru.STYLE_METAVAR = f"bold {BRAND_COLOR}"
    ru.STYLE_USAGE = BRAND_COLOR
    ru.STYLE_USAGE_COMMAND = "bold"
    ru.STYLE_DEPRECATED = ERROR_COLOR
    ru.STYLE_HELPTEXT_FIRST_LINE = ""
    ru.STYLE_HELPTEXT = "dim"
    ru.STYLE_OPTION_ENVVAR = f"dim {BRAND_COLOR}"
    ru.STYLE_REQUIRED_SHORT = ERROR_COLOR
    ru.STYLE_REQUIRED_LONG = f"dim {ERROR_COLOR}"
    ru.STYLE_OPTIONS_PANEL_BORDER = BRAND_COLOR
    ru.STYLE_COMMANDS_PANEL_BORDER = BRAND_COLOR
    ru.STYLE_COMMANDS_TABLE_FIRST_COLUMN = f"bold {BRAND_COLOR}"
    ru.STYLE_ERRORS_PANEL_BORDER = ERROR_COLOR
    ru.STYLE_ABORTED = ERROR_COLOR
    ru.RICH_HELP = f"Try [{BRAND_COLOR}]'{{command_path}} {{help_option}}'[/] for help."
