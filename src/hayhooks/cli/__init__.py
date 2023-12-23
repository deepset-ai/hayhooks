# SPDX-FileCopyrightText: 2023-present Massimiliano Pippi <mpippi@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
import click

from hayhooks.__about__ import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="Hayhooks")
def hayhooks():
    click.echo("Hello world!")
