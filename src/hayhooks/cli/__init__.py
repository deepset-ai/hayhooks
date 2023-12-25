# SPDX-FileCopyrightText: 2023-present Massimiliano Pippi <mpippi@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
import click

from hayhooks.__about__ import __version__
from hayhooks.cli.run import run
from hayhooks.cli.serve import serve
from hayhooks.cli.status import status


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="Hayhooks")
def hayhooks():
    pass


hayhooks.add_command(run)
hayhooks.add_command(serve)
hayhooks.add_command(status)
