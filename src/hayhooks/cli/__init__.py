# SPDX-FileCopyrightText: 2023-present Massimiliano Pippi <mpippi@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
import click

from hayhooks.cli.run import run
from hayhooks.cli.deploy import deploy
from hayhooks.cli.status import status
from hayhooks.cli.undeploy import undeploy


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.version_option(prog_name="Hayhooks")
@click.option('-s', '--server', default="http://localhost:1416", help="Hayhooks server URL")
@click.option('-k', '--disable-ssl', default=False, is_flag=True, help="Disable SSL certificate verification")
@click.pass_context
def hayhooks(ctx, server, disable_ssl):
    ctx.obj = server, disable_ssl


hayhooks.add_command(run)
hayhooks.add_command(deploy)
hayhooks.add_command(status)
hayhooks.add_command(undeploy)
