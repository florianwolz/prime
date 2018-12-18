import click
import os

from dask.distributed import Client

@click.group()
@click.option(
    "--local/--distributed",
    default=False,
    help="Force Prime to execute the calculations in a local cluster"
)
def main(local):
    # Setup the client
    client = Client()

@main.command()
@click.argument("input", nargs=1)
def solve(input):
    print(os.getcwd())

    # Read the script file
    script = None
    with open(input, "rb") as f:
        script = f.read()

    # Compile the script
    code = compile(script, input, 'exec')

    namespace = {
        "scheduler": "local"
    }

    # Start the execution
    exec(code, namespace)
