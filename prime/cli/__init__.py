import click
import os
import datetime

from dask.distributed import Client
from prime.checkpoints import Checkpoints

def deltaToStr(delta):
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    seconds += delta.microseconds / 1e6

    lines = []
    if days > 0: lines.append("{}d".format(days))
    if hours > 0: lines.append("{}h".format(hours))
    if minutes > 0: lines.append("{}min".format(minutes))
    if seconds > 0: lines.append("{:.2f}s".format(seconds))

    return " ".join(lines)

@click.group()
@click.option(
    "--local/--distributed",
    default=False,
    help="Force Prime to execute the calculations in a local cluster"
)
def main(local):
    pass

@main.command()
@click.argument("input", nargs=1)
def solve(input):
    # Read the script file
    script = None
    with open(input, "rb") as f:
        script = f.read()
    
    cs = Checkpoints(os.path.splitext(os.path.basename(input))[0] + "_checkpoints.dat")

    # Compile the script
    code = compile(script, input, 'exec')

    namespace = {
        "scheduler": "local",
        "checkpoints": cs
    }

    # Start the execution
    start_date = datetime.datetime.now()

    from yaspin import yaspin
    with yaspin().cyan.point as sp:
        exec(code, namespace)
    end_date = datetime.datetime.now()

    delta = end_date - start_date

    print("Finished. (Took {})".format(deltaToStr(delta)))
