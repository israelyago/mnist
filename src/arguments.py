import argparse
import pathlib

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "-r", "--release", help="Run as a release version", action="store_true"
)
_parser.add_argument(
    "-l",
    "--logs",
    help="Folder path to save the logs",
    default="logs",
    type=pathlib.Path,
)
_parser.add_argument(
    "-m",
    "--model",
    help="Path to load the model from",
    type=pathlib.Path,
)
_parser.add_argument(
    "-a",
    "--arch",
    help="Which architecture to use. One of ['lenet5', 'udlbook']",
    type=str,
    default="udlbook",
)
_args = _parser.parse_args()


def get_args():
    return _args
