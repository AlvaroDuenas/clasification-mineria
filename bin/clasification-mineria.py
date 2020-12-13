import sys
from argparse import ArgumentParser

from clasification_mineria.main import App


def main(train_path: str, dev_path: str, output: str) -> None:
    app = App(train_path, dev_path)
    if output is not None:
        app.start(output)
    else:
        app.start()


if __name__ == '__main__':
    parser = ArgumentParser(prog=sys.argv[0])
    parser.add_argument("--train", "-t", nargs=1, metavar="<train_folder>",
                        dest="train", required=True, type=str)
    parser.add_argument("--dev", "-d", nargs=1, metavar="<dev_folder>",
                        dest="dev", required=True, type=str)
    parser.add_argument("--output", "-o", nargs=1, metavar="<output_folder>",
                        dest="output", required=False, type=str)
    args = parser.parse_args(sys.argv[1:])
    main(args.train[0], args.dev[0], args.output[0])
