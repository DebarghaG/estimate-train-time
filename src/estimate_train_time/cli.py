"""
Command-line interface for estimate-train-time.
"""

import argparse
import os
import sys

from estimate_train_time.estimator.prediction import one_batch_predict
from estimate_train_time.data import get_examples_path


def list_examples():
    """List all available example configurations."""
    examples_path = get_examples_path()

    if hasattr(examples_path, '__fspath__'):
        examples_dir = str(examples_path)
    else:
        examples_dir = examples_path

    if not os.path.isdir(examples_dir):
        print("No example configurations found.")
        return

    examples = [f for f in os.listdir(examples_dir) if f.endswith(('.yml', '.yaml'))]

    if not examples:
        print("No example configurations found.")
        return

    print("Available example configurations:")
    print("-" * 40)
    for example in sorted(examples):
        name = os.path.splitext(example)[0]
        print(f"  {name}")
    print()
    print("Use 'estimate-train-time show-example <name>' to view a configuration.")
    print("Use 'estimate-train-time predict --example <name>' to run prediction.")


def show_example(name):
    """Show the contents of an example configuration."""
    examples_path = get_examples_path()

    if hasattr(examples_path, '__fspath__'):
        examples_dir = str(examples_path)
    else:
        examples_dir = examples_path

    # Try with .yml and .yaml extensions
    for ext in ['.yml', '.yaml', '']:
        config_path = os.path.join(examples_dir, name + ext)
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                print(f"# Configuration: {name}")
                print(f"# Path: {config_path}")
                print("-" * 40)
                print(f.read())
            return

    print(f"Example '{name}' not found.")
    print("Use 'estimate-train-time list-examples' to see available examples.")
    sys.exit(1)


def predict(config_path=None, example=None):
    """Run prediction with a configuration file."""
    if example:
        examples_path = get_examples_path()

        if hasattr(examples_path, '__fspath__'):
            examples_dir = str(examples_path)
        else:
            examples_dir = examples_path

        # Try with .yml and .yaml extensions
        for ext in ['.yml', '.yaml', '']:
            candidate = os.path.join(examples_dir, example + ext)
            if os.path.isfile(candidate):
                config_path = candidate
                break

        if config_path is None:
            print(f"Example '{example}' not found.")
            print("Use 'estimate-train-time list-examples' to see available examples.")
            sys.exit(1)

    if not config_path:
        print("Error: Please provide --config or --example")
        sys.exit(1)

    if not os.path.isfile(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Running prediction with config: {config_path}")
    print("-" * 40)

    try:
        result = one_batch_predict(config_path)
        print(f"\nEstimated time cost of current training config: {result:.2f} us")
        print(f"                                               = {result/1000:.2f} ms")
        print(f"                                               = {result/1000000:.4f} s")
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog='estimate-train-time',
        description='Distributed training time estimator for Large Language Models'
    )
    parser.add_argument('--version', action='store_true', help='Show version and exit')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # predict command
    predict_parser = subparsers.add_parser('predict', help='Run time estimation prediction')
    predict_parser.add_argument('--config', '-c', type=str, help='Path to configuration YAML file')
    predict_parser.add_argument('--example', '-e', type=str, help='Name of bundled example configuration')

    # list-examples command
    subparsers.add_parser('list-examples', help='List available example configurations')

    # show-example command
    show_parser = subparsers.add_parser('show-example', help='Show an example configuration')
    show_parser.add_argument('name', type=str, help='Name of the example configuration')

    args = parser.parse_args()

    if args.version:
        from estimate_train_time import __version__
        print(f"estimate-train-time {__version__}")
        return

    if args.command == 'predict':
        predict(config_path=args.config, example=args.example)
    elif args.command == 'list-examples':
        list_examples()
    elif args.command == 'show-example':
        show_example(args.name)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
