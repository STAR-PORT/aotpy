"""
Entrypoint for CLI utility for converting other telemetry formats to AOT format files.
"""

import argparse

from .translators import get_available_translators


# TODO: Define type for available_translators
def make_parser(available_translators: dict) -> argparse.ArgumentParser:
    """Constructs an argparser for CLI converter utility updated with the available translators."""
    parser = argparse.ArgumentParser(
        prog='AOT Converter',
        description="Convert telemetry files to AOT format.",
        epilog="")

    parser.add_argument("-o", "--out", type=str, default="out.aot", help="output file name/path")

    subparsers = parser.add_subparsers(help="available translators", dest="system")
    for system_name, system_info in available_translators.items():
        system_parser = subparsers.add_parser(system_name)
        for param in system_info["params"]:
            system_parser.add_argument(f"--{param.name}", type=param.type, metavar=param.type.__name__.upper(), required=True)

    return parser


def main() -> None:
    available_translators = get_available_translators()

    # Create argparser
    parser = make_parser(available_translators)
    args = parser.parse_args()

    if args.system is None:
        parser.print_help()
        return

    # Retrieve desired system name and check if translator is vailable
    system_name = args.system.strip().lower()
    if system_name not in available_translators:
        raise ValueError(f"Translator for system \'{system_name}\' not available.")

    # Parse additional parameters for translator initialization
    required_params = [param.name for param in available_translators.get(system_name).get("params")]
    system_params = {param_name: param_value for param_name, param_value in vars(args).items() if
                     param_name in required_params}

    # Create system object and export AOT-compatible file
    system = available_translators[system_name]["cls"](**system_params)
    system.write_to_file(filename=args.out)


if __name__ == "__main__":
    main()
