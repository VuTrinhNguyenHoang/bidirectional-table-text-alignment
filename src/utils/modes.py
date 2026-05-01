MODES = ("debug", "small", "medium", "large", "full")


def add_mode_arg(parser, default="small"):
    parser.add_argument(
        "--mode",
        type=str,
        default=default,
        choices=MODES,
        help="Pipeline/evaluation mode. Role-specific modes default to this value.",
    )


def add_generator_mode_arg(parser):
    parser.add_argument(
        "--generator-mode",
        "--generator_mode",
        dest="generator_mode",
        type=str,
        default=None,
        choices=MODES,
        help="Mode used for generator data/checkpoints. Defaults to --mode.",
    )


def add_selector_mode_arg(parser):
    parser.add_argument(
        "--selector-mode",
        "--selector_mode",
        dest="selector_mode",
        type=str,
        default=None,
        choices=MODES,
        help="Mode used for cell selector data/checkpoints. Defaults to --mode.",
    )


def resolve_generator_mode(args):
    return args.generator_mode or args.mode


def resolve_selector_mode(args):
    return args.selector_mode or args.mode


def unique_modes(*modes):
    result = []
    seen = set()

    for mode in modes:
        if mode is None or mode in seen:
            continue
        result.append(mode)
        seen.add(mode)

    return result
