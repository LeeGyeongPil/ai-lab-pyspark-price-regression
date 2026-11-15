"""Command line entry point: ``price-regression <step> [options]``."""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import __version__, steps
from .data import BODIES
from .i18n import LANGS, detect_lang, t
from .session import get_spark, stop_spark
from .steps import EXIT_ERROR, EXIT_INPUT, EXIT_OK, Options, Run

MIN_ROWS = 50


def build_parser(lang: str) -> argparse.ArgumentParser:
    """Build the parser with help text in ``lang`` (so ``--help --lang ko`` is Korean)."""
    parser = argparse.ArgumentParser(prog="price-regression", description=t(lang, "prog_description"))
    parser.add_argument("step", nargs="?", default="all", choices=steps.STEPS, metavar="step",
                        help=t(lang, "help_step") + " [" + ", ".join(steps.STEPS) + "]")
    parser.add_argument("--lang", choices=LANGS, help=t(lang, "help_lang"))
    parser.add_argument("--rows", type=int, default=5000, help=t(lang, "help_rows"))
    parser.add_argument("--seed", type=int, default=42, help=t(lang, "help_seed"))
    parser.add_argument("--nonlinear", action="store_true", help=t(lang, "help_nonlinear"))
    parser.add_argument("--out", default="artifacts", help=t(lang, "help_out"))
    parser.add_argument("--age", type=float, help=t(lang, "help_age"))
    parser.add_argument("--km", type=float, help=t(lang, "help_km"))
    parser.add_argument("--body", help=t(lang, "help_body") + " (" + "/".join(BODIES) + ")")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Decide the language first: argparse prints --help before it has read --lang.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--lang")
    known, _ = pre.parse_known_args(argv)
    lang = detect_lang(known.lang)

    args = build_parser(lang).parse_args(argv)
    lang = detect_lang(args.lang)

    if args.step == "doctor":
        return steps.step_doctor(lang)

    if args.rows < MIN_ROWS:
        print(t(lang, "err_rows", value=args.rows), file=sys.stderr)
        return EXIT_INPUT
    if args.step == "predict" and (args.age is None or args.km is None or args.body is None):
        print(t(lang, "err_age_km_required"), file=sys.stderr)
        return EXIT_INPUT

    # Every other step needs Spark, which needs Java: check before starting the JVM.
    problem = steps.check_java(lang)
    if problem is not None:
        return problem

    opts = Options(lang=lang, rows=args.rows, seed=args.seed, nonlinear=args.nonlinear, out=args.out)
    try:
        print(t(lang, "starting_spark"), file=sys.stderr)
        run = Run(opts=opts, spark=get_spark())
        return _dispatch(run, args)
    except Exception as exc:  # noqa: BLE001 - last line of defense: show a readable message
        print(t(lang, "err_generic", error=exc), file=sys.stderr)
        return EXIT_ERROR
    finally:
        stop_spark()


def _dispatch(run: Run, args: argparse.Namespace) -> int:
    step = args.step
    if step == "predict":
        return steps.step_predict(run, args.age, args.km, args.body)
    if step == "all":
        steps.step_all(run)
    else:
        getattr(steps, f"step_{step}")(run)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
