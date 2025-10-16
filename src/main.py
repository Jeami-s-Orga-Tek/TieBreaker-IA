##
## PROJECT PRO, 2025
## TieBreaker
## File description:
## main
##

import argparse
import sys
import os
import stat
from pathlib import Path

def make_launcher(repo_root: Path) -> Path:
    launcher = repo_root / "TieBreaker"

    src = '''#!/usr/bin/env python3
import sys,os,importlib.util
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
SRC_DIR=os.path.join(BASE_DIR,"src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0,SRC_DIR)
try:
    import tiebreaker_cli as cli
except ModuleNotFoundError:
    cli_path=os.path.join(SRC_DIR,"tiebreaker_cli.py")
    if not os.path.exists(cli_path):
        print("Erreur: src/tiebreaker_cli.py introuvable à la racine du projet.", file=sys.stderr)
        sys.exit(1)
    spec=importlib.util.spec_from_file_location("tiebreaker_cli",cli_path)
    cli=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
sys.exit(cli.main(sys.argv[1:]))
'''

    launcher.write_text(src, encoding="utf-8")
    mode = launcher.stat().st_mode
    launcher.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return launcher


def cmd_build(args):
    repo_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parent.parent
    )
    cli = repo_root / "src" / "tiebreaker_cli.py"

    if not cli.exists():
        sys.stderr.write(
            f"[Error] {cli} untraceable. Place tiebreaker_cli.py in src/ at the root ({repo_root}).\n"
        )
        sys.exit(1)

    out = make_launcher(repo_root)
    print(f"Create: {out}")


def cmd_clean(args):
    repo_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parent.parent
    )
    p = repo_root / "TieBreaker"

    if p.exists():
        p.unlink()
        print(f"Delete: {p}")
    else:
        print("Nothing to delete.")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build POSIX launcher at repo root (src/tiebreaker_cli.py)"
    )
    ap.add_argument(
        "--project-root",
        help=(
            "Project root path (optional). If absent, auto detection in "
            "going through the files."
        ),
    )

    sp = ap.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("build", help="Generate ./TieBreaker at the root of project")
    b.set_defaults(func=cmd_build)

    c = sp.add_parser("clean", help="Delete ./TieBreaker at the root of project")
    c.set_defaults(func=cmd_clean)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())