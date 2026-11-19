"""Command line: python -m player_development {download,build,analyse,all}."""
from __future__ import annotations

import argparse
import sys
import time

from . import config


def cmd_download(args) -> None:
    from .download import download
    ids = download(workers=args.workers)
    print(f"Cached {len(ids)} matches (events + lineups) in {config.RAW_DIR}")


def cmd_build(args) -> None:
    from . import parse, quality
    tables = parse.build(workers=args.workers)
    checks = quality.run_checks(tables["matches"], tables["shots"], tables["players"])
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    checks.to_csv(config.REPORTS_DIR / "data_quality.csv", index=False)
    print(f"{len(tables['matches'])} matches, {len(tables['shots'])} shots, {len(tables['players'])} player-match rows")
    for row in checks.itertuples():
        print(f"  [{'ok' if row.passed else 'FAIL'}] {row.check}: {row.detail}")
    if not checks.passed.all():
        sys.exit("Data-quality checks failed; see reports/data_quality.csv")


def cmd_analyse(args) -> None:
    from . import finishing, panel, parse, profiles, quality, reliability, report, trajectories, xg

    tables = parse.load_processed()
    start = time.time()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    quality.run_checks(tables["matches"], tables["shots"], tables["players"]) \
        .to_csv(config.REPORTS_DIR / "data_quality.csv", index=False)
    xg_result = xg.run(tables["shots"])
    xg.save(xg_result)
    print(f"xG: selected {xg_result.selected}")

    names = tables["players"].drop_duplicates("player_id", keep="last").set_index("player_id").player_name
    fin = finishing.run(xg_result.shots, names)
    finishing.save(fin)
    print(f"Finishing: skill SD {fin.prior.sd:.3f} (95% upper bound {fin.wide_prior.sd:.3f})")

    seasons = panel.player_seasons(tables["players"])
    traj = trajectories.run(seasons)
    trajectories.save(traj, seasons)
    print(f"Trajectories: {len(seasons)} player-seasons, ICC range {traj.icc.min():.2f}-{traj.icc.max():.2f}")

    rel = reliability.run(tables["players"], seasons, iterations=args.iterations)
    reliability.save(rel)
    examples = profiles.run(seasons, rel)
    profiles.save(examples)
    report.build()
    print(f"Wrote {config.REPORTS_DIR / 'RESULTS.md'} in {time.time() - start:.0f} s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m player_development",
                                     description="Player development analytics on StatsBomb open data")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "build", "analyse", "all"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=4, help="parallel downloads or parsing processes")
        p.add_argument("--iterations", type=int, default=200, help="random splits in the reliability study")
    args = parser.parse_args(argv)
    if args.command in ("download", "all"):
        cmd_download(args)
    if args.command in ("build", "all"):
        cmd_build(args)
    if args.command in ("analyse", "all"):
        cmd_analyse(args)


if __name__ == "__main__":
    main()
