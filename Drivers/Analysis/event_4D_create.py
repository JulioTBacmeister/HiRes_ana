#!/usr/bin/env python
"""
Extract atmospheric events (e.g. gravity-wave / blocking event lists) from
case output and pickle the results, for one or more regions/zlev combos.

Refactored from notebook cells. See `run_pipeline` for the core logic, and
the bottom of the file for CLI + YAML config wiring.
"""
import sys
import argparse
import importlib
from pathlib import Path

import yaml

rootdir_ = '../'
if rootdir_ not in sys.path:
    sys.path.append(rootdir_)
    print(f"a path to {rootdir_} added in {__name__}")

from Utils import MyConstants as Co

#import analysis_utils as auti
import file_utils as futi
import event_utils as euti
import event_io as eio
#from Utils import numerical_utils as nuti

Rdair = Co.Rdair()


def load_case(case, nsteps=None, start_date=None, super_lat_range=(-90., 90.)):
    """Read case netcdf output into the working object `A`."""
    A = futi.read_case(
        case=case,
        nsteps=nsteps,
        start_date=start_date,
        super_lat_range=list(super_lat_range),
    )
    return A


def extract_events(A, *, zlev_event, lat_range, lon_range=(0, 360),
                    exclude_orography=True,
                    fracs=(0.995, 0.90, 0.50, 0.25, 0.125, 0.0625),
                    peak_footprint=(3, 3), return_after_stage1=False,
                    label=None):
    """Build an event list (El) for one region/zlev and pickle it to disk."""
    El = euti.make_El(
        A=A,
        fractions_for_thresholds=list(fracs),
        zlev_event=zlev_event,
        lat_range=list(lat_range),
        lon_range=list(lon_range),
        exclude_orography=exclude_orography,
        peak_footprint=tuple(peak_footprint),
        return_after_stage1=return_after_stage1,
    )

    f = eio.pickle_write(El)
    print(f"[{label or 'region'}] wrote {f}")
    del El
    return f


def run_pipeline(config):
    """
    Run the full pipeline from a single config dict.

    Expected keys:
        case             str
        nsteps           int or None
        start_date       list or None
        super_lat_range  [min, max]
        regions          list of dicts, each matching extract_events() kwargs
                          (zlev_event, lat_range, lon_range, exclude_orography,
                           fracs, peak_footprint, label)
    """
    A = load_case(
        case=config["case"],
        nsteps=config.get("nsteps"),
        start_date=config.get("start_date"),
        super_lat_range=config.get("super_lat_range", (-90., 90.)),
    )

    outputs = []
    for region in config["regions"]:
        outputs.append(extract_events(A, **region))

    return outputs


# --------------------------------------------------------------------------
# CLI / YAML wiring
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None,
                    help="Path to YAML config (recommended for region lists). "
                         "CLI flags below override individual top-level fields.")
    p.add_argument("--case", type=str, default=None)
    p.add_argument("--nsteps", type=int, default=None)
    return p.parse_args()


def build_config_from_cli(args):
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    if args.case is not None:
        config["case"] = args.case
    if args.nsteps is not None:
        config["nsteps"] = args.nsteps

    return config


if __name__ == "__main__":
    args = parse_args()
    config = build_config_from_cli(args)
    run_pipeline(config)







