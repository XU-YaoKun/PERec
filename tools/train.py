#!/usr/bin/env python

import argparse
import os

from perec.config.base import cfg
from perec.engine.trainer import train


def parse_args():
    parser = argparse.ArgumentParser(description="personalized recommendation")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using command line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load configuration
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = os.path.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        os.makedirs(output_dir, exist_ok=True)

    train(cfg, output_dir)


if __name__ == "__main__":
    main()
