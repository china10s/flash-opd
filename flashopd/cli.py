"""FlashOPD CLI: 一行命令启动蒸馏.

    flashopd --config configs/default.yaml
    flashopd --student_model Qwen/Qwen2.5-1.5B --teacher_model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import fields

from flashopd.config import OPDConfig


def parse_args() -> OPDConfig:
    parser = argparse.ArgumentParser(
        prog="flashopd",
        description="FlashOPD — 300行核心代码实现 On-Policy Distillation",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path")

    for f in fields(OPDConfig):
        flag = f"--{f.name}"
        if f.type == "bool" or f.type is bool:
            parser.add_argument(flag, type=lambda x: x.lower() in ("true", "1", "yes"), default=None)
        elif f.type == "int" or f.type is int:
            parser.add_argument(flag, type=int, default=None)
        elif f.type == "float" or f.type is float:
            parser.add_argument(flag, type=float, default=None)
        else:
            parser.add_argument(flag, type=str, default=None)

    args = parser.parse_args()

    if args.config:
        cfg = OPDConfig.from_yaml(args.config)
    else:
        cfg = OPDConfig()

    for f in fields(OPDConfig):
        val = getattr(args, f.name, None)
        if val is not None:
            setattr(cfg, f.name, val)

    return cfg


def main():
    cfg = parse_args()

    from flashopd.train import run_training

    run_training(cfg)


if __name__ == "__main__":
    main()
