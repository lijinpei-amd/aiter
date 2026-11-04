# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare two T0b layout-field dumps case by case, without the known additions.

    layout_ref_diff.py OLD NEW [--drop-field NAME]... [--drop-cfg-item NAME]...

OLD and NEW are ``layout_fields_*.txt`` files (``layout_fields.py`` output). Each
``--drop-field`` removes a field line (``  NAME = ...``) from both, each
``--drop-cfg-item`` removes ``NAME=cx(...)`` from the config serialization, and the
``#mangle_sha256`` lines (hashes of that serialization) are always removed. Every
other difference is printed under its case. The exit status is 1 when any is left.
"""

import argparse
import re
import sys


def _cases(path):
    out, cur = {}, None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("== "):
                cur = line[3:]
                out[cur] = []
            elif cur is not None:
                out[cur].append(line)
    return out


def _normalize(lines, fields, cfg_items):
    kept = []
    for line in lines:
        name = line.strip().split(" ")[0]
        if name == "#mangle_sha256" or name in fields:
            continue
        for item in cfg_items:
            line = re.sub(rf", {item}=cx\([^)]*\)", "", line)
        kept.append(line)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--drop-field", action="append", default=[])
    parser.add_argument("--drop-cfg-item", action="append", default=[])
    args = parser.parse_args()
    old, new = _cases(args.old), _cases(args.new)
    left = 0
    for case in sorted(set(old) | set(new)):
        if case not in old or case not in new:
            print(f"case only in {'NEW' if case in new else 'OLD'}: {case}")
            left += 1
            continue
        a = _normalize(old[case], args.drop_field, args.drop_cfg_item)
        b = _normalize(new[case], args.drop_field, args.drop_cfg_item)
        if a == b:
            continue
        left += 1
        print(f"== {case}")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                pairs = zip(re.split(r"(?<=\)), ", x), re.split(r"(?<=\)), ", y))
                diff = [f"{u.strip()} -> {v.strip()}" for u, v in pairs if u != v]
                print(f"  line {i}: " + "; ".join(diff)[:400])
        if len(a) != len(b):
            print(f"  {len(a)} lines -> {len(b)} lines")
    print(f"{len(set(old) | set(new))} cases, {left} with other differences")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
