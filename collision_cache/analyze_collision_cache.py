#!/usr/bin/env python3
"""
Analyze collision_cache/collision_map_debug.json for disconnected "map components".

Why this exists:
- Some areas (instances/dungeons/templates) can have world coords far away from the main overworld.
- If you treat the cache as one map and render min/max bounds, you'll get a gigantic image with huge empty gaps.

This script:
- Buckets tiles into region-bins (default 64x64 tiles) per plane
- Builds a 4-neighbor adjacency graph of region bins
- Computes connected components (islands)
- Prints component sizes + bounding boxes so you can decide what to render/pathfind against

Usage (PowerShell):
  py .\\collision_cache\\analyze_collision_cache.py
  py .\\collision_cache\\analyze_collision_cache.py --bin-size 64 --top 10
  py .\\collision_cache\\analyze_collision_cache.py --cache D:\\repos\\bot_runelite_IL\\collision_cache\\collision_map_debug.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Iterable


RegionKey = Tuple[int, int, int]  # (rx, ry, plane)


@dataclass(frozen=True)
class RegionStats:
    tiles: int
    min_x: int
    max_x: int
    min_y: int
    max_y: int


@dataclass
class ComponentSummary:
    plane: int
    region_count: int
    tile_count: int
    min_x: int
    max_x: int
    min_y: int
    max_y: int

    @property
    def span_x(self) -> int:
        return self.max_x - self.min_x

    @property
    def span_y(self) -> int:
        return self.max_y - self.min_y


def _iter_tiles(collision_data: Dict) -> Iterable[Tuple[int, int, int]]:
    for t in collision_data.values():
        try:
            x = int(t.get("x", 0))
            y = int(t.get("y", 0))
            p = int(t.get("p", 0))
        except Exception:
            continue
        yield x, y, p


def load_collision_cache(cache_file: Path) -> Dict:
    data = json.load(cache_file.open("r", encoding="utf-8"))
    collision_data = data.get("collision_data") or {}
    if not isinstance(collision_data, dict):
        raise ValueError("cache file does not contain a dict at key 'collision_data'")
    return collision_data


def compute_regions(collision_data: Dict, bin_size: int) -> tuple[set[RegionKey], Dict[RegionKey, RegionStats]]:
    counts: Counter[RegionKey] = Counter()
    # region -> (min_x, max_x, min_y, max_y)
    mins: Dict[RegionKey, Tuple[int, int]] = {}
    maxs: Dict[RegionKey, Tuple[int, int]] = {}

    for x, y, p in _iter_tiles(collision_data):
        rx, ry = x // bin_size, y // bin_size
        k = (rx, ry, p)
        counts[k] += 1
        if k not in mins:
            mins[k] = (x, y)
            maxs[k] = (x, y)
        else:
            mnx, mny = mins[k]
            mxx, mxy = maxs[k]
            mins[k] = (min(mnx, x), min(mny, y))
            maxs[k] = (max(mxx, x), max(mxy, y))

    regions = set(counts.keys())
    stats: Dict[RegionKey, RegionStats] = {}
    for k, n in counts.items():
        mnx, mny = mins[k]
        mxx, mxy = maxs[k]
        stats[k] = RegionStats(tiles=n, min_x=mnx, max_x=mxx, min_y=mny, max_y=mxy)
    return regions, stats


def compute_components(regions: set[RegionKey], region_stats: Dict[RegionKey, RegionStats]) -> list[ComponentSummary]:
    seen: set[RegionKey] = set()
    comps: list[ComponentSummary] = []

    for start in regions:
        if start in seen:
            continue

        q = deque([start])
        seen.add(start)

        plane = start[2]
        region_count = 0
        tile_count = 0
        min_x = 10**9
        min_y = 10**9
        max_x = -(10**9)
        max_y = -(10**9)

        while q:
            rx, ry, p = q.popleft()
            if p != plane:
                # Shouldn't happen because we only enqueue same-plane neighbors, but keep safe.
                continue
            region_count += 1
            s = region_stats.get((rx, ry, p))
            if s:
                tile_count += s.tiles
                min_x = min(min_x, s.min_x)
                min_y = min(min_y, s.min_y)
                max_x = max(max_x, s.max_x)
                max_y = max(max_y, s.max_y)

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (rx + dx, ry + dy, plane)
                if n in regions and n not in seen:
                    seen.add(n)
                    q.append(n)

        comps.append(
            ComponentSummary(
                plane=plane,
                region_count=region_count,
                tile_count=tile_count,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
            )
        )

    comps.sort(key=lambda c: (c.tile_count, c.region_count), reverse=True)
    return comps


def main():
    ap = argparse.ArgumentParser(description="Analyze collision cache and find disconnected map components.")
    ap.add_argument(
        "--cache",
        type=str,
        default="collision_cache/collision_map_debug.json",
        help="Path to collision_map_debug.json (default: collision_cache/collision_map_debug.json)",
    )
    ap.add_argument("--bin-size", type=int, default=64, help="Region bin size in tiles (default: 64)")
    ap.add_argument("--top", type=int, default=10, help="How many top components to print (default: 10)")
    args = ap.parse_args()

    cache_file = Path(args.cache).expanduser().resolve()
    if not cache_file.exists():
        raise SystemExit(f"Cache file not found: {cache_file}")

    collision_data = load_collision_cache(cache_file)
    total_tiles = len(collision_data)
    print(f"cache_file: {cache_file}")
    print(f"tiles: {total_tiles:,}")

    regions, region_stats = compute_regions(collision_data, bin_size=args.bin_size)
    print(f"bin_size: {args.bin_size} tiles")
    print(f"region_bins: {len(regions):,}")

    # Basic plane distribution
    by_plane = Counter()
    for _, _, p in regions:
        by_plane[p] += 1
    print(f"regions_by_plane: {dict(sorted(by_plane.items()))}")

    comps = compute_components(regions, region_stats)
    print(f"components: {len(comps)}")

    # Summaries
    print("")
    print(f"Top {min(args.top, len(comps))} components by tile_count:")
    for i, c in enumerate(comps[: args.top], start=1):
        print(
            f"  #{i}: plane={c.plane} tiles={c.tile_count:,} regions={c.region_count:,} "
            f"bounds=({c.min_x},{c.min_y})..({c.max_x},{c.max_y}) span=({c.span_x},{c.span_y})"
        )

    # Quick outlier hint: show tiles beyond 10k in either axis
    out = Counter()
    for x, y, _ in _iter_tiles(collision_data):
        if x > 5000 or y > 5000:
            out["x>5000 or y>5000"] += 1
        if x > 10000 or y > 10000:
            out["x>10000 or y>10000"] += 1
    print("")
    print(f"outlier_counts: {dict(out)}")


if __name__ == "__main__":
    main()


