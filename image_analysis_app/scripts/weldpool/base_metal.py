"""Basemetal surface localization from edge-row continuity."""

import numpy as np

SEARCH_START_OFFSET_PX = 50


def localize_basemetal(edges, min_edge_pixels, min_edge_width):
    """
    Find basemetal surface y-location by scanning upward, starting 50 px above the
    bottom edge, until a sufficiently long contiguous edge segment is found.

    - min_edge_pixels: minimum number of edge pixels in the row (basic density filter)
    - min_edge_width: minimum contiguous run length of edge pixels
    """

    h = edges.shape[0]
    start_y = max(h - SEARCH_START_OFFSET_PX - 1, 0)

    for y in range(start_y, -1, -1):
        row = edges[y, :]

        # Basic filter: skip rows with too few edge pixels overall
        if np.count_nonzero(row) < min_edge_pixels:
            continue

        xs = np.flatnonzero(row)
        if xs.size == 0:
            continue

        # Compute lengths of contiguous runs in xs
        # A break occurs when diff != 1
        breaks = np.where(np.diff(xs) != 1)[0]

        # Start indices of runs in xs array
        run_starts = np.r_[0, breaks + 1]
        # End indices of runs in xs array
        run_ends = np.r_[breaks, xs.size - 1]

        # Run lengths in pixel units
        run_lengths = xs[run_ends] - xs[run_starts] + 1

        max_run = int(run_lengths.max())

        if max_run >= min_edge_width:
            print(f"Base metal localized at y={y} (max contiguous run={max_run})")
            return int(y)

    print("Base metal not localized")
    return None
