"""Locate wire/tool transition points along offset scan lines."""

import numpy as np


def find_wire_tool_transition(
    edges,
    wire_edges,
    offset_from_center,
    start_offset,
    extend_upward,
):
    '''
    Find transition points by searching along two lines parallel to the wire edges.
    Parameters:
    - edges: Canny edges of a preprocessed image (numpy array).
    - wire_edges: A tuple of two line segments representing the wire edges: ((x1,y1,x2,y2), (x1,y1,x2,y2)).
    - offset_from_center: Distance in pixels to offset the search lines from the center between the two wire edges.
    - start_offset: Distance in pixels to start the search from the top of the offset lines.
    - extend_upward: Distance in pixels to extend the search lines upward beyond the start point.

    Returns:
    - A list of transition points found on the two lines, each as a tuple (x, y). Returns None if no points are found.
    '''

    l1, l2 = wire_edges
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    vx = x2 - x1
    vy = y2 - y1
    vlen = float(np.hypot(vx, vy))
    if vlen == 0:
        return None
    ux = vx / vlen
    uy = vy / vlen
    nx = -uy
    ny = ux

    d1 = x1 * nx + y1 * ny
    d2 = x3 * nx + y3 * ny
    center_d = (d1 + d2) / 2.0

    def _offset_line(xa, ya, xb, yb, d_line):
        sign = 1.0 if (d_line - center_d) >= 0 else -1.0
        ox = nx * offset_from_center * sign
        oy = ny * offset_from_center * sign
        return (xa + ox, ya + oy, xb + ox, yb + oy)

    l1o = _offset_line(x1, y1, x2, y2, d1)
    l2o = _offset_line(x3, y3, x4, y4, d2)

    def _search_on_line(line):
        xa, ya, xb, yb = line
        if ya <= yb:
            x_up, y_up, x_dn, y_dn = xa, ya, xb, yb
        else:
            x_up, y_up, x_dn, y_dn = xb, yb, xa, ya

        dx = x_dn - x_up
        dy = y_dn - y_up
        length = float(np.hypot(dx, dy))
        if length == 0 or start_offset >= length:
            return None
        ux_l = dx / length
        uy_l = dy / length

        x_up_ext = x_up - ux_l * extend_upward
        y_up_ext = y_up - uy_l * extend_upward

        start_x = x_up_ext + ux_l * start_offset
        start_y = y_up_ext + uy_l * start_offset

        scan_len = length + extend_upward - start_offset
        if scan_len <= 0:
            print("Invalid scan length for wire tool transition search")
            return None
        for t in range(0, int(scan_len) + 1):
            x = int(round(start_x + ux_l * t))
            y = int(round(start_y + uy_l * t))
            if x < 0 or y < 0 or x >= edges.shape[1] or y >= edges.shape[0]:
                continue
            if edges[y, x] > 0:
                return (int(x), int(y))
        print("Wire tool transition point not found on line")
        return None

    pt1 = _search_on_line(l1o)
    pt2 = _search_on_line(l2o)

    points = [p for p in (pt1, pt2) if p is not None]
    return points if points else None
