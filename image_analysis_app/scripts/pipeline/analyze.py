"""Per-frame welding feature analysis.

This module combines preprocessing, geometry localization, arc/plasma checks,
and droplet/spatter detection into one frame-level result payload.
"""

import cv2
import numpy as np
from pathlib import Path
from colorama import Fore, Style

from scripts.preprocessing.backlit.wire import preprocess_backlit
from scripts.preprocessing.laserlit.wire import preprocess_laserlit
from scripts.preprocessing.backlit.arc import preprocess_backlit_arc
from scripts.preprocessing.laserlit.arc import preprocess_laserlit_arc
from scripts.weldpool.weldpool import localize_weldpool
from scripts.weldpool.base_metal import localize_basemetal
from scripts.wire.contours import identify_contours
from scripts.wire.edges import identify_wire_edges
from scripts.wire.searchspace import define_searchspace
from scripts.wire.pixels_to_mm import pixels_to_mm
from scripts.wire.wire_contour import identify_wire_contour
from scripts.wire.wire_tool_transition import find_wire_tool_transition
from scripts.wire.solid_molten_transition import solid_molten_transition
from scripts.wire.wiretip_localization import wiretip_localization
from scripts.wire.tapering_point import localize_tapering_point
from scripts.arc.arc import detect_arc
from scripts.arc.plasma.plasma import detect_plasma_channel_edges
from scripts.arc.plasma.attachment import plasma_attachment_height
from scripts.arc.plasma.width import calculate_plasma_channel_width
from scripts.droplets_spatters.detect import detect_droplets_spatters
from scripts.droplets_spatters.draw import draw_detected_blobs
from scripts.config import CONFIG

def analyze_image(folder_path: str, img_name: str, video_mode: str):
    """Analyze a single frame and return structured results plus visualization.

    Args:
        folder_path: Directory containing the image.
        img_name: File name of the image to analyze.

    Returns:
        Tuple `(results_dict, vis_image)` where `results_dict` contains
        extracted welding features and `vis_image` is an annotated debug frame.
    """

    # ================================
    # INITIALIZATION
    # ================================
    status = "success"

    results_dict = {
        "basemetal_y": None,
        "weldpool_location": None,
        "tooltip": None,
        "searchspace_box": None,
        "wiretip_location": None,
        "transition_point": None,
        "tapering_point": None,
        "arc_detected": None,
        "plasma_channel_avg_width": None,
        "plasma_attachment_height": None,
        "plasma_attachment_points": None,
        "mm_per_px": None,
        "droplets": [],
        "spatters": [],
    }

    print(f"\nAnalyzing image: {img_name}")

    image_path = Path(folder_path) / img_name
    image = cv2.imread(str(image_path))

    # ================================
    # LOAD CONFIGURATION
    # ================================
    if image is None:
        status = "error"
        print(Fore.RED + "ERROR: Could not read image.")
        print(Style.RESET_ALL)
        return results_dict, None


    # ================================
    # PREPROCESSING AND ANALYSIS STEPS
    # ================================
    
    print(f"Detected mode: {video_mode}")

    if video_mode == "backlit":
        preprocess = preprocess_backlit
        preprocess_arc_img = preprocess_backlit_arc
    elif video_mode == "laserlit":
        preprocess = preprocess_laserlit
        preprocess_arc_img = preprocess_laserlit_arc
    else:
        print(Fore.RED + f"ERROR: Unknown mode '{video_mode}'" + Style.RESET_ALL)
        return results_dict, None
    
    mode_cfg = CONFIG[video_mode]

    preprocessed_img, edges = preprocess(
        image,
        mode_cfg[f"{video_mode}_edge"]["canny1"],
        mode_cfg[f"{video_mode}_edge"]["canny2"],
        mode_cfg[f"{video_mode}_edge"]["morph"]
    )

    preprocessed_arc_img = preprocess_arc_img(image)

    # ================================
    # BASEMETAL LOCALIZATION
    # ================================
    if video_mode == "backlit":
        basemetal_y = localize_basemetal(
            edges,
            mode_cfg[f"{video_mode}_basemetal"]["min_edge_pixels"],
            mode_cfg[f"{video_mode}_basemetal"]["min_edge_width"],
        )
    else:
        basemetal_y = 375

    results_dict["basemetal_y"] = basemetal_y

    # ================================
    # WIRE EDGE DETECTION
    # ================================
    wire_edges = identify_wire_edges(
        image,edges,
        basemetal_y,
        mode_cfg[f"{video_mode}_wire_edges"]["hough_threshold"],
        mode_cfg[f"{video_mode}_wire_edges"]["min_line_length"],
        mode_cfg[f"{video_mode}_wire_edges"]["max_line_gap"],
        mode_cfg[f"{video_mode}_wire_edges"]["rho"],
        mode_cfg[f"{video_mode}_wire_edges"]["theta"],
        mode_cfg[f"{video_mode}_wire_edges"]["min_length"],
        mode_cfg[f"{video_mode}_wire_edges"]["angle_tolerance_deg"],
        mode_cfg[f"{video_mode}_wire_edges"]["vertical_tolerance_deg"],
        mode_cfg[f"{video_mode}_wire_edges"]["min_x_distance"],
        mode_cfg[f"{video_mode}_wire_edges"]["max_x_distance"],
    )

    if wire_edges is None:
        print(Fore.RED + "ERROR: Wire edges could not be detected.")
        print(Style.RESET_ALL)
        status = "error"
        return results_dict, None

    mm_per_px = pixels_to_mm(wire_edges, CONFIG["wire_width_mm"])
    results_dict["mm_per_px"] = mm_per_px

    # ================================
    # WIRE-TOOL TRANSITION (TOOLTIP) LOCALIZATION
    # ================================
    tooltip = find_wire_tool_transition(
        edges,
        wire_edges,
        CONFIG["wire_tool_transition"]["offset_from_center"],
        CONFIG["wire_tool_transition"]["start_offset"],
        CONFIG["wire_tool_transition"]["extend_upward"],
    )

    results_dict["tooltip"] = tooltip

    # ================================
    # WIRE SEARCHSPACE DEFINITION
    # ================================
    searchspace = define_searchspace(
        preprocessed_img,
        wire_edges,
        basemetal_y,
        tooltip,
        CONFIG["wire_searchspace"]["top_shift_px"],
        CONFIG["wire_searchspace"]["length_scale"],
        CONFIG["wire_searchspace"]["width_scale"],
    )

    if searchspace:
        preprocessed_wire, searchspace_inv, searchspace_box = searchspace
    else:
        preprocessed_wire = preprocessed_img
        searchspace_inv = None
        searchspace_box = None

    results_dict["searchspace_box"] = searchspace_box

    # ================================
    # WIRE CONTOUR IDENTIFICATION & WIRETIP LOCALIZATION
    # ================================
    contours, canny_edges_wire = identify_contours(preprocessed_wire, 
                                 mode_cfg[f"{video_mode}_contours"]["canny1"],
                                 mode_cfg[f"{video_mode}_contours"]["canny2"],
                                 mode_cfg[f"{video_mode}_contours"]["morph"])
    if contours is None:
        status = "warning"
        print("No contours found")

    wire_contour = identify_wire_contour(contours,
                                         CONFIG["wire_contour"]["min_length"],
                                         CONFIG["wire_contour"]["min_width"],
                                         CONFIG["wire_contour"]["max_width"],
                                         CONFIG["wire_contour"]["max_angle"],
                                         CONFIG["wire_contour"]["min_h_to_w"],
                                         CONFIG["wire_contour"]["split_threshold"])
    if wire_contour is None:
        status = "warning"
        print("No suitable wire contour found")

    wiretip_location= wiretip_localization(wire_contour)
    if wiretip_location is not None:
        if searchspace_inv is not None:
            wiretip_location = np.array([[[wiretip_location[0], wiretip_location[1]]]], dtype=np.float32)
            wiretip_location = cv2.perspectiveTransform(wiretip_location, searchspace_inv)[0, 0]
            wiretip_location = (int(wiretip_location[0]), int(wiretip_location[1]))

    if wiretip_location is None:
        status = "warning"
        print("Wiretip location not found")
    
    results_dict["wiretip_location"] = wiretip_location
    
    # ================================
    # WELDPOOL LOCALIZATION
    # ================================
    weldpool_location = localize_weldpool(
        wire_edges,
        edges,
        wiretip_location,
        basemetal_y,
        min_edge_width_px=CONFIG["weldpool"]["min_edge_width_px"],
        scan_half_width_px=CONFIG["weldpool"]["scan_half_width_px"],
        remove_circular_droplets=CONFIG["weldpool"]["remove_circular_droplets"],
        droplet_min_area_px=CONFIG["weldpool"]["droplet_min_area_px"],
        droplet_circularity_threshold=CONFIG["weldpool"]["droplet_circularity_threshold"],
        show_verification=CONFIG["weldpool"]["show_verification"],
    )

    if (
        weldpool_location is not None
        and basemetal_y is not None
        and int(round(float(weldpool_location[1]))) > int(round(float(basemetal_y)))
    ):
        weldpool_location = None

    if weldpool_location is None:
        status = "warning"
        weldpool_location = wiretip_location  # Assume short circuit
        print("Weldpool location not found")

    results_dict["weldpool_location"] = weldpool_location

    # ================================
    # PLASMA CHANNEL DETECTION
    # ================================
    plasma_edges = None
    plasma_attachment_points = None

    if wiretip_location is None or weldpool_location is None:
        arc_present = False
    else:
        arc_present = detect_arc(image, 
                                 wiretip_location, 
                                 weldpool_location,
                                 wire_edges,
                                 )

    if arc_present:
        print("Arc detected, attempting plasma channel detection...")
        results_dict["arc_detected"] = True
        plasma_edges = detect_plasma_channel_edges(
            preprocessed_arc_img,
            wiretip_location,
            basemetal_y,
            weldpool_location,
            mode_cfg["arc_detection"]["plasma_channel_edge"]["box_width"],
            mode_cfg,
        )
        if plasma_edges is not None and plasma_edges.size != 0:
            print("Plasma channel edges detected, calculating characteristics.")
            plasma_attachment_y, plasma_attachment_points = plasma_attachment_height(
                preprocessed_arc_img,
                wiretip_location,
                mode_cfg["arc_detection"]["plasma_channel_attachment"],
            )
            plasma_avg_width = calculate_plasma_channel_width(plasma_edges)
            results_dict["plasma_attachment_height"] = plasma_attachment_y
            results_dict["plasma_attachment_points"] = plasma_attachment_points
            results_dict["plasma_channel_avg_width"] = plasma_avg_width
        else:
            status = "warning"
            print("Plasma channel and attachment points not found")

    else:
        print("No arc detected, skipping plasma channel detection.")
        plasma_edges = None

    
    # ================================
    # TRANSITION POINT LOCALIZATION
    # ================================
    if video_mode == "laserlit":
        transition_point = solid_molten_transition(image,
                                                   wiretip_location,
                                                   wire_edges,
                                                   )
    else:
        transition_point = None

    results_dict["transition_point"] = transition_point

    # ================================
    # TAPERING POINT LOCALIZATION
    # ================================
    tapering_point = localize_tapering_point(edges, 
                                             wire_edges,
                                             mode_cfg[f"{video_mode}_tapering_point"]["search_window_radius"],
                                             mode_cfg[f"{video_mode}_tapering_point"]["vicinity_radius"],
                                             mode_cfg[f"{video_mode}_tapering_point"]["max_consecutive_misses"]
                                            )
    
    if tapering_point is None:
        status = "warning"
        print("Tapering point not found")

    results_dict["tapering_point"] = tapering_point

    # ================================
    # DROPLET AND SPATTER DETECTION
    # ================================
    droplets, spatters = detect_droplets_spatters(
        image,
        searchspace_box,
        weldpool_location,
        wiretip_location,
    )
    results_dict["droplets"] = droplets
    results_dict["spatters"] = spatters
    print(f"Detected {len(droplets)} droplets and {len(spatters)} spatters in {img_name}")


    # ================================
    # VISUALIZATION
    # ================================
    vis_image = image.copy()

    # Draw basemetal line
    if basemetal_y is not None:
        cv2.line(vis_image, (0, int(basemetal_y)),
                 (vis_image.shape[1], int(basemetal_y)),
                 (0, 255, 255), 2)
        
    # Draw wire contour
    #if wire_contour is not None:
    #    for contour in wire_contour:
    #        contour_pts = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
    #        if searchspace_inv is not None:
    #            contour_pts = cv2.perspectiveTransform(contour_pts, searchspace_inv)
    #        contour_pts = contour_pts.astype(np.int32)
    #        cv2.polylines(vis_image, [contour_pts], True, (255, 255, 0), 2)

    # Draw wire edges
    if wire_edges is not None:
        for line in wire_edges:
            x1, y1, x2, y2 = line
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw weldpool location
    if weldpool_location is not None:
        x, y = weldpool_location
        cv2.circle(vis_image, (x, y), 5, (255, 0, 255), -1)
        cv2.putText(vis_image, "Weldpool", (x + 10, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Draw wiretip
    if wiretip_location is not None:
        cv2.circle(vis_image, (int(wiretip_location[0]), int(wiretip_location[1])), 3, (0, 0, 255), -1)
        cv2.putText(vis_image, "Wiretip", (int(wiretip_location[0]) + 30, int(wiretip_location[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw transition point
    if transition_point is not None and video_mode == "laserlit":
        x, y = transition_point
        cv2.circle(vis_image, (int(x), int(y)), 3, (255, 0, 0), -1)
        cv2.putText(vis_image, "Transition point", (int(x) + 30, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw tapering point
    if tapering_point is not None:
        x, y = tapering_point
        cv2.circle(vis_image, (int(x), int(y)), 3, (255, 0, 255), -1)
        cv2.putText(vis_image, "Tapering point", (int(x) - 150, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Draw droplet and spatter detections
    draw_detected_blobs(vis_image, droplets, "droplet", (0, 200, 255))
    draw_detected_blobs(vis_image, spatters, "spatter", (0, 0, 255))

    # Draw searchspace box
    #if searchspace_box is not None:
    #    x, y, w, h = searchspace_box
    #    cv2.polylines(vis_image, [searchspace_box], True, (255, 0, 0), 1)

    # Draw tooltip line
    if tooltip is not None and len(tooltip) == 2:
        pt1, pt2 = tooltip
        cv2.line(vis_image, pt1, pt2, (255, 0, 255), 2)

    # Show if arc is detected
    if arc_present is not None:
        text = "Arc detected" if arc_present else "No arc"
        color = (0, 255, 0) if arc_present else (0, 0, 255)
        cv2.putText(vis_image, text, (10, vis_image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw plasma channel edges
    if plasma_edges is not None:
        vis_image[plasma_edges > 0] = (0, 0, 255)

    # Draw plasma attachment points
    if plasma_attachment_points is not None:
        for point in plasma_attachment_points:
            cv2.circle(vis_image, (int(point[0]), int(point[1])), 3, (255, 255, 0), -1)
        cv2.putText(vis_image, "Plasma attachment points", (int(plasma_attachment_points[0][0]) + 10, int(plasma_attachment_points[0][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    if mm_per_px is not None:
        print(f"Estimated mm per pixel: {mm_per_px:.4f} mm/px")

    
    if status == "success":
        print(Fore.GREEN + "SUCCESS")
    elif status == "warning":
        print(Fore.YELLOW + "WARNING")
    else:
        print(Fore.RED + "ERROR")

    print(Style.RESET_ALL)
    
    # Close all debugging windows (skip in headless OpenCV builds)
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    return results_dict, vis_image
