#!/usr/bin/env python3
"""
map_quality.py

Оценивает .pgm карту (результат SLAM) и сохраняет:
 - пошаговые визуализации (gray, normalized, threshold, LoG, cleaned, corners, contours, skeleton)
 - summary.json и печатает метрики

Метрики (возвращаемые):
 - total_cells, occupied_cells, free_cells, occupied_fraction
 - num_connected_occupied_components, mean_cluster_area, largest_cluster_area
 - num_closed_regions (контуров, замкнутых областей)
 - num_harris_corners
 - num_branch_points (пересечений), num_end_points (концы)
 - hough_lines_count, line_angle_histogram (binned)
 - intermediate image filenames (в results/)

Методы, использованные согласно статье:
 - порог: mean или Otsu
 - LoG (Laplacian of Gaussian) для выделения структуры
 - Harris для структурных углов
 - Suzuki (cv2.findContours) для закрытых областей
 - скелетизация + подсчёт степеней для пересечений/концов

Пример:
  python map_quality.py map.pgm --outdir results --threshold_method otsu
"""

import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- helper functions ----------

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def imwrite(path, img):
    cv2.imwrite(str(path), img)

def normalize_to_uint8(img):
    # img may be 16-bit .pgm or 8-bit. Scale to 0..255 where lighter==free
    if img.dtype == np.uint8:
        return img.copy()
    mi = float(img.min())
    ma = float(img.max())
    if ma == mi:
        return (np.zeros_like(img, dtype=np.uint8))
    scaled = (255.0 * (img - mi) / (ma - mi)).astype(np.uint8)
    return scaled

def otsu_threshold(img_gray):
    # expects uint8 image
    _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def mean_threshold(img_gray):
    thr = int(np.mean(img_gray))
    _, th = cv2.threshold(img_gray, thr, 255, cv2.THRESH_BINARY)
    return th

def laplacian_of_gaussian(img_gray, ksize=3, sigma=1.0):
    # Gaussian blur then Laplacian
    g = cv2.GaussianBlur(img_gray, (0,0), sigmaX=sigma, sigmaY=sigma)
    lap = cv2.Laplacian(g, cv2.CV_64F, ksize=ksize)
    # normalize to uint8 for visualization
    lap_norm = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return lap, lap_norm

def remove_small_components(bin_img, min_area=30):
    # bin_img: 0 or 255 (occupied as 255)
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img//255, connectivity=8)
    out = np.zeros_like(bin_img)
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels==i] = 255
    return out, nb_components-1

def harris_corners_steps(img_gray, blockSize=2, ksize=3, k=0.04, thresh_rel=0.01):
    # return image of Harris response normalized and corner points
    dst = cv2.cornerHarris(img_gray, blockSize, ksize, k)
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, 0, 255, cv2.NORM_MINMAX)
    dst_norm_uint8 = dst_norm.astype(np.uint8)
    thresh = thresh_rel * dst_norm.max()
    # corners coordinates
    corners = np.argwhere(dst_norm > thresh)  # y,x pairs
    return dst_norm_uint8, corners

# Zhang-Suen thinning implementation (binary image: foreground==255, background==0)
def zhang_suen_thinning(bin_img):
    # input: uint8 0/255
    img = (bin_img//255).astype(np.uint8)
    prev = np.zeros(img.shape, np.uint8)
    diff = None

    def neighbours(x, y, im):
        x_1 = im[x-1:x+2, y-1:y+2].flatten()
        return [x_1[1], x_1[2], x_1[5], x_1[8], x_1[7], x_1[6], x_1[3], x_1[0]]
    def transitions(neis):
        n = neis + neis[0:1]
        return sum((n[i]==0 and n[i+1]==1) for i in range(8))

    rows, cols = img.shape
    changing = True
    while changing:
        changing = False
        markers = []
        for step in [0,1]:
            to_be_removed = []
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    P = img[i,j]
                    if P != 1:
                        continue
                    neis = neighbours(i,j,img)
                    C = sum(neis)
                    if C < 2 or C > 6:
                        continue
                    if transitions(neis) != 1:
                        continue
                    if step == 0:
                        if neis[0] * neis[2] * neis[4] != 0:
                            continue
                        if neis[2] * neis[4] * neis[6] != 0:
                            continue
                    else:
                        if neis[0] * neis[2] * neis[6] != 0:
                            continue
                        if neis[0] * neis[4] * neis[6] != 0:
                            continue
                    to_be_removed.append((i,j))
            if to_be_removed:
                changing = True
                for (i,j) in to_be_removed:
                    img[i,j] = 0
    return (img*255).astype(np.uint8)

def count_branch_and_end_points(skel):
    # skel: binary 0/255
    s = (skel//255).astype(np.uint8)
    rows, cols = s.shape
    branch = 0
    endp = 0
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if s[i,j] == 0:
                continue
            nb = int(np.sum(s[i-1:i+2, j-1:j+2])//255) - 1
            if nb == 1:
                endp += 1
            elif nb > 2:
                branch += 1
    return branch, endp

def compute_hough_lines(img_bin, minLineLength=20, maxLineGap=5):
    # expects edges or binary where lines present
    lines = cv2.HoughLinesP(img_bin, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is None:
        return [], None
    lines = lines.reshape(-1,4)
    # compute angles in degrees (0..180)
    angles = []
    for x1,y1,x2,y2 in lines:
        ang = np.degrees(np.arctan2((y2-y1),(x2-x1)))
        if ang < 0:
            ang += 180.0
        angles.append(ang)
    return lines.tolist(), np.array(angles)

# ---------- main processing pipeline ----------
def extend_to_match(img_small, img_big, fill_value=255):
    """
    Делает img_small того же размера, что img_big,
    добавляя поля (unknown) вокруг.

    fill_value=255 — неизвестная область (обычно белый).
    """
    h_big, w_big = img_big.shape[:2]
    h_small, w_small = img_small.shape[:2]

    # Если карта уже больше — не уменьшаем
    new_h = max(h_big, h_small)
    new_w = max(w_big, w_small)

    # Создаём новый "холст"
    result = np.full((new_h, new_w), fill_value, dtype=img_small.dtype)

    # Вставляем маленькую карту по центру (симметричное расширение)
    top = (new_h - h_small) // 2
    left = (new_w - w_small) // 2

    result[top:top + h_small, left:left + w_small] = img_small
    return result


def process_map(path_in, outdir, threshold_method='mean', min_component_area=30, clean_area=20):
    ensure_dir(outdir)
    p_in = Path(path_in)
    # 1) load .pgm as grayscale
    img = cv2.imread(str(p_in), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Не удалось загрузить {path_in}")
    # normalize to 0..255 uint8 (following article: brighter == free)
    img8 = normalize_to_uint8(img)
    imwrite(Path(outdir)/"01_normalized.png", img8)

    # According to article: unknown cells are white by default (treated as free by human),
    # but for closed area detection they consider unknown as occupied in some iterations.
    # We'll compute threshold both by mean and Otsu, allow user to choose.
    if threshold_method == 'otsu':
        bin_img = otsu_threshold(img8)
    else:
        bin_img = mean_threshold(img8)

    # In the map representation often darker == occupied. We assume: darker -> occupied.
    # But thresholding above produced white==free (255). Convert to occupied==255:
    # If threshold produced binary with free==255, invert.
    # We'll decide by checking: count of white vs black: if white > black, likely white=free => invert.
    white = int(np.sum(bin_img==255))
    black = int(np.sum(bin_img==0))
    if white > black:
        # assume white == free -> invert so occupied==255
        bin_occ = cv2.bitwise_not(bin_img)
    else:
        bin_occ = bin_img.copy()

    imwrite(Path(outdir)/"02_threshold_occupied.png", bin_occ)

    # 2) remove small components (noise)
    cleaned, nc = remove_small_components(bin_occ, min_area=min_component_area)
    imwrite(Path(outdir)/"03_cleaned.png", cleaned)

    # 3) LoG for structure visualization (as in article)
    lap, lap_vis = laplacian_of_gaussian(img8, ksize=3, sigma=1.0)
    imwrite(Path(outdir)/"04_log.png", lap_vis)

    # 4) Harris corners on preprocessed structure:
    # Prepare image for Harris: article suggests remapping intensities so higher -> free, unknown->0
    # Here we use normalized grayscale
    # optionally apply Gaussian and threshold small blobs
    img_harris_input = cv2.GaussianBlur(img8, (3,3), 0)
    dst_norm_uint8, corners = harris_corners_steps(img_harris_input, blockSize=3, ksize=3, k=0.04, thresh_rel=0.4)
    # overlay corners
    corner_vis = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    for (y,x) in corners:
        cv2.circle(corner_vis, (int(x), int(y)), 2, (0,0,255), -1)
    imwrite(Path(outdir)/"05_harris_corners.png", corner_vis)

    # 5) find closed regions (contours) on a binary where unknown treated as occupied -> do twice as described:
    # We'll follow article: iterate treating unknown as occupied with different reductions? Simpler: find contours on cleaned occupancy.
    contours, hierarchy = cv2.findContours(cleaned.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # count closed areas: contours that have no child? We'll count external contours with area > threshold
    closed_count = 0
    contour_vis = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area >= 50:  # ignore tiny
            cv2.drawContours(contour_vis, [cnt], -1, (0,255,0), 1)
            closed_count += 1
    imwrite(Path(outdir)/"06_contours.png", contour_vis)

    # 6) connected components on occupied for clustering stats
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned//255, connectivity=8)
    cluster_areas = []
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        cluster_areas.append(int(area))
    if len(cluster_areas)==0:
        mean_cluster_area = 0
        largest_cluster_area = 0
    else:
        mean_cluster_area = float(np.mean(cluster_areas))
        largest_cluster_area = int(np.max(cluster_areas))

    # 7) skeletonization + branch/end points
    skel = zhang_suen_thinning(cleaned)
    imwrite(Path(outdir)/"07_skeleton.png", skel)
    branch_points, end_points = count_branch_and_end_points(skel)

    # For visualization overlay branch and endpoints
    skel_color = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    s_coords = np.argwhere(skel>0)
    for (y,x) in s_coords:
        skel_color[int(y),int(x)] = [200,200,200]  # faint gray for skeleton
    # mark branch points red, endpoints blue
    rows, cols = skel.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if skel[i,j]==0: continue
            nb = int(np.sum(skel[i-1:i+2, j-1:j+2])//255) - 1
            if nb == 1:
                cv2.circle(skel_color, (j,i), 2, (255,0,0), -1)  # blue
            elif nb > 2:
                cv2.circle(skel_color, (j,i), 2, (0,0,255), -1)  # red
    imwrite(Path(outdir)/"08_skeleton_marked.png", skel_color)

    # 8) Hough lines on edges of the cleaned map (to assess straightness of walls)
    edges = cv2.Canny(cleaned, 50, 150)
    imwrite(Path(outdir)/"09_edges.png", edges)
    lines, angles = compute_hough_lines(edges, minLineLength=20, maxLineGap=10)
    # draw lines
    hough_vis = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    for l in lines:
        x1,y1,x2,y2 = l
        cv2.line(hough_vis, (x1,y1),(x2,y2),(0,255,255),1)
    imwrite(Path(outdir)/"10_hough_lines.png", hough_vis)

    # 9) counts of occupied/free
    total_cells = img8.size
    occupied_cells = int(np.sum(cleaned>0))
    free_cells = total_cells - occupied_cells
    occupied_fraction = occupied_cells / total_cells

    # assemble summary
    summary = {
        "input_file": str(p_in),
        "image_shape": img8.shape,
        "total_cells": int(total_cells),
        "occupied_cells": int(occupied_cells),
        "free_cells": int(free_cells),
        "occupied_fraction": float(occupied_fraction),
        "num_connected_occupied_components": int(nb_components-1),
        "mean_cluster_area": float(mean_cluster_area),
        "largest_cluster_area": int(largest_cluster_area),
        "num_closed_regions_contours": int(closed_count),
        "num_harris_corners": int(len(corners)),
        "num_branch_points": int(branch_points),
        "num_end_points": int(end_points),
        "hough_lines_count": int(len(lines)),
        "hough_angle_bins_deg": {
            "0-22.5": int(np.sum((angles>=0) & (angles<22.5))) if angles is not None else 0,
            "22.5-45": int(np.sum((angles>=22.5) & (angles<45))) if angles is not None else 0,
            "45-67.5": int(np.sum((angles>=45) & (angles<67.5))) if angles is not None else 0,
            "67.5-90": int(np.sum((angles>=67.5) & (angles<90))) if angles is not None else 0,
            "90-112.5": int(np.sum((angles>=90) & (angles<112.5))) if angles is not None else 0,
            "112.5-135": int(np.sum((angles>=112.5) & (angles<135))) if angles is not None else 0,
            "135-157.5": int(np.sum((angles>=135) & (angles<157.5))) if angles is not None else 0,
            "157.5-180": int(np.sum((angles>=157.5) & (angles<180))) if angles is not None else 0,
        },
        "intermediate_images": {
            "normalized": "01_normalized.png",
            "threshold_occupied": "02_threshold_occupied.png",
            "cleaned": "03_cleaned.png",
            "log": "04_log.png",
            "harris_corners_overlay": "05_harris_corners.png",
            "contours_overlay": "06_contours.png",
            "skeleton": "07_skeleton.png",
            "skeleton_marked": "08_skeleton_marked.png",
            "edges": "09_edges.png",
            "hough_lines": "10_hough_lines.png"
        }
    }

    # save summary
    with open(Path(outdir)/"summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # print brief summary
    print("=== Map quality summary ===")
    for k,v in summary.items():
        if k == "intermediate_images" or k=="hough_angle_bins_deg":
            continue
        print(f"{k}: {v}")
    print("Hough angle bins:", summary["hough_angle_bins_deg"])
    print("Intermediate images saved to:", outdir)
    return summary


def diff_heatmap(img1, img2):
    """
    Создаёт heatmap различий между двумя картами одинакового размера.

    img1, img2 — uint8, 0..255
    0 (чёрный) – occupied
    255 (белый) – unknown/free

    Возвращает BGR-изображение heatmap.
    """

    if img1.shape != img2.shape:
        raise ValueError("Images must be same size for heatmap")

    # Разница по модулю
    diff = cv2.absdiff(img1, img2)

    # Создаём цветную карту
    h, w = img1.shape
    heat = np.zeros((h, w, 3), dtype=np.uint8)

    # Маски
    same = diff < 10  # почти одинаковые пиксели
    img1_occ = img1 < 128  # занято (чёрное)
    img2_occ = img2 < 128

    # Отличия разных типов
    only1 = np.logical_and(img1_occ, ~img2_occ)  # карта1 занята → карта2 свободна
    only2 = np.logical_and(img2_occ, ~img1_occ)  # карта2 занята → карта1 свободна
    both_diff = np.logical_and(~same, ~only1)
    both_diff = np.logical_and(both_diff, ~only2)

    # Раскраска
    heat[same] = [180, 180, 180]  # серый (совпадают)
    heat[only1] = [0, 0, 255]  # красный (занято только у карты1)
    heat[only2] = [255, 0, 0]  # синий (занято только у карты2)
    heat[both_diff] = [0, 255, 0]  # зелёный (отличаются, но не занятостью: неизвестно/серая зона)

    return heat



def compare_maps(path1, path2, outdir, threshold_method='mean', min_component_area=30):
    ensure_dir(outdir)

    img1 = normalize_to_uint8(cv2.imread(path1, cv2.IMREAD_UNCHANGED))
    img2 = normalize_to_uint8(cv2.imread(path2, cv2.IMREAD_UNCHANGED))

    # выравнивание размеров
    if img1.shape != img2.shape:
        if img1.shape[0]*img1.shape[1] > img2.shape[0]*img2.shape[1]:
            img2_ext = extend_to_match(img2, img1, fill_value=205)
            img1_ext = img1
        else:
            img1_ext = extend_to_match(img1, img2, fill_value=205)
            img2_ext = img2

        cv2.imwrite(os.path.join(outdir, "map1_extended.png"), img1_ext)
        cv2.imwrite(os.path.join(outdir, "map2_extended.png"), img2_ext)
    else:
        img1_ext, img2_ext = img1, img2

    heatmap = diff_heatmap(img1_ext, img2_ext)
    cv2.imwrite(os.path.join(outdir, "maps_difference_heatmap.png"), heatmap)
    # временные файлы
    tmp1 = os.path.join(outdir, "tmp_map1.pgm")
    tmp2 = os.path.join(outdir, "tmp_map2.pgm")
    cv2.imwrite(tmp1, img1_ext)
    cv2.imwrite(tmp2, img2_ext)

    print("\n=== Processing map 1 ===")
    summary1 = process_map(tmp1, os.path.join(outdir, "map1"), threshold_method, min_component_area)

    print("\n=== Processing map 2 ===")
    summary2 = process_map(tmp2, os.path.join(outdir, "map2"), threshold_method, min_component_area)

    # Создание файла сравнения результатов
    comparison = {
        "map1": summary1,
        "map2": summary2,
        "occupied_fraction_ratio": summary1["occupied_fraction"] / summary2["occupied_fraction"]
            if summary2["occupied_fraction"] > 0 else "inf"
    }

    with open(os.path.join(outdir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print("\n=== Comparison summary ===")
    print("Occupied fraction map1:", summary1["occupied_fraction"])
    print("Occupied fraction map2:", summary2["occupied_fraction"])
    print("Ratio map1/map2:", comparison["occupied_fraction_ratio"])

    print("\nDetailed comparison saved to comparison.json")



# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Map quality evaluator (PGM) — OpenCV implementation")
    sub = ap.add_subparsers(dest="command")
    c1 = sub.add_parser("compare")
    c1.add_argument("map1")
    c1.add_argument("map2")
    c1.add_argument("--outdir", "-o", default="results")
    c1.add_argument("--threshold_method", choices=["mean", "otsu"], default="mean")
    c1.add_argument("--min_component_area", type=int, default=30)

    p_single = sub.add_parser("single", help="Process one map")
    p_single.add_argument("input", help="Map file")
    p_single.add_argument("--outdir", "-o", default="results")
    p_single.add_argument("--threshold_method", choices=["mean", "otsu"], default="mean")
    p_single.add_argument("--min_component_area", type=int, default=30)





    args = ap.parse_args()

    if args.command == "compare":
        compare_maps(args.map1, args.map2, args.outdir,
                     args.threshold_method, args.min_component_area)
    elif args.command == "single":
        process_map(args.input, args.outdir,
                    args.threshold_method, args.min_component_area)


if __name__ == "__main__":
    main()
