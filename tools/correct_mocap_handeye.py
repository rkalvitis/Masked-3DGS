#!/usr/bin/env python3
"""Apply the fitted hand-eye correction to the mocap COLMAP model.

Root cause (popillia 48img_20_08): the rig-anchored export used a wrong
marker->camera (hand-eye) transform — off by ~4.4 deg and ~68 mm (mostly
along the optical axis). Every camera pose carries the same camera-frame
error, so no 3D point satisfied all 48 silhouettes (empty visual hull).

The correction below was fit by aligning the projected specimen centre to
the 48 mask centroids (scipy least squares), then verified: with it, the
visual hull of all 48 masks is non-empty at ZERO dilation (11x21x14 mm,
beetle-sized). Corrected pose: R' = dR @ R,  t' = dR @ t + dt.

    python correct_mocap_handeye.py <dataset> [--out sparse_corrected]

Reads <dataset>/sparse_mocap/0, writes <dataset>/<out>/0 with all 48
cameras and an empty points3D.txt (triangulate afterwards with the
prior+tri stages of refine_on_server.sh).
"""

import argparse
import os

import numpy as np

# fitted 2026-08-25 on 48img_20_08_masked (mask-centroid fit, hull-verified)
DR_ROTVEC = np.array([0.03849, -0.05150, 0.04273])   # 4.42 deg total
DT = np.array([-0.10031, 0.20165, 0.63968])          # units (1 = 100 mm)


def rodrigues(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def quat_wxyz_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def mat_to_quat_wxyz(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-8:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
        v = [0.0, 0.0, 0.0]
        v[i] = s / 4
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = v
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--out', default='sparse_corrected')
    args = ap.parse_args()

    ds = args.dataset.rstrip('/')
    src = os.path.join(ds, 'sparse_mocap/0')
    out = os.path.join(ds, args.out, '0')
    os.makedirs(out, exist_ok=True)

    dR = rodrigues(DR_ROTVEC)

    import shutil
    shutil.copy2(os.path.join(src, 'cameras.txt'),
                 os.path.join(out, 'cameras.txt'))
    open(os.path.join(out, 'points3D.txt'), 'w').close()

    n = 0
    with open(os.path.join(src, 'images.txt')) as fin, \
            open(os.path.join(out, 'images.txt'), 'w') as fout:
        fout.write('# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n')
        # NOTE: identify pose lines by content, NOT by alternation — the
        # observations line after each pose may be blank, and a skip-blanks
        # + toggle parser silently drops every second image (this exact bug
        # made every earlier model odd-images-only).
        for line in fin:
            t = line.split()
            if len(t) != 10 or t[0].startswith('#') \
                    or not t[9].lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            R = quat_wxyz_to_mat(np.array(list(map(float, t[1:5]))))
            tv = np.array(list(map(float, t[5:8])))
            Rn, tn = dR @ R, dR @ tv + DT
            q = mat_to_quat_wxyz(Rn)
            n += 1
            fout.write(f'{n} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} '
                       f'{q[3]:.9f} {tn[0]:.9f} {tn[1]:.9f} '
                       f'{tn[2]:.9f} {t[8]} {t[9]}\n\n')
    print(f'{n} cameras corrected -> {out}')


if __name__ == '__main__':
    main()
