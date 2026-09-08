#!/usr/bin/env python3
"""Align a COLMAP SfM model onto the mocap camera centres (sim3, Umeyama).

Server-side stage 2 of the pose refinement (stage 1 = COLMAP CLI, see
refine_on_server.sh). Pure numpy — runs inside the 3dgs.sif container, no
pycolmap needed. Reads the SfM model in TXT form, aligns it to the mocap
model's camera centres (metric scale + insect frame from mocap, relative
geometry from SfM), and writes the refined COLMAP text model:

    python align_sfm_to_mocap.py <dataset> <sfm_txt_model_dir> [--out sparse_refined]

<dataset> must contain sparse/0 (the mocap model). Output:
<dataset>/<out>/0/{cameras,images,points3D}.txt — cameras copied unchanged
(intrinsics were fixed during SfM), poses refined, points3D = the real
triangulated cloud (use it as the GS init).
"""

import argparse
import os
import shutil

import numpy as np


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


def read_images_txt(path):
    """COLMAP images.txt → {name: (R_wc, t_wc)}. Handles the observations
    line that follows every pose line (possibly empty)."""
    out = {}
    # pose lines identified by content — a toggle-based parser drops every
    # second image when the observations lines are blank
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) != 10 or t[0].startswith('#'):
                continue
            if not t[9].lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            q = np.array([float(v) for v in t[1:5]])
            tv = np.array([float(v) for v in t[5:8]])
            out[t[9]] = (quat_wxyz_to_mat(q), tv)
    return out


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) * len(src) / (xs ** 2).sum()
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('sfm_model', help='SfM model dir with TXT files')
    ap.add_argument('--out', default='sparse_refined')
    args = ap.parse_args()

    ds = args.dataset.rstrip('/')
    mocap = read_images_txt(os.path.join(ds, 'sparse/0/images.txt'))
    sfm = read_images_txt(os.path.join(args.sfm_model, 'images.txt'))

    common = sorted(set(mocap) & set(sfm))
    if len(common) < 10:
        raise SystemExit(f'only {len(common)} common cameras — SfM failed?')
    sfm_c = np.array([-sfm[n][0].T @ sfm[n][1] for n in common])
    moc_c = np.array([-mocap[n][0].T @ mocap[n][1] for n in common])
    s, Ra, ta = umeyama(sfm_c, moc_c)
    resid = np.linalg.norm((s * (Ra @ sfm_c.T).T + ta) - moc_c, axis=1)
    print(f'{len(common)} cameras aligned; sim3 scale {s:.4f}')
    print(f'centre residual vs mocap: p50 {np.percentile(resid, 50)*1000:.1f} '
          f'mm, p90 {np.percentile(resid, 90)*1000:.1f} mm, '
          f'max {resid.max()*1000:.1f} mm  (= mocap pose error audit)')

    out_dir = os.path.join(ds, args.out, '0')
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(os.path.join(ds, 'sparse/0/cameras.txt'),
                 os.path.join(out_dir, 'cameras.txt'))

    corrections = []
    with open(os.path.join(out_dir, 'images.txt'), 'w') as f:
        f.write('# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n')
        for k, name in enumerate(common, 1):
            R_wc, t_wc = sfm[name]
            C = -R_wc.T @ t_wc
            Cn = s * Ra @ C + ta
            Rn = R_wc @ Ra.T
            tn = -Rn @ Cn
            q = mat_to_quat_wxyz(Rn)
            f.write(f'{k} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} '
                    f'{tn[0]:.9f} {tn[1]:.9f} {tn[2]:.9f} 1 {name}\n\n')
            Rm, tm = mocap[name]
            corrections.append(
                (np.linalg.norm(Cn - (-Rm.T @ tm)) * 1000,
                 np.degrees(np.arccos(np.clip(
                     (np.trace(Rn @ Rm.T) - 1) / 2, -1, 1)))))

    n_pts = 0
    with open(os.path.join(args.sfm_model, 'points3D.txt')) as fin, \
            open(os.path.join(out_dir, 'points3D.txt'), 'w') as fout:
        fout.write('# POINT3D_ID X Y Z R G B ERROR TRACK[]\n')
        for line in fin:
            if line.startswith('#') or not line.strip():
                continue
            t = line.split()
            p = s * Ra @ np.array([float(v) for v in t[1:4]]) + ta
            fout.write(f'{t[0]} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} '
                       f'{t[4]} {t[5]} {t[6]} {t[7]}\n')
            n_pts += 1

    corr = np.array(corrections)
    print(f'{n_pts} triangulated points transformed')
    print(f'per-camera correction: position p50 '
          f'{np.percentile(corr[:, 0], 50):.1f} mm max {corr[:, 0].max():.1f} '
          f'mm; rotation p50 {np.percentile(corr[:, 1], 50):.2f} deg '
          f'max {corr[:, 1].max():.2f} deg')
    missing = sorted(set(mocap) - set(sfm))
    if missing:
        print(f'WARNING: {len(missing)} images not registered by SfM, '
              f'excluded from the refined model: {missing}')
    print(f'-> {out_dir}')


if __name__ == '__main__':
    main()
