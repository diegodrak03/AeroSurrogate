import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================
#  NACA 4-DIGIT PARAMETERISATION
#
#  Three coefficients map to standard NACA parameters:
#
#  t = c1 + c2        maximum thickness (fraction of chord)
#                     physical range: (0, 0.4]
#
#  m = (c1 - c2) / 2  maximum camber
#                     physical range: [0, 0.09)  (m=0 → symmetric)
#
#  p = c3             chordwise position of maximum camber
#                     physical range: [0.1, 0.9]
#
#  Equivalent NACA 4-digit notation: NACA [m*100] [p*10] [t*100]
# ============================================================


def generate_profile(c1, c2, c3, alpha_deg, n_points=200):
    """
    Generate upper and lower surface coordinates of a NACA 4-digit airfoil.

    Parameters
    ----------
    c1, c2, c3 : float   Geometry coefficients (see module header).
    alpha_deg  : float   Angle of attack in degrees.
    n_points   : int     Number of chordwise stations (cosine spacing).

    Returns
    -------
    x_up, y_up, x_lo, y_lo : ndarray  Surface coordinates in chord units.
    """
    t = c1 + c2
    m = (c1 - c2) / 2
    p = c3

    if t <= 0 or t > 0.4:
        raise ValueError(f"Thickness t={t:.3f} outside physical range (0, 0.4].")
    if m > 0 and not (0.1 <= p <= 0.9):
        raise ValueError(f"Camber position p={p} outside range [0.1, 0.9].")

    alpha_rad = np.radians(alpha_deg)
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))

    # NACA thickness distribution polynomial
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1015 * x**4)

    # Camber line: two-segment parabolic arc
    yc = np.zeros(n_points)
    for i in range(n_points):
        if x[i] <= p:
            if p > 0:
                yc[i] = (m / p**2) * (2 * p * x[i] - x[i]**2)
        else:
            if p < 1:
                yc[i] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[i] - x[i]**2)

    cos_a, sin_a = np.cos(alpha_rad), np.sin(alpha_rad)
    x_up =  x * cos_a + (yc + yt) * sin_a
    y_up = -x * sin_a + (yc + yt) * cos_a
    x_lo =  x * cos_a + (yc - yt) * sin_a
    y_lo = -x * sin_a + (yc - yt) * cos_a

    return x_up, y_up, x_lo, y_lo


# ============================================================
#  STANDALONE PREVIEW
# ============================================================
C1 = 0.20
C2 = 0.00
C3 = 0.40
ALPHA_DEG = 0.0

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    x_up, y_up, x_lo, y_lo = generate_profile(C1, C2, C3, ALPHA_DEG)

    txt_path = os.path.join(out_dir, "profile_points.txt")
    with open(txt_path, "w") as f:
        f.write(f"# NACA profile  c1={C1}  c2={C2}  c3={C3}  alpha={ALPHA_DEG} deg\n")
        f.write("# x  y  side (1=upper, -1=lower)\n")
        for xi, yi in zip(x_up, y_up):
            f.write(f"{xi:.6f}  {yi:.6f}  1\n")
        for xi, yi in zip(x_lo, y_lo):
            f.write(f"{xi:.6f}  {yi:.6f}  -1\n")
    print(f"-> profile_points.txt  ({len(x_up) * 2} points)")

    png_path = os.path.join(out_dir, "profile_preview.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_up, y_up, 'b-', linewidth=2, label='Upper surface')
    ax.plot(x_lo, y_lo, 'r-', linewidth=2, label='Lower surface')
    ax.fill(np.concatenate([x_up, x_lo[::-1]]),
            np.concatenate([y_up, y_lo[::-1]]),
            alpha=0.15, color='steelblue')
    ax.set_aspect('equal')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_title(f'NACA  c1={C1}  c2={C2}  c3={C3}  alpha={ALPHA_DEG} deg')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print("-> profile_preview.png")
