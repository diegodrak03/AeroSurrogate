import numpy as np
import subprocess
import os
import shutil
import itertools
import re
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path as MplPath

# ============================================================
#  PARAMETER SPACE
# ============================================================
C1_VALUES    = [0.05, 0.10, 0.20]
C2_VALUES    = [-0.03, 0.0, 0.04, 0.08, 0.10]
C3_VALUES    = [0.2, 0.4, 0.5]
U_VALUES     = [15.0, 45.0, 80.0, 100.0]
ALPHA_VALUES = [-8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 16.0]
CHORD_VALUES = [1.0, 2.0, 3.0]

DATASET_DIR = Path(__file__).parent / "dataset"
TEMP_DIR    = Path(__file__).parent / "_tmp"

# Mesh resolution near the airfoil (fraction of chord). A mesh convergence
# study (symmetric airfoil C1=C2=0.1, alpha=0, where CL must be exactly 0)
# showed that with lc_airfoil=0.01 (6742 cells) the Gmsh unstructured mesh is
# not symmetric (only 2.9% of the points have a mirror counterpart) and that
# asymmetry produces a spurious CL of up to ~0.11 -- a bias of the same
# magnitude also appears on cambered airfoils. Refining to 0.005 (~9600 cells,
# ~1.4x more expensive) reduces the bias to ~0.03-0.05.
LC_AIRFOIL = 0.005

# ============================================================
#  GEOMETRY
# ============================================================

def build_naca_geo(c1, c2, c3, alpha_deg, lc_airfoil, n_points=100, chord=1.0):
    alpha_rad = np.radians(alpha_deg)
    t = c1 + c2
    m = (c1 - c2) / 2
    p = c3

    if t <= 0 or t > 0.4:
        raise ValueError(f"Thickness t={t:.3f} outside physical range (0, 0.4].")
    if m > 0 and not (0.1 <= p <= 0.9):
        raise ValueError(f"Camber position p={p} outside range [0.1, 0.9].")

    lc_airfoil = lc_airfoil * chord  # maintain relative refinement regardless of chord scale

    beta   = np.linspace(0, np.pi, n_points)
    x_norm = 0.5 * (1 - np.cos(beta))           # normalized chord fraction ∈ [0, 1]
    x      = x_norm * chord                       # physical coordinates [m]

    # NACA thickness and camber formulas use normalized x; output scaled by chord
    yt = chord * 5 * t * (0.2969 * np.sqrt(x_norm) - 0.1260 * x_norm - 0.3516 * x_norm**2
                          + 0.2843 * x_norm**3 - 0.1015 * x_norm**4)

    yc = np.zeros(n_points)
    for i in range(n_points):
        if x_norm[i] <= p:
            if p > 0:
                yc[i] = chord * (m / p**2) * (2 * p * x_norm[i] - x_norm[i]**2)
        else:
            if p < 1:
                yc[i] = chord * (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x_norm[i] - x_norm[i]**2)

    x_up, y_up = x, yc + yt
    x_lo, y_lo = x, yc - yt

    def rotate(px, py):
        rx =  px * np.cos(alpha_rad) + py * np.sin(alpha_rad)
        ry = -px * np.sin(alpha_rad) + py * np.cos(alpha_rad)
        return rx, ry

    lines = [f"lc_ala = {lc_airfoil};", "Point(1) = {0, 0, 0, lc_ala};"]
    pid = 2
    for i in range(1, n_points):
        rx, ry = rotate(x_up[i], y_up[i])
        lines.append(f"Point({pid}) = {{{rx:.6f}, {ry:.6f}, 0, lc_ala}};")
        pid += 1
    for i in range(n_points - 2, 0, -1):
        rx, ry = rotate(x_lo[i], y_lo[i])
        lines.append(f"Point({pid}) = {{{rx:.6f}, {ry:.6f}, 0, lc_ala}};")
        pid += 1

    ids_upper = list(range(1, n_points + 1))
    lines.append(f"BSpline(1) = {{{', '.join(map(str, ids_upper))}}};")
    ids_lower = [n_points] + list(range(n_points + 1, pid)) + [1]
    lines.append(f"BSpline(2) = {{{', '.join(map(str, ids_lower))}}};")

    return "\n".join(lines)


def generate_mesh(airfoil_geo, work_dir, name="profile", chord=1.0):
    c = chord
    lines = [airfoil_geo, "\n// --- External domain ---"]
    lines += [
        f"lc_ext = {0.5 * c};",
        f"Point(500) = {{{-5*c}, {-5*c}, 0, lc_ext}}; Point(501) = {{{10*c}, {-5*c}, 0, lc_ext}};",
        f"Point(502) = {{{10*c}, {5*c}, 0, lc_ext}};  Point(503) = {{{-5*c}, {5*c}, 0, lc_ext}};",
        "L_suelo  = 3; Line(L_suelo)  = {500, 501};",
        "L_outlet = 4; Line(L_outlet) = {501, 502};",
        "L_techo  = 5; Line(L_techo)  = {502, 503};",
        "L_inlet  = 6; Line(L_inlet)  = {503, 500};",
        "Curve Loop(1) = {L_inlet, L_suelo, L_outlet, L_techo};",
        "Curve Loop(2) = {1, 2};",
        "Plane Surface(1) = {1, 2};",
        "v[] = Extrude {0, 0, 0.1} { Surface{1}; Layers{1}; Recombine; };",
        "\n// --- Physical groups ---",
        'Physical Surface("frontAndBack") = {1, v[0]};',
        'Physical Surface("inlet")        = {v[2]};',
        'Physical Surface("outlet")       = {v[4]};',
        'Physical Surface("walls")        = {v[3], v[5]};',
        'Physical Surface("ala")          = {v[6], v[7]};',
        'Physical Volume("internal")      = {v[1]};',
    ]
    geo_path = os.path.join(work_dir, f"{name}.geo")
    msh_path = os.path.join(work_dir, f"{name}.msh")
    with open(geo_path, "w") as f:
        f.write("\n".join(lines))
    subprocess.run(
        ["gmsh", "-3", geo_path, "-o", msh_path, "-format", "msh2"],
        check=True, capture_output=True
    )
    return msh_path


def setup_openfoam_case(case_dir, U_mag, alpha_deg, c=1.0, span=0.1):
    # Turbulence inlet conditions: 5% intensity, length scale = 7% chord
    I   = 0.05
    L   = c * 0.07
    nu  = 1.5e-5   # kinematic viscosity of air [m2/s]
    rho = 1.225    # air density at sea level [kg/m3]

    k     = 1.5 * (U_mag * I)**2
    omega = (k**0.5) / (0.09**0.25 * L)
    nut   = k / omega

    # Wind-frame axes for forceCoeffs (alpha-independent):
    # e1=(1,0,0) → Cd = aerodynamic drag, e2=(0,0,-1) → Cs≈0 (2D),
    # e3=e1×e2=(0,1,0) → Cl = aerodynamic lift
    e1_wind = "(1 0 0)"
    e2_wind = "(0 0 -1)"

    for folder in ('0', 'constant', 'system'):
        os.makedirs(os.path.join(case_dir, folder), exist_ok=True)

    def header(cls, obj, loc):
        return (
            "/*--------------------------------*- C++ -*----------------------------------*\\\n"
            "| =========                 |                                                 |\n"
            "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
            "|  \\\\    /   O peration     | Version:  2406                                  |\n"
            "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n"
            "|    \\\\/     M anipulation  |                                                 |\n"
            "\\*---------------------------------------------------------------------------*/\n"
            "FoamFile\n{\n"
            f"    version     2.0;\n    format      ascii;\n"
            f"    class       {cls};\n    location    \"{loc}\";\n"
            f"    object      {obj};\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
        )

    bc_fields = {
        "U": (header("volVectorField", "U", "0") +
              f"dimensions [0 1 -1 0 0 0 0];\ninternalField uniform ({U_mag} 0 0);\n"
              f"boundaryField {{ inlet {{ type fixedValue; value uniform ({U_mag} 0 0); }} "
              f"outlet {{ type zeroGradient; }} ala {{ type noSlip; }} walls {{ type slip; }} "
              f"frontAndBack {{ type empty; }} }}"),
        "p": (header("volScalarField", "p", "0") +
              "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\n"
              "boundaryField { inlet { type zeroGradient; } outlet { type fixedValue; value uniform 0; } "
              "ala { type zeroGradient; } walls { type zeroGradient; } frontAndBack { type empty; } }"),
        "k": (header("volScalarField", "k", "0") +
              f"dimensions [0 2 -2 0 0 0 0];\ninternalField uniform {k};\n"
              f"boundaryField {{ inlet {{ type fixedValue; value uniform {k}; }} outlet {{ type zeroGradient; }} "
              f"ala {{ type kqRWallFunction; value uniform {k}; }} walls {{ type zeroGradient; }} "
              f"frontAndBack {{ type empty; }} }}"),
        "omega": (header("volScalarField", "omega", "0") +
                  f"dimensions [0 0 -1 0 0 0 0];\ninternalField uniform {omega};\n"
                  f"boundaryField {{ inlet {{ type fixedValue; value uniform {omega}; }} outlet {{ type zeroGradient; }} "
                  f"ala {{ type omegaWallFunction; value uniform {omega}; }} walls {{ type zeroGradient; }} "
                  f"frontAndBack {{ type empty; }} }}"),
        "nut": (header("volScalarField", "nut", "0") +
                f"dimensions [0 2 -1 0 0 0 0];\ninternalField uniform {nut};\n"
                f"boundaryField {{ inlet {{ type calculated; value uniform {nut}; }} outlet {{ type calculated; value uniform {nut}; }} "
                f"ala {{ type nutkWallFunction; value uniform {nut}; }} walls {{ type calculated; value uniform {nut}; }} "
                f"frontAndBack {{ type empty; }} }}"),
    }
    for name, content in bc_fields.items():
        with open(os.path.join(case_dir, "0", name), "w") as f:
            f.write(content)

    with open(os.path.join(case_dir, "constant", "transportProperties"), "w") as f:
        f.write(header("dictionary", "transportProperties", "constant") +
                f"transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {nu};")

    with open(os.path.join(case_dir, "constant", "turbulenceProperties"), "w") as f:
        f.write(header("dictionary", "turbulenceProperties", "constant") +
                "simulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }")

    # controlDict with forceCoeffs function object
    control = (
        f"application     simpleFoam;\n"
        f"startFrom       startTime;\nstartTime       0;\n"
        f"stopAt          endTime;\nendTime         1000;\ndeltaT          1;\n"
        f"writeControl    timeStep;\nwriteInterval   200;\npurgeWrite      2;\n"
        f"writeFormat     ascii;\nwritePrecision  6;\nrunTimeModifiable true;\n\n"
        f"functions\n{{\n    forces\n    {{\n"
        f"        type            forceCoeffs;\n        libs            (\"libforces.so\");\n"
        f"        writeControl    timeStep;\n        writeInterval   1;\n"
        f"        patches         (ala);\n        rho             rhoInf;\n        rhoInf          {rho};\n"
        f"        pitchAxis       (0 0 1);\n        magUInf         {U_mag};\n"
        f"        lRef            {c};\n        Aref            {c * span};\n"
        f"\n        coordinateSystem\n        {{\n"
        f"            type    cartesian;\n            origin  (0 0 0);\n"
        f"            e1      {e1_wind};\n            e2      {e2_wind};\n"
        f"        }}\n    }}\n}}"
    )
    with open(os.path.join(case_dir, "system", "controlDict"), "w") as f:
        f.write(header("dictionary", "controlDict", "system") + control)

    with open(os.path.join(case_dir, "system", "fvSchemes"), "w") as f:
        f.write(header("dictionary", "fvSchemes", "system") +
                "ddtSchemes { default steadyState; }\n"
                "gradSchemes { default Gauss linear; }\n"
                "divSchemes {\n"
                "    default none;\n"
                "    div(phi,U)      bounded Gauss upwind;\n"
                "    div(phi,k)      bounded Gauss upwind;\n"
                "    div(phi,omega)  bounded Gauss upwind;\n"
                "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n"
                "}\n"
                "laplacianSchemes { default Gauss linear corrected; }\n"
                "interpolationSchemes { default linear; }\n"
                "snGradSchemes { default corrected; }\n"
                "wallDist { method meshWave; }\n")

    with open(os.path.join(case_dir, "system", "fvSolution"), "w") as f:
        f.write(header("dictionary", "fvSolution", "system") +
                'solvers { p { solver GAMG; tolerance 1e-6; relTol 0.01; smoother GaussSeidel; } '
                '"(U|k|omega)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-8; relTol 0.1; } } '
                'SIMPLE { nNonOrthogonalCorrectors 0; consistent yes; } '
                'relaxationFactors { fields { p 0.3; } equations { U 0.7; k 0.7; omega 0.7; } }')


def convert_mesh(msh_path, case_dir):
    subprocess.run(["gmshToFoam", os.path.basename(msh_path)], cwd=case_dir,
                   check=True, capture_output=True)
    boundary_path = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    if os.path.exists(boundary_path):
        with open(boundary_path) as f:
            content = f.read()
        # gmshToFoam assigns generic patch type; override with correct types
        content = content.replace(
            'ala\n    {\n        type            patch;',
            'ala\n    {\n        type            wall;')
        content = content.replace(
            'frontAndBack\n    {\n        type            patch;',
            'frontAndBack\n    {\n        type            empty;')
        with open(boundary_path, "w") as f:
            f.write(content)
    subprocess.run(["checkMesh"], cwd=case_dir, check=True, capture_output=True)


# ============================================================
#  POST-PROCESSING
# ============================================================

def read_foam_field(path):
    with open(path) as f:
        text = f.read()
    match = re.search(
        r'internalField\s+nonuniform\s+List<\w+>\s*\d+\s*\((.+?)\)\s*;',
        text, re.DOTALL
    )
    if not match:
        raise ValueError(f"Could not read internalField in {path}")
    block = match.group(1)
    values = list(map(float, block.replace('(', ' ').replace(')', ' ').split()))
    return np.array(values).reshape(-1, 3) if '(' in block else np.array(values)


def naca_coords(c1, c2, c3, alpha_deg, n_points=100):
    alpha_rad = np.radians(alpha_deg)
    t = c1 + c2
    m = (c1 - c2) / 2
    p = c3

    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1015 * x**4)

    yc = np.zeros(n_points)
    for i in range(n_points):
        if x[i] <= p:
            if p > 0:
                yc[i] = (m / p**2) * (2 * p * x[i] - x[i]**2)
        else:
            if p < 1:
                yc[i] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[i] - x[i]**2)

    cos_a, sin_a = np.cos(alpha_rad), np.sin(alpha_rad)
    return (x * cos_a + (yc + yt) * sin_a, -x * sin_a + (yc + yt) * cos_a,
            x * cos_a + (yc - yt) * sin_a, -x * sin_a + (yc - yt) * cos_a)


def postprocess(case_dir, c1, c2, c3, U_mag, alpha_deg, chord=1.0):
    case_name    = f"Sim_C1_{c1}_C2_{c2}_C3_{c3}_U_{U_mag}_Alpha_{alpha_deg}_Chord_{chord}"
    case_root    = DATASET_DIR / case_name
    dir_contours = case_root / "contours"
    dir_openfoam = case_root / "openfoam"

    for d in (case_root, dir_contours, dir_openfoam):
        d.mkdir(parents=True, exist_ok=True)

    # Locate last converged time directory
    time_dirs = []
    for entry in os.listdir(case_dir):
        try:
            t = float(entry)
            if t > 0:
                time_dirs.append(t)
        except ValueError:
            pass
    if not time_dirs:
        raise RuntimeError("No time directories found in case.")
    last_t = str(int(max(time_dirs)))
    t_path = os.path.join(case_dir, last_t)

    subprocess.run(
        ["postProcess", "-func", "writeCellCentres", "-time", last_t],
        cwd=case_dir, check=True, capture_output=True
    )
    centres     = read_foam_field(os.path.join(t_path, "C"))
    cx_dim, cy_dim = centres[:, 0], centres[:, 1]
    # Normalize to unit-chord coordinates so the field network always sees x/c, y/c ∈ same range
    cx = cx_dim / chord
    cy = cy_dim / chord
    p_field     = read_foam_field(os.path.join(t_path, "p"))
    U_field     = read_foam_field(os.path.join(t_path, "U"))
    k_field     = read_foam_field(os.path.join(t_path, "k"))
    omega_field = read_foam_field(os.path.join(t_path, "omega"))
    Ux, Uy      = U_field[:, 0], U_field[:, 1]

    # Store x, y already normalized to x/c, y/c; adimensionalisation of flow quantities in build_dataset.py
    cells_csv = case_root / "cell_data.csv"
    with open(cells_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "p", "Ux", "Uy", "k", "omega"])
        for i in range(len(cx)):
            w.writerow([f"{cx[i]:.6f}", f"{cy[i]:.6f}", f"{p_field[i]:.6f}",
                        f"{Ux[i]:.6f}", f"{Uy[i]:.6f}",
                        f"{k_field[i]:.6f}", f"{omega_field[i]:.6f}"])
    print(f"    -> cell_data.csv ({len(cx)} cells)")

    # naca_coords returns unit-chord coordinates; cx, cy are also unit-chord after normalization
    x_up, y_up, x_lo, y_lo = naca_coords(c1, c2, c3, alpha_deg)
    mask = (cx > -0.5) & (cx < 2.0) & (cy > -0.8) & (cy < 0.8)
    tri  = mtri.Triangulation(cx[mask], cy[mask])

    contour_x = np.concatenate([x_up, x_lo[::-1], [x_up[0]]])
    contour_y = np.concatenate([y_up, y_lo[::-1], [y_up[0]]])
    airfoil_path = MplPath(np.column_stack([contour_x, contour_y]))
    xv, yv = tri.x, tri.y
    cx_tri = (xv[tri.triangles[:, 0]] + xv[tri.triangles[:, 1]] + xv[tri.triangles[:, 2]]) / 3
    cy_tri = (yv[tri.triangles[:, 0]] + yv[tri.triangles[:, 1]] + yv[tri.triangles[:, 2]]) / 3
    tri.set_mask(airfoil_path.contains_points(np.column_stack([cx_tri, cy_tri])))

    for data, cmap, label, fname in [
        (p_field[mask],                          'RdBu_r',  'p [m2/s2]', 'contour_pressure.png'),
        (np.sqrt(Ux[mask]**2 + Uy[mask]**2),     'viridis', '|U| [m/s]', 'contour_velocity.png'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        cf = ax.tricontourf(tri, data, levels=50, cmap=cmap)
        plt.colorbar(cf, ax=ax, label=label)
        ax.fill(contour_x, contour_y, color=ax.get_facecolor(), zorder=2)
        ax.plot(x_up, y_up, 'k-', linewidth=1.5, zorder=3)
        ax.plot(x_lo, y_lo, 'k-', linewidth=1.5, zorder=3)
        ax.set_xlim(-0.5, 2.0); ax.set_ylim(-0.8, 0.8)
        ax.set_aspect('equal')
        ax.set_xlabel('x/c'); ax.set_ylabel('y/c')
        ax.set_title(f'{label}  U={U_mag} m/s  alpha={alpha_deg} deg  c={chord} m')
        fig.tight_layout()
        fig.savefig(dir_contours / fname, dpi=150)
        plt.close(fig)

    Re  = U_mag * chord / 1.5e-5
    I   = 0.05
    L   = chord * 0.07
    k0  = 1.5 * (U_mag * I)**2
    om0 = (k0**0.5) / (0.09**0.25 * L)
    with open(dir_contours / "simulation_summary.txt", "w") as f:
        f.write(f"U={U_mag} m/s  chord={chord} m  alpha={alpha_deg} deg  Re={Re:.0f}\n")
        f.write(f"c1={c1}  c2={c2}  c3={c3}\n")
        f.write(f"k_inlet={k0:.6f} m2/s2  omega_inlet={om0:.4f} 1/s\n")
        f.write(f"cells={len(cx)}\n")

    foam_items = ('0', 'constant', 'system', 'postProcessing')
    for item in os.listdir(case_dir):
        src = os.path.join(case_dir, item)
        is_time = os.path.isdir(src) and item.replace('.', '', 1).isdigit()
        if item in foam_items or item.startswith('log.') or is_time:
            shutil.move(src, dir_openfoam / item)
    print(f"    -> archived to dataset/{case_name}/")


# ============================================================
#  SIMULATION LOOP
# ============================================================

def clean_tmp():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True)


def log_failure(params, error):
    log_path = DATASET_DIR / "failures.log"
    with open(log_path, "a") as f:
        f.write(f"FAIL | {params} | {error}\n")
    print(f"  [FAIL] logged to failures.log")


FORCES_COLS = ['Time', 'Cd', 'Cd_f', 'Cd_r', 'Cl', 'Cl_f', 'Cl_r',
               'CmPitch', 'CmRoll', 'CmYaw', 'Cs', 'Cs_f', 'Cs_r']

def check_convergence(case_dir, threshold=0.05):
    """Verify force coefficients converged (std/mean of last 50 Cd and Cl < threshold).

    With fixed wind-frame axes (see setup_openfoam_case), Cd and Cl are
    already the aerodynamic drag and lift -- both channels are checked
    (previously only Cd was checked).
    """
    import pandas as pd
    coeff_path = os.path.join(case_dir, "postProcessing", "forces", "0", "coefficient.dat")
    df = pd.read_csv(coeff_path, sep=r'\s+', comment='#', names=FORCES_COLS)
    tail = df.tail(50)
    for col in ('Cd', 'Cl'):
        mean = abs(tail[col].mean())
        if mean < 1e-10:
            continue
        rel_std = tail[col].std() / mean
        if rel_std > threshold:
            raise RuntimeError(f"Not converged: {col} relative std = {rel_std:.3f} > {threshold}")
    return True


def run_case(c1, c2, c3, U_mag, alpha_deg, chord=1.0):
    case_name = f"Sim_C1_{c1}_C2_{c2}_C3_{c3}_U_{U_mag}_Alpha_{alpha_deg}_Chord_{chord}"
    if (DATASET_DIR / case_name).exists():
        print(f"  [SKIP] {case_name}")
        return

    clean_tmp()
    airfoil_geo = build_naca_geo(c1=c1, c2=c2, c3=c3, alpha_deg=alpha_deg, lc_airfoil=LC_AIRFOIL, chord=chord)
    msh_path    = generate_mesh(airfoil_geo, str(TEMP_DIR), "profile", chord=chord)
    setup_openfoam_case(str(TEMP_DIR), U_mag=U_mag, alpha_deg=alpha_deg, c=chord)
    convert_mesh(msh_path, str(TEMP_DIR))

    log_path = TEMP_DIR / "log.simpleFoam"
    with open(log_path, "w") as log_file:
        subprocess.run(["simpleFoam"], cwd=str(TEMP_DIR),
                       stdout=log_file, stderr=subprocess.STDOUT, check=True)

    check_convergence(str(TEMP_DIR))
    postprocess(str(TEMP_DIR), c1=c1, c2=c2, c3=c3, U_mag=U_mag, alpha_deg=alpha_deg, chord=chord)


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    combinations = list(itertools.product(C1_VALUES, C2_VALUES, C3_VALUES, U_VALUES, ALPHA_VALUES, CHORD_VALUES))
    total = len(combinations)
    print(f"Total cases: {total}")
    print(f"Output: {DATASET_DIR}\n")

    for idx, (c1, c2, c3, U, alpha, chord) in enumerate(combinations, start=1):
        print(f"\n[{idx}/{total}] C1={c1} C2={c2} C3={c3} U={U} alpha={alpha} chord={chord}")
        try:
            run_case(c1, c2, c3, U, alpha, chord=chord)
        except Exception as e:
            log_failure(f"c1={c1} c2={c2} c3={c3} U={U} alpha={alpha} chord={chord}", str(e))

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    print(f"\nCompleted. Dataset at: {DATASET_DIR}")
