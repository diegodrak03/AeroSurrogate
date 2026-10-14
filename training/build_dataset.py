import os
import glob
import pandas as pd
from pathlib import Path

NU = 1.5e-5  # kinematic viscosity of air [m2/s]

DATASET_DIR  = Path(__file__).parent.parent / "simulations" / "dataset"
OUTPUT_DIR   = Path(__file__).parent

FORCES_COLS = ['Time', 'Cd', 'Cd_f', 'Cd_r', 'Cl', 'Cl_f', 'Cl_r',
               'CmPitch', 'CmRoll', 'CmYaw', 'Cs', 'Cs_f', 'Cs_r']

case_dirs = glob.glob(str(DATASET_DIR / "*"))
print(f"Detected {len(case_dirs)} simulation directories.")

CONV_THRESHOLD = 0.05  # max relative std over last 50 iterations

forces_rows   = []
fields_frames = []
n_skipped     = 0

for i, case_dir in enumerate(case_dirs):
    case_name = os.path.basename(case_dir)
    parts = case_name.split('_')

    if len(parts) < 13:
        continue

    c1    = float(parts[2])
    c2    = float(parts[4])
    c3    = float(parts[6])
    U     = float(parts[8])
    alpha = float(parts[10])
    chord = float(parts[12])
    Re    = (U * chord) / NU

    try:
        forces_path = os.path.join(case_dir, "openfoam", "postProcessing", "forces", "0", "coefficient.dat")
        cells_path  = os.path.join(case_dir, "cell_data.csv")

        df_forces = pd.read_csv(forces_path, sep=r'\s+', comment='#', names=FORCES_COLS)

        # Convergence filter: reject cases where Cd or Cl still oscillate
        tail50 = df_forces.tail(50)
        cd_mean = abs(tail50['Cd'].mean())
        cl_mean = abs(tail50['Cl'].mean())
        cd_rel = tail50['Cd'].std() / cd_mean if cd_mean > 1e-10 else 0
        cl_rel = tail50['Cl'].std() / cl_mean if cl_mean > 1e-10 else 0
        if cd_rel > CONV_THRESHOLD or cl_rel > CONV_THRESHOLD:
            n_skipped += 1
            continue

        last = df_forces.tail(10).mean()

        # OpenFOAM forceCoeffs se configura con ejes wind-frame fijos
        # (e1=(1,0,0), e2=(0,0,-1); ver simulations/pipeline.py::setup_openfoam_case).
        # El flujo de entrada siempre apunta en +x -- es el perfil el que rota
        # con alpha, no el flujo -- asi que Cd y Cl de OpenFOAM ya son
        # directamente drag y lift aerodinamicos, sin necesitar ninguna rotacion.
        CL_aero = last['Cl']
        CD_aero = last['Cd']
        forces_rows.append([c1, c2, c3, Re, alpha, CL_aero, CD_aero])

        df_cells = pd.read_csv(cells_path)

        # Adimensionalise velocity components by inlet speed U
        df_cells['Ux_adim'] = df_cells['Ux'] / U
        df_cells['Uy_adim'] = df_cells['Uy'] / U

        # Pressure coefficient: Cp = p_kinematic / (0.5 * U^2)
        # p in OpenFOAM is kinematic pressure [m2/s2]
        df_cells['Cp'] = df_cells['p'] / (0.5 * U**2)

        df_cells.drop(columns=['Ux', 'Uy', 'p'], inplace=True)

        df_cells['C1']    = c1
        df_cells['C2']    = c2
        df_cells['C3']    = c3
        df_cells['Re']    = Re
        df_cells['Alpha'] = alpha

        fields_frames.append(df_cells)

    except Exception as e:
        print(f"[ERROR] {case_name}: {e}")
        continue

    if i % 50 == 0:
        print(f"  Processed {i}/{len(case_dirs)} ...")

print(f"\nConvergence filter: {n_skipped} cases rejected (rel_std > {CONV_THRESHOLD})")

print("\nBuilding output files ...")

df_forces = pd.DataFrame(forces_rows, columns=['C1', 'C2', 'C3', 'Re', 'Alpha', 'CL', 'CD'])
df_forces.to_csv(OUTPUT_DIR / "dataset_forces.csv", index=False)
print(f"dataset_forces.csv  ({len(df_forces)} rows)")

print("Concatenating fields dataset (may take a moment) ...")
df_fields = pd.concat(fields_frames, ignore_index=True)
df_fields.to_csv(OUTPUT_DIR / "dataset_fields.csv", index=False)
print(f"dataset_fields.csv  ({len(df_fields)} rows)")
