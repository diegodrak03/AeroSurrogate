import streamlit as st
import numpy as np
import torch
import joblib
import sys
import plotly.graph_objects as go
from pathlib import Path
from matplotlib.path import Path as MplPath

ENABLE_CL_TARGET = 0      # 1 = show the "Alcanzar un CL objetivo" (target CL) option
ENABLE_CL_MIN    = 0      # 1 = show the "Imponer CL mínimo" (minimum CL) constraint

# Physical constants (SI)
NU_AIR         = 1.5e-5   # kinematic viscosity [m2/s]
RHO_AIR        = 1.225    # air density at sea level [kg/m3]
SPEED_OF_SOUND = 340.3    # speed of sound, ISA sea level [m/s]
MACH_WARN      = 0.3      # approx. incompressible-flow threshold (simpleFoam assumption)

# Path setup
APP_DIR    = Path(__file__).resolve().parent
ROOT_DIR   = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "training"

# Dataset ranges (read from CSV so limits auto-update when dataset grows)
import pandas as pd
_df_ranges = pd.read_csv(MODELS_DIR / "dataset_forces.csv")
DS_C1_MIN, DS_C1_MAX       = float(_df_ranges['C1'].min()), float(_df_ranges['C1'].max())
DS_C2_MIN, DS_C2_MAX       = float(_df_ranges['C2'].min()), float(_df_ranges['C2'].max())
DS_C3_MIN, DS_C3_MAX       = float(_df_ranges['C3'].min()), float(_df_ranges['C3'].max())
DS_ALPHA_MIN, DS_ALPHA_MAX = float(_df_ranges['Alpha'].min()), float(_df_ranges['Alpha'].max())
DS_CL_MIN, DS_CL_MAX       = float(_df_ranges['CL'].min()), float(_df_ranges['CL'].max())
DS_T_MIN = DS_C1_MIN + DS_C2_MIN
DS_T_MAX = DS_C1_MAX + DS_C2_MAX
RE_TRAIN_MIN = float(_df_ranges['Re'].min())
RE_TRAIN_MAX = float(_df_ranges['Re'].max())
del _df_ranges

geo_dir = str(ROOT_DIR / "geometry")
if geo_dir not in sys.path:
    sys.path.append(geo_dir)

from airfoil import generate_profile

# ============================================================
#  NEURAL NETWORK ARCHITECTURES
#  Defined in training/models.py -- shared with train_*_net.py
# ============================================================

models_dir = str(MODELS_DIR)
if models_dir not in sys.path:
    sys.path.append(models_dir)

from models import GlobalForceNet, FlowFieldNet


@st.cache_resource
def load_models():
    force_net = GlobalForceNet()
    force_net.load_state_dict(torch.load(MODELS_DIR / "force_net.pth", map_location="cpu"))
    force_net.eval()

    field_net = FlowFieldNet()
    field_net.load_state_dict(torch.load(MODELS_DIR / "field_net.pth", map_location="cpu"))
    field_net.eval()

    scaler_force   = joblib.load(MODELS_DIR / "scaler_force.pkl")
    scaler_force_y = joblib.load(MODELS_DIR / "scaler_force_y.pkl")
    scaler_field_X = joblib.load(MODELS_DIR / "scaler_field_X.pkl")
    scaler_field_y = joblib.load(MODELS_DIR / "scaler_field_y.pkl")

    return force_net, field_net, scaler_force, scaler_force_y, scaler_field_X, scaler_field_y


force_net, field_net, scaler_force, scaler_force_y, scaler_field_X, scaler_field_y = load_models()

# ============================================================
#  CORE PREDICTION FUNCTIONS
# ============================================================

def predict_forces(c1, c2, c3, Re, alpha):
    """Return predicted CL and CD for the given geometry and flow conditions."""
    X = scaler_force.transform(np.array([[c1, c2, c3, np.log10(Re), alpha]]))
    with torch.no_grad():
        pred_scaled = force_net(torch.tensor(X).float()).numpy()
    pred = scaler_force_y.inverse_transform(pred_scaled)
    CL = pred[0, 0]
    CD = np.exp(pred[0, 1])
    return CL, CD


def compute_flow_fields(c1, c2, c3, Re, alpha):
    """
    Build a 300x150 structured grid, mask the airfoil interior, and
    predict flow field variables at all exterior points.
    """
    x_up, y_up, x_lo, y_lo = generate_profile(c1, c2, c3, alpha, n_points=200)
    polygon = np.vstack((
        np.hstack((x_up, x_lo[::-1])),
        np.hstack((y_up, y_lo[::-1]))
    )).T
    airfoil_mask_path = MplPath(polygon)

    nx, ny = 300, 150
    x_grid = np.linspace(-0.5, 1.8, nx)
    y_grid = np.linspace(-0.6, 0.6, ny)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    fluid_mask  = ~airfoil_mask_path.contains_points(grid_points)
    fluid_pts   = grid_points[fluid_mask]
    n_fluid     = fluid_pts.shape[0]

    features = np.zeros((n_fluid, 7))
    features[:, 0:2] = fluid_pts
    features[:, 2]   = c1
    features[:, 3]   = c2
    features[:, 4]   = c3
    features[:, 5]   = np.log10(Re)
    features[:, 6]   = alpha

    features_scaled = scaler_field_X.transform(features)
    with torch.no_grad():
        pred_scaled = field_net(torch.tensor(features_scaled).float()).numpy()
    pred = scaler_field_y.inverse_transform(pred_scaled)
    # FlowFieldNet predicts log(k) and log(omega) (see training/train_field_net.py) --
    # undo it to recover the physical values.
    pred[:, 3] = np.exp(pred[:, 3])
    pred[:, 4] = np.exp(pred[:, 4])

    nan_grid = lambda: np.full(X_grid.shape, np.nan)
    fields = {k: nan_grid() for k in ('Ux', 'Uy', 'Umag', 'Cp', 'k', 'omega')}

    Ux_vals = pred[:, 0]
    Uy_vals = pred[:, 1]
    fields['Ux'].flat[fluid_mask]   = Ux_vals
    fields['Uy'].flat[fluid_mask]   = Uy_vals
    fields['Umag'].flat[fluid_mask] = np.sqrt(Ux_vals**2 + Uy_vals**2)
    fields['Cp'].flat[fluid_mask]   = pred[:, 2]
    fields['k'].flat[fluid_mask]    = pred[:, 3]
    fields['omega'].flat[fluid_mask] = pred[:, 4]

    return X_grid, Y_grid, fields, x_up, y_up, x_lo, y_lo


def render_flow_map(X, Y, z_adim, z_si, unit, var_name, x_up, y_up, x_lo, y_lo, chord=1.0):
    """Render an interactive Plotly contour map in physical coordinates (metres)."""
    fig = go.Figure()

    x_phys = X[0, :] * chord
    y_phys = Y[:, 0] * chord

    fig.add_trace(go.Contour(
        x=x_phys, y=y_phys, z=z_adim,
        colorscale='Turbid' if var_name in ('k', 'omega') else 'Jet',
        contours_coloring='heatmap',
        line_width=0,
        line_smoothing=1.3,
        colorbar=dict(title=dict(text=f"{var_name} (adim.)", side='right')),
        connectgaps=False,
        customdata=z_si,
        hovertemplate=(
            "<b>Coordenadas:</b><br>"
            "x = %{x:.3f} m<br>"
            "y = %{y:.3f} m<br>"
            "<b>Valor físico:</b><br>"
            "%{customdata:.3f} " + unit +
            "<extra></extra>"
        )
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([x_up, x_lo[::-1]]) * chord,
        y=np.concatenate([y_up, y_lo[::-1]]) * chord,
        fill='toself',
        fillcolor='rgba(200, 200, 200, 1.0)',
        line=dict(color='black', width=2.5),
        showlegend=False,
    ))

    fig.update_layout(
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        xaxis=dict(range=[-0.4 * chord, 1.6 * chord], constrain="domain"),
        yaxis=dict(range=[-0.5 * chord, 0.5 * chord], scaleanchor="x", scaleratio=1),
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white",
    )
    return fig


# ============================================================
#  STREAMLIT INTERFACE
# ============================================================

st.set_page_config(page_title="AeroSurrogate", layout="wide")
st.title("Plataforma de Análisis Aerodinámico — Surrogate Model")
st.markdown("**TFG Diego Antúnez** · Diseño de perfiles NACA mediante redes neuronales entrenadas sobre datos CFD")
st.divider()

mode = st.sidebar.selectbox(
    "Modo de operación:",
    ["Calculadora de Campos — Modo Directo", "Optimizador Geométrico — Modo Inverso"]
)

# ============================================================
#  MODE 1: DIRECT FIELD CALCULATOR
# ============================================================

if mode == "Calculadora de Campos — Modo Directo":
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.header("Parámetros de Entrada")

        st.subheader("Condiciones del Fluido")
        velocity = st.number_input("Velocidad de flujo libre U (m/s):", min_value=1.0, max_value=300.0, value=45.0, step=5.0)
        chord    = st.number_input("Longitud de cuerda c (m):",          min_value=0.01, max_value=5.0,   value=1.0,  step=0.1)
        alpha    = st.slider("Ángulo de ataque α (°):", DS_ALPHA_MIN, DS_ALPHA_MAX, 0.0, 0.5)

        Re = (velocity * chord) / NU_AIR
        st.info(f"Número de Reynolds: **{Re:,.0f}**")
        if Re < RE_TRAIN_MIN or Re > RE_TRAIN_MAX:
            st.warning(
                f"Re={Re:,.0f} está fuera del rango de entrenamiento "
                f"[{RE_TRAIN_MIN/1e6:.0f} M – {RE_TRAIN_MAX/1e6:.0f} M]. "
                "Los resultados pueden ser poco fiables."
            )

        mach = velocity / SPEED_OF_SOUND
        if mach > MACH_WARN:
            st.warning(
                f"M={mach:.2f} (U={velocity:.0f} m/s) supera el umbral de flujo incompresible "
                f"(M>{MACH_WARN}). El dataset se generó con simpleFoam (incompresible) — dos "
                "condiciones con el mismo Re pero distinto Mach son indistinguibles para el "
                "modelo, y este régimen no fue simulado. Resultados no fiables."
            )

        st.subheader("Geometría del Perfil (metros)")
        c1_m = st.slider("C1 extradós (m):",       round(DS_C1_MIN*chord, 4), round(DS_C1_MAX*chord, 4), round(0.10*chord, 4), round(0.01*chord, 4))
        c2_m = st.slider("C2 intradós (m):",       round(DS_C2_MIN*chord, 4), round(DS_C2_MAX*chord, 4), round(0.0*chord, 4),  round(0.01*chord, 4))
        c3_m = st.slider("C3 posición camber (m):", round(DS_C3_MIN*chord, 4), round(DS_C3_MAX*chord, 4), round(0.40*chord, 4), round(0.05*chord, 4))
        c1 = c1_m / chord
        c2 = c2_m / chord
        c3 = c3_m / chord

        t = c1 + c2
        m = (c1 - c2) / 2
        t_m = t * chord
        st.metric("Espesor máximo", f"{t_m:.4f} m  ({t * 100:.1f} %c)")
        st.metric("Camber máximo",  f"{m * chord:.4f} m  ({m * 100:.1f} %c)")

    with col_right:
        st.header("Resultados")
        if t < DS_T_MIN or t > DS_T_MAX:
            st.error(f"Espesor t={t_m:.4f} m ({t*100:.1f}%c) fuera del rango del dataset ({DS_T_MIN*100:.0f}–{DS_T_MAX*100:.0f}%c).")
        else:
            CL, CD = predict_forces(c1, c2, c3, Re, alpha)
            efficiency = CL / CD if CD != 0 else 0.0

            k1, k2, k3 = st.columns(3)
            k1.metric("CL", f"{CL:.4f}")
            k2.metric("CD", f"{CD:.5f}")
            k3.metric("CL / CD", f"{efficiency:.2f}")

            X, Y, fields, xu, yu, xl, yl = compute_flow_fields(c1, c2, c3, Re, alpha)

            st.subheader("Campos Fluidodinámicos")
            var_sel = st.selectbox(
                "Variable a visualizar:",
                ["Magnitud de velocidad (Umag)", "Velocidad X (Ux)", "Velocidad Y (Uy)",
                 "Coeficiente de presión (Cp)", "Energía cinética turbulenta (k)",
                 "Disipación específica (omega)"]
            )

            var_map = {
                "Magnitud de velocidad (Umag)":    ('Umag',  velocity,                       "m/s"),
                "Velocidad X (Ux)":                ('Ux',    velocity,                       "m/s"),
                "Velocidad Y (Uy)":                ('Uy',    velocity,                       "m/s"),
                "Coeficiente de presión (Cp)":     ('Cp',    0.5 * RHO_AIR * velocity**2,    "Pa"),
                "Energía cinética turbulenta (k)": ('k',     velocity**2,                    "m2/s2"),
                "Disipación específica (omega)":   ('omega', velocity / chord,               "1/s"),
            }

            key, factor, unit = var_map[var_sel]
            z_adim = fields[key]
            z_si   = z_adim * factor

            st.plotly_chart(render_flow_map(X, Y, z_adim, z_si, unit, key, xu, yu, xl, yl, chord=chord),
                            use_container_width=True)

# ============================================================
#  MODE 2: GENETIC ALGORITHM OPTIMISER
# ============================================================

else:
    st.header("Optimizador Geométrico — Modo Inverso")
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Objetivos del Diseño")
        v_target = st.number_input("Velocidad de operación U (m/s):", min_value=1.0, max_value=300.0, value=45.0, step=5.0)
        c_target = st.number_input("Cuerda de construcción c (m):",   min_value=0.01, max_value=5.0,   value=1.0,  step=0.1)

        Re_target = (v_target * c_target) / NU_AIR
        st.info(f"Reynolds de diseño: **{Re_target:,.0f}**")
        if Re_target < RE_TRAIN_MIN or Re_target > RE_TRAIN_MAX:
            st.warning(
                f"Re={Re_target:,.0f} está fuera del rango de entrenamiento "
                f"[{RE_TRAIN_MIN/1e6:.0f} M – {RE_TRAIN_MAX/1e6:.0f} M]. "
                "Los resultados del optimizador pueden ser poco fiables."
            )

        mach_target = v_target / SPEED_OF_SOUND
        if mach_target > MACH_WARN:
            st.warning(
                f"M={mach_target:.2f} (U={v_target:.0f} m/s) supera el umbral de flujo "
                f"incompresible (M>{MACH_WARN}). El dataset se generó con simpleFoam "
                "(incompresible) — este régimen no fue simulado. Resultados del optimizador "
                "no fiables."
            )

        _opt_options = ["Maximizar eficiencia aerodinámica (CL/CD)"]
        if ENABLE_CL_TARGET:
            _opt_options.append("Alcanzar un CL objetivo")
        opt_criterion = st.radio("Criterio de optimización:", _opt_options)
        CL_target = 0.0
        if opt_criterion == "Alcanzar un CL objetivo":
            CL_target = st.slider("CL deseado:", round(DS_CL_MIN, 2), round(DS_CL_MAX, 2), 0.6, 0.01)

        with st.expander("Restricciones avanzadas (opcional)"):
            st.caption("Fija parámetros que no deben variar durante la búsqueda.")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                fix_c1 = st.checkbox("Fijar C1")
                c1_fixed = st.slider("C1 (m):", round(DS_C1_MIN*c_target,4), round(DS_C1_MAX*c_target,4), round(0.10*c_target,4), round(0.01*c_target,4), key="c1_fijo") / c_target if fix_c1 else None
                fix_c3 = st.checkbox("Fijar C3")
                c3_fixed = st.slider("C3 (m):", round(DS_C3_MIN*c_target,4), round(DS_C3_MAX*c_target,4), round(0.40*c_target,4), round(0.05*c_target,4), key="c3_fijo") / c_target if fix_c3 else None
            with col_r2:
                fix_c2 = st.checkbox("Fijar C2")
                c2_fixed = st.slider("C2 (m):", round(DS_C2_MIN*c_target,4), round(DS_C2_MAX*c_target,4), 0.0, round(0.01*c_target,4), key="c2_fijo") / c_target if fix_c2 else None
                fix_alpha = st.checkbox("Fijar ángulo de ataque")
                alpha_fixed = st.slider("Alpha (°):", DS_ALPHA_MIN, DS_ALPHA_MAX, 0.0, 0.5, key="alpha_fijo") if fix_alpha else None
            CL_min_req = None
            if ENABLE_CL_MIN:
                use_CL_min = st.checkbox("Imponer CL mínimo")
                CL_min_req = st.slider("CL mínimo:", round(DS_CL_MIN, 2), round(DS_CL_MAX, 2), 0.3, 0.01, key="cl_min_req") if use_CL_min else None

            st.divider()
            fix_t = st.checkbox("Fijar espesor (t = C1 + C2)")
            t_fixed = None
            if fix_t:
                if fix_c1 and fix_c2:
                    t_fixed = round(c1_fixed + c2_fixed, 3)
                    st.info(f"Espesor determinado por C1 y C2: **{t_fixed*c_target:.4f} m** ({t_fixed*100:.1f}%c)")
                else:
                    t_min_ui, t_max_ui = DS_T_MIN, DS_T_MAX
                    if fix_c1:
                        t_min_ui = max(t_min_ui, round(c1_fixed + DS_C2_MIN, 4))
                        t_max_ui = min(t_max_ui, round(c1_fixed + DS_C2_MAX, 4))
                    if fix_c2:
                        t_min_ui = max(t_min_ui, round(DS_C1_MIN + c2_fixed, 4))
                        t_max_ui = min(t_max_ui, round(DS_C1_MAX + c2_fixed, 4))
                    if t_min_ui >= t_max_ui:
                        st.error("Rango de espesor inválido con los coeficientes fijados.")
                    else:
                        t_def_m = float(np.clip(0.12 * c_target, t_min_ui * c_target, t_max_ui * c_target))
                        t_m_sel = st.slider("Espesor (m):", round(t_min_ui * c_target, 4), round(t_max_ui * c_target, 4),
                                            round(t_def_m, 4), round(0.01 * c_target, 4), key="t_fijo")
                        t_fixed = t_m_sel / c_target
                        if fix_c1:
                            st.caption(f"→ C2 = **{(t_fixed - c1_fixed)*c_target:.4f} m**")
                        elif fix_c2:
                            st.caption(f"→ C1 = **{(t_fixed - c2_fixed)*c_target:.4f} m**")

        run_btn = st.button("Iniciar optimización", type="primary")

    with col_right:
        if run_btn:
            with st.spinner("Ejecutando optimización genética ..."):

                # Build free/fixed parameter map: [C1, C2, C3, Alpha]
                LIM_MIN = np.array([DS_C1_MIN, DS_C2_MIN, DS_C3_MIN, DS_ALPHA_MIN])
                LIM_MAX = np.array([DS_C1_MAX, DS_C2_MAX, DS_C3_MAX, DS_ALPHA_MAX])
                fixed_map = {0: c1_fixed, 1: c2_fixed, 2: c3_fixed, 3: alpha_fixed}

                use_t_constraint = False
                if t_fixed is not None:
                    if fix_c1 and not fix_c2:
                        fixed_map[1] = t_fixed - c1_fixed
                    elif fix_c2 and not fix_c1:
                        fixed_map[0] = t_fixed - c2_fixed
                    elif not fix_c1 and not fix_c2:
                        use_t_constraint = True

                if use_t_constraint:
                    c1_t_min = max(DS_C1_MIN, t_fixed - DS_C2_MAX)
                    c1_t_max = min(DS_C1_MAX, t_fixed - DS_C2_MIN)
                    LIM_MIN_ADJ = LIM_MIN.copy(); LIM_MIN_ADJ[0] = c1_t_min
                    LIM_MAX_ADJ = LIM_MAX.copy(); LIM_MAX_ADJ[0] = c1_t_max
                    free_idx    = [i for i in range(4) if fixed_map[i] is None and i != 1]
                    lim_min     = LIM_MIN_ADJ[free_idx]
                    lim_max     = LIM_MAX_ADJ[free_idx]

                    def reconstruct(pop):
                        n = len(pop)
                        full = np.empty((n, 4))
                        j = 0
                        for i in range(4):
                            if i == 1:
                                full[:, 1] = t_fixed - full[:, 0]
                            elif fixed_map[i] is None:
                                full[:, i] = pop[:, j]; j += 1
                            else:
                                full[:, i] = fixed_map[i]
                        return full
                else:
                    free_idx = [i for i in range(4) if fixed_map[i] is None]
                    lim_min  = LIM_MIN[free_idx]
                    lim_max  = LIM_MAX[free_idx]

                    def reconstruct(pop):
                        n = len(pop)
                        full = np.empty((n, 4))
                        j = 0
                        for i in range(4):
                            if fixed_map[i] is None:
                                full[:, i] = pop[:, j]; j += 1
                            else:
                                full[:, i] = fixed_map[i]
                        return full

                n_dims = len(free_idx)
                n_pop, n_gen, n_restarts = 200, 100, 5
                best_individual = None
                best_fitness    = -np.inf
                progress_bar    = st.progress(0)
                total_steps     = n_restarts * n_gen

                for restart in range(n_restarts):
                    np.random.seed(restart * 13)
                    population = np.random.uniform(lim_min, lim_max, (n_pop, n_dims))

                    for gen in range(n_gen):
                        full = reconstruct(population)
                        X_eval = np.zeros((len(full), 5))
                        X_eval[:, 0:3] = full[:, 0:3]
                        X_eval[:, 3]   = np.log10(Re_target)
                        X_eval[:, 4]   = full[:, 3]

                        X_eval_scaled = scaler_force.transform(X_eval)
                        with torch.no_grad():
                            preds_scaled = force_net(torch.tensor(X_eval_scaled).float()).numpy()
                        preds = scaler_force_y.inverse_transform(preds_scaled)

                        CLs = preds[:, 0]
                        CDs = np.exp(preds[:, 1])

                        if opt_criterion == "Maximizar eficiencia aerodinámica (CL/CD)":
                            fitness = CLs / CDs
                        else:
                            fitness = -np.abs(CLs - CL_target)

                        if CL_min_req is not None:
                            fitness -= np.maximum(0, CL_min_req - CLs) * 100

                        t_vals = full[:, 0] + full[:, 1]
                        fitness[~((t_vals >= DS_T_MIN) & (t_vals <= DS_T_MAX))] = -np.inf

                        sorted_idx = np.argsort(fitness)[::-1]
                        parents    = population[sorted_idx[:n_pop // 2]]

                        sigma    = 0.05 * (1 - gen / n_gen) + 0.005
                        children = parents + np.random.normal(0, sigma, parents.shape)
                        children = np.clip(children, lim_min, lim_max)
                        population = np.vstack((parents, children))

                        progress_bar.progress(int((restart * n_gen + gen + 1) / total_steps * 100))

                    top_fitness = fitness[sorted_idx[0]]
                    if top_fitness > best_fitness:
                        best_fitness    = top_fitness
                        # population was already reassigned to the next generation (vstack(parents,children))
                        # just above; by elitism, population[0] == parents[0] == the best individual
                        # of THIS generation (population[sorted_idx[0]] would index the reassigned array
                        # with an index from the old population -- bug fixed).
                        best_individual = reconstruct(population[0].reshape(1, -1))[0]

                if best_individual is None:
                    st.error("No se encontró ningún perfil válido. Revisa las restricciones.")
                    st.stop()

                st.session_state.c1_opt      = best_individual[0]
                st.session_state.c2_opt      = best_individual[1]
                st.session_state.c3_opt      = best_individual[2]
                st.session_state.alpha_opt   = best_individual[3]
                st.session_state.v_target    = v_target
                st.session_state.c_target    = c_target
                st.session_state.Re_target   = Re_target
                st.session_state.opt_done    = True

        if st.session_state.get('opt_done', False):
            c1_opt   = st.session_state.c1_opt
            c2_opt   = st.session_state.c2_opt
            c3_opt   = st.session_state.c3_opt
            alpha_opt = st.session_state.alpha_opt
            v_opt    = st.session_state.v_target
            c_opt    = st.session_state.c_target
            Re_opt   = st.session_state.Re_target

            t_opt = c1_opt + c2_opt
            CL_opt, CD_opt = predict_forces(c1_opt, c2_opt, c3_opt, Re_opt, alpha_opt)
            eff_opt = CL_opt / CD_opt if CD_opt != 0 else 0.0

            st.success("Optimización completada.")

            st.subheader("Perfil Óptimo")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("C1",    f"{c1_opt:.4f}")
            r2.metric("C2",    f"{c2_opt:.4f}")
            r3.metric("C3",    f"{c3_opt:.4f}")
            r4.metric("Alpha", f"{alpha_opt:.2f} °")

            k1, k2, k3 = st.columns(3)
            k1.metric("CL",      f"{CL_opt:.4f}")
            k2.metric("CD",      f"{CD_opt:.5f}")
            k3.metric("CL / CD", f"{eff_opt:.2f}")

            st.divider()
            st.subheader("Campos Fluidodinámicos del Perfil Óptimo")
            st.metric("Espesor máximo t", f"{t_opt:.3f}  ({t_opt * 100:.1f} %)")

            var_sel_opt = st.selectbox(
                "Variable a visualizar:",
                ["Magnitud de velocidad (Umag)", "Velocidad X (Ux)", "Velocidad Y (Uy)",
                 "Coeficiente de presión (Cp)", "Energía cinética turbulenta (k)",
                 "Disipación específica (omega)"],
                key="var_opt"
            )

            var_map_opt = {
                "Magnitud de velocidad (Umag)":    ('Umag',  v_opt,                      "m/s"),
                "Velocidad X (Ux)":                ('Ux',    v_opt,                      "m/s"),
                "Velocidad Y (Uy)":                ('Uy',    v_opt,                      "m/s"),
                "Coeficiente de presión (Cp)":     ('Cp',    0.5 * RHO_AIR * v_opt**2,  "Pa"),
                "Energía cinética turbulenta (k)": ('k',     v_opt**2,                  "m2/s2"),
                "Disipación específica (omega)":   ('omega', v_opt / c_opt,             "1/s"),
            }

            key_opt, factor_opt, unit_opt = var_map_opt[var_sel_opt]

            X_g, Y_g, fields_g, xu_g, yu_g, xl_g, yl_g = compute_flow_fields(
                c1_opt, c2_opt, c3_opt, Re_opt, alpha_opt
            )
            z_adim_g = fields_g[key_opt]
            z_si_g   = z_adim_g * factor_opt

            st.plotly_chart(
                render_flow_map(X_g, Y_g, z_adim_g, z_si_g, unit_opt, key_opt,
                                xu_g, yu_g, xl_g, yl_g, chord=c_target),
                use_container_width=True
            )
