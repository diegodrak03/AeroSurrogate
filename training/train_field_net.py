import gc
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
PLOTS_DIR  = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from models import FlowFieldNet  # noqa: E402 (arquitectura compartida con train_force_net.py y app.py)

RANDOM_STATE = 42
VAL_FRACTION = 0.2
TARGET_NAMES = ['Ux_adim', 'Uy_adim', 'Cp', 'k', 'omega']
FLOAT_COLS   = ['x', 'y', 'C1', 'C2', 'C3', 'Re', 'Alpha'] + TARGET_NAMES


class FlowDataset(Dataset):
    def __init__(self, X, y):
        # X, y ya vienen en float32 -- .float() aqui no duplica memoria.
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Data loading ---
# dataset_fields.csv tiene decenas de millones de filas (~37M con la malla
# actual). Todo lo de abajo esta pensado para minimizar el pico de memoria:
# float32 en vez del float64 por defecto de pandas (mitad de RAM por celda),
# y `del` + `gc.collect()` en cuanto una copia intermedia deja de hacer
# falta, en vez de dejar que todas las copias convivan hasta el final del
# script. Version anterior (sin esto) llego a agotar la RAM de la maquina.
print("Loading dataset (may take a while, decenas de millones de filas) ...")
df = pd.read_csv(SCRIPT_DIR / "dataset_fields.csv", dtype={c: np.float32 for c in FLOAT_COLS})
print(f"Points: {len(df)}")

# Split por simulacion, no por fila: las filas de una misma simulacion son
# puntos de malla muy correlacionados entre si (celdas vecinas). Un split
# aleatorio por fila dejaria fugar informacion geometrica de la misma
# simulacion a train y val, subestimando el error real de generalizacion.
# Se usa groupby().ngroup() (array de enteros compacto) en vez de construir
# un identificador de simulacion como string por fila, que con ~37M filas
# supondria varios GB solo en objetos string de Python.
group_id = df.groupby(['C1', 'C2', 'C3', 'Re', 'Alpha'], sort=False).ngroup().to_numpy()
n_sims = int(group_id.max()) + 1
rng = np.random.RandomState(RANDOM_STATE)
perm = rng.permutation(n_sims)
n_val_sims = max(1, int(n_sims * VAL_FRACTION))
val_group_ids = perm[:n_val_sims]
is_val = np.isin(group_id, val_group_ids)
del group_id, perm, val_group_ids
gc.collect()

print(f"Simulaciones: {n_sims}  (train={n_sims - n_val_sims}  val={n_val_sims})")
print(f"Filas: train={int((~is_val).sum())}  val={int(is_val.sum())}")

X_raw = df[['x', 'y', 'C1', 'C2', 'C3', 'Re', 'Alpha']].to_numpy(dtype=np.float32)
y_raw = df[TARGET_NAMES].to_numpy(dtype=np.float32)
del df
gc.collect()

X_raw[:, 5] = np.log10(X_raw[:, 5])

# k y omega tienen un rango dinamico muy amplio (varios ordenes de magnitud,
# cola larga) y no llevaban ninguna adimensionalizacion fisica previa (a
# diferencia de Ux_adim/Uy_adim/Cp) -- con MSE en escala bruta, el modelo
# subpredecia sistematicamente los valores altos (ver docs §9.9). Se
# transforman a log antes de escalar, igual que ya se hace con CD en
# train_force_net.py; se deshace con exp() tras predecir (mas abajo, y en
# app.py::compute_flow_fields). Verificado: 0 filas con k<=0 u omega<=0 en
# las 36.87M filas del dataset real, log() no puede dar NaN/-inf aqui.
y_raw[:, 3] = np.log(y_raw[:, 3])   # k -> log(k)
y_raw[:, 4] = np.log(y_raw[:, 4])   # omega -> log(omega)

X_train_raw, y_train_raw = X_raw[~is_val], y_raw[~is_val]
X_val_raw,   y_val_raw   = X_raw[is_val],  y_raw[is_val]
del X_raw, y_raw, is_val
gc.collect()

# Los scalers se ajustan solo con las filas de entrenamiento (sin fuga de
# estadisticas del conjunto de validacion), igual que en train_force_net.py.
# sklearn trabaja internamente en float64 durante el fit/transform, pero el
# resultado se vuelve a bajar a float32 de inmediato para no arrastrar el
# doble de memoria el resto del script.
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw).astype(np.float32)
X_val_scaled   = scaler_X.transform(X_val_raw).astype(np.float32)
del X_train_raw, X_val_raw
gc.collect()

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_raw).astype(np.float32)
y_val_scaled   = scaler_y.transform(y_val_raw).astype(np.float32)
del y_train_raw, y_val_raw
gc.collect()

joblib.dump(scaler_X, SCRIPT_DIR / "scaler_field_X.pkl")
joblib.dump(scaler_y, SCRIPT_DIR / "scaler_field_y.pkl")

train_ds = FlowDataset(X_train_scaled, y_train_scaled)
val_ds   = FlowDataset(X_val_scaled, y_val_scaled)
del X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled
gc.collect()

train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=8192, shuffle=False)

# --- Training ---
model     = FlowFieldNet()
loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

EPOCHS        = 25
PATIENCE      = 5
best_val_loss = float('inf')
wait          = 0
best_state    = None
history       = {'train': [], 'val': []}

print(f"\nTraining FlowFieldNet ({sum(p.numel() for p in model.parameters())} params) ...")
print("-" * 60)
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0
    for X_batch, y_batch in train_loader:
        preds = model(X_batch)
        loss  = loss_fn(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches  += 1
        if n_batches % 200 == 0:
            print(f"  batch {n_batches}  loss={loss.item():.6f}", end="\r")
    train_loss = epoch_loss / n_batches

    model.eval()
    val_loss_sum  = 0.0
    n_val_batches = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            val_loss_sum  += loss_fn(model(X_batch), y_batch).item()
            n_val_batches += 1
    val_loss = val_loss_sum / n_val_batches

    history['train'].append(train_loss)
    history['val'].append(val_loss)
    scheduler.step(val_loss)

    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:02d}/{EPOCHS}  train={train_loss:.6f}  val={val_loss:.6f}  lr={lr:.1e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (best val={best_val_loss:.6f})")
            break

print("-" * 60)

# --- Restore best & evaluate on held-out simulations ---
model.load_state_dict(best_state)
model.eval()

preds_list, reals_list = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        pred_scaled = model(X_batch).numpy()
        preds_list.append(scaler_y.inverse_transform(pred_scaled))
        reals_list.append(scaler_y.inverse_transform(y_batch.numpy()))
pred_real = np.concatenate(preds_list)
y_real    = np.concatenate(reals_list)

# Deshacer el log-transform de k/omega (columnas 3 y 4) para que MAE y
# graficas queden en unidades fisicas reales, no en escala logaritmica.
pred_real[:, 3] = np.exp(pred_real[:, 3])
pred_real[:, 4] = np.exp(pred_real[:, 4])
y_real[:, 3]    = np.exp(y_real[:, 3])
y_real[:, 4]    = np.exp(y_real[:, 4])

print(f"Val MAE por variable ({n_val_sims} simulaciones no vistas en entrenamiento):")
for i, name in enumerate(TARGET_NAMES):
    mae = np.mean(np.abs(pred_real[:, i] - y_real[:, i]))
    print(f"  {name:10s}: {mae:.6f}")

# --- Save artefacts ---
torch.save(model.state_dict(), SCRIPT_DIR / "field_net.pth")
print("Saved: field_net.pth, scaler_field_X.pkl, scaler_field_y.pkl")

# --- Training plot ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history['train'], label='Train loss', linewidth=1.0)
ax.plot(history['val'],   label='Val loss',   linewidth=1.0)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE (scaled)')
ax.set_title('FlowFieldNet — Training History')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "field_net_loss.png", dpi=150)
plt.close(fig)

# --- Scatter plot (submuestreado para no saturar el grafico con millones de puntos) ---
n_plot = min(20000, len(y_real))
idx = np.random.RandomState(RANDOM_STATE).choice(len(y_real), n_plot, replace=False)
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
for i, name in enumerate(TARGET_NAMES):
    ax = axes[i]
    real = y_real[idx, i]
    pred = pred_real[idx, i]
    ax.scatter(real, pred, s=2, alpha=0.25, edgecolors='none')
    lims = [min(real.min(), pred.min()), max(real.max(), pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel(f'{name} real')
    ax.set_ylabel(f'{name} predicho')
    ax.set_title(f'{name} (MAE={np.mean(np.abs(real - pred)):.4f})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
fig.suptitle('FlowFieldNet — Predicción vs Realidad (simulaciones de validación)', fontsize=12)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "field_net_scatter.png", dpi=150)
plt.close(fig)

print(f"Plots guardados en {PLOTS_DIR}/")
