import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
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
from models import GlobalForceNet  # noqa: E402 (arquitectura compartida con train_field_net.py y app.py)


# --- Data loading ---
print("Loading dataset ...")
df = pd.read_csv(SCRIPT_DIR / "dataset_forces.csv")

X_raw = df[['C1', 'C2', 'C3', 'Re', 'Alpha']].values
y_raw = df[['CL', 'CD']].values

X_raw[:, 3] = np.log10(X_raw[:, 3])
y_raw[:, 1] = np.log(y_raw[:, 1])

print(f"Samples: {X_raw.shape[0]}")
print(f"CL range:      [{y_raw[:, 0].min():.4f}, {y_raw[:, 0].max():.4f}]")
print(f"log(CD) range:  [{y_raw[:, 1].min():.4f}, {y_raw[:, 1].max():.4f}]")

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled  = scaler_y.transform(y_test)

train_ds = TensorDataset(torch.tensor(X_train_scaled).float(),
                         torch.tensor(y_train_scaled).float())
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

X_test_t  = torch.tensor(X_test_scaled).float()
y_test_t  = torch.tensor(y_test_scaled).float()

# --- Training ---
model     = GlobalForceNet()
loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40, factor=0.5)

EPOCHS        = 2000
PATIENCE      = 100
best_val_loss = float('inf')
wait          = 0
best_state    = None
history       = {'train': [], 'val': []}

print(f"\nTraining GlobalForceNet ({sum(p.numel() for p in model.parameters())} params) ...")
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

    train_loss = epoch_loss / n_batches

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_test_t), y_test_t).item()

    history['train'].append(train_loss)
    history['val'].append(val_loss)
    scheduler.step(val_loss)

    if epoch % 50 == 0 or epoch == 1:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:04d}/{EPOCHS}  train={train_loss:.6f}  val={val_loss:.6f}  lr={lr:.1e}")

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

# --- Restore best & evaluate ---
model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    pred_scaled = model(X_test_t).numpy()
    pred_log = scaler_y.inverse_transform(pred_scaled)
    y_test_log = scaler_y.inverse_transform(y_test_t.numpy())

# Convert log(CD) back to CD for metrics and plots
pred_real = pred_log.copy()
pred_real[:, 1] = np.exp(pred_real[:, 1])
y_real = y_test_log.copy()
y_real[:, 1] = np.exp(y_real[:, 1])

mae_CL = np.mean(np.abs(pred_real[:, 0] - y_real[:, 0]))
mae_CD = np.mean(np.abs(pred_real[:, 1] - y_real[:, 1]))
print(f"Test MAE  CL: {mae_CL:.5f}   CD: {mae_CD:.6f}")

# --- Save artefacts ---
torch.save(model.state_dict(), SCRIPT_DIR / "force_net.pth")
joblib.dump(scaler_X, SCRIPT_DIR / "scaler_force.pkl")
joblib.dump(scaler_y, SCRIPT_DIR / "scaler_force_y.pkl")
print("Saved: force_net.pth, scaler_force.pkl, scaler_force_y.pkl")

# --- Training plots ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history['train'], label='Train loss', linewidth=0.8)
ax.plot(history['val'],   label='Val loss',   linewidth=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE (scaled)')
ax.set_title('GlobalForceNet — Training History')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "force_net_loss.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for i, (name, unit) in enumerate([("CL", ""), ("CD", "")]):
    ax = axes[i]
    real = y_real[:, i]
    pred = pred_real[:, i]
    ax.scatter(real, pred, s=4, alpha=0.4, edgecolors='none')
    lims = [min(real.min(), pred.min()), max(real.max(), pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect')
    ax.set_xlabel(f'{name} real')
    ax.set_ylabel(f'{name} predicho')
    ax.set_title(f'{name}  (MAE={np.mean(np.abs(real - pred)):.4f})')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
fig.suptitle('GlobalForceNet — Predicción vs Realidad (test set)', fontsize=12)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "force_net_scatter.png", dpi=150)
plt.close(fig)

print(f"Plots saved to {PLOTS_DIR}/")
