import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SOM Rainfall Analysis", layout="wide")
st.title("🌧️ Analisis Curah Hujan dengan SOM (Self-Organizing Map)")

# -------------------------
# 1. BACA DATA LANGSUNG
# -------------------------
DATA_PATH = "data_kompres.nc"  # ganti path sesuai lokasi data Anda

try:
    ds_tmp = xr.open_dataset(DATA_PATH)
except Exception as e:
    st.error(f"Gagal membuka file NetCDF: {e}")
    st.stop()

# -------------------------
# Fungsi bantu
# -------------------------
def pick_coord_names(ds):
    lat_names = ['latitude', 'lat', 'y']
    lon_names = ['longitude', 'lon', 'x']
    lat_name = next((n for n in lat_names if n in ds.coords), None)
    lon_name = next((n for n in lon_names if n in ds.coords), None)
    return lat_name, lon_name

def load_and_climatology(ds, var_name):
    da = ds[var_name]
    if 'time' not in da.dims:
        raise ValueError("Variabel tidak memiliki dimensi 'time'.")
    clim = da.groupby("time.month").mean("time")
    return clim

def smooth_majority(clust, coords, tree, r_km=80, k=30):
    r = r_km * 1000
    smoothed = clust.copy()
    n = coords.shape[0]
    for i in range(n):
        idx = tree.query_ball_point(coords[i], r)
        if len(idx) < 5:
            _, idx = tree.query(coords[i], k=min(k, n))
            if np.isscalar(idx): idx = [idx]
        vals, counts = np.unique(clust[idx], return_counts=True)
        smoothed[i] = vals[np.argmax(counts)]
    return smoothed

# -------------------------
# 2. Ambil variabel & koordinat
# -------------------------
lat_name, lon_name = pick_coord_names(ds_tmp)
if lat_name is None or lon_name is None:
    st.error("Tidak menemukan koordinat latitude/longitude.")
    st.stop()

var_name = st.selectbox("Pilih variabel:", list(ds_tmp.data_vars.keys()))
st.write(f"Koordinat terdeteksi: lat='{lat_name}', lon='{lon_name}'")

# Hitung climatology
try:
    clim = load_and_climatology(ds_tmp, var_name)
except Exception as e:
    st.error(f"Error saat menghitung climatology: {e}")
    st.stop()

st.markdown(f"**Climatology shape:** {clim.shape} (month, lat, lon)")

lats = clim[lat_name].values
lons = clim[lon_name].values
data = clim.values  # (12, nlat, nlon)

# -------------------------
# Flatten grid
# -------------------------
grid_points = []
features = []
for i_lat, lat in enumerate(lats):
    for j_lon, lon in enumerate(lons):
        vals = data[:, i_lat, j_lon]
        if not np.all(np.isnan(vals)):
            grid_points.append((float(lat), float(lon)))
            features.append(vals)

features = np.array(features)
grid_points = np.array(grid_points)
st.write(f"Jumlah grid valid: {features.shape[0]}")

if features.shape[0] == 0:
    st.error("Semua grid NaN. Periksa data Anda.")
    st.stop()

col_mean = np.nanmean(features, axis=0)
inds = np.where(np.isnan(features))
if inds[0].size > 0:
    features[inds] = np.take(col_mean, inds[1])

scaler = StandardScaler()
X = scaler.fit_transform(features)

# -------------------------
# Parameter SOM
# -------------------------
st.subheader("🧠 Konfigurasi SOM")
som_size = st.slider("Ukuran SOM (N × N)", 4, 10, 6)
n_iter = st.slider("Iterasi SOM", 500, 10000, 3000, step=500)

# -------------------------
# Proses SOM
# -------------------------
if st.button("Latih SOM & Analisis"):
    with st.spinner("Melatih SOM..."):
        weights = np.random.RandomState(42).rand(som_size, som_size, X.shape[1])
        alpha = 0.5
        sigma0 = max(som_size, som_size) / 2.0

        for t in range(n_iter):
            xi = X[np.random.randint(0, X.shape[0])]
            dists = np.linalg.norm(weights - xi, axis=2)
            wi, wj = np.unravel_index(np.argmin(dists), dists.shape)
            for i in range(som_size):
                for j in range(som_size):
                    grid_dist = np.sqrt((i - wi)**2 + (j - wj)**2)
                    h = np.exp(-(grid_dist**2) / (2 * (sigma0**2)))
                    weights[i, j] += alpha * h * (xi - weights[i, j])
            alpha = 0.5 * (1 - t / n_iter)
            sigma0 = max(som_size, som_size) / 2.0 * (1 - t / n_iter)

    st.success("✅ Training selesai")

    # Assign cluster
    cluster_ids = []
    for x in X:
        dists = np.linalg.norm(weights - x, axis=2)
        bi, bj = np.unravel_index(np.argmin(dists), dists.shape)
        cluster_ids.append(int(bi * som_size + bj))
    cluster_ids = np.array(cluster_ids)

    # ============================
    # Smoothing spasial
    # ============================
    coords_xy = np.column_stack([
        np.radians(grid_points[:, 1]) * 6371000 * np.cos(np.radians(grid_points[:, 0])),
        np.radians(grid_points[:, 0]) * 6371000
    ])
    tree = cKDTree(coords_xy)
    cluster_smooth = smooth_majority(cluster_ids, coords_xy, tree, r_km=80, k=40)

    # DataFrame hasil
    df = pd.DataFrame({
        'lat': grid_points[:, 0],
        'lon': grid_points[:, 1],
        'cluster_raw': cluster_ids,
        'cluster_smooth': cluster_smooth
    })
    df_months = pd.DataFrame(features, columns=[f'Bulan_{i+1}' for i in range(12)])
    df = pd.concat([df.reset_index(drop=True), df_months.reset_index(drop=True)], axis=1)

    # Simpan CSV
    st.download_button("💾 Download CSV", df.to_csv(index=False).encode("utf-8"), "som_clusters_grid.csv")

    # Statistik cluster
    stats = df.groupby('cluster_smooth').agg(
        {**{f'Bulan_{i+1}': 'mean' for i in range(12)}, 'lat': 'count'}
    ).rename(columns={'lat': 'n_points'}).reset_index().sort_values('n_points', ascending=False)
    st.download_button("📊 Download Statistik Cluster", stats.to_csv(index=False).encode("utf-8"), "cluster_stats.csv")

    # ============================
    # Visualisasi interaktif
    # ============================
    st.subheader("🗺️ Peta Interaktif: Toggle Raw vs Smoothed")
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=df['lat'], lon=df['lon'], mode='markers',
        marker=dict(size=4, color=df['cluster_raw'], colorscale='Viridis'),
        name='Raw'
    ))
    fig.add_trace(go.Scattermapbox(
        lat=df['lat'], lon=df['lon'], mode='markers',
        marker=dict(size=4, color=df['cluster_smooth'], colorscale='Plasma'),
        name='Smoothed', visible=False
    ))
    fig.update_layout(
        mapbox=dict(style='carto-positron', center=dict(lat=-2, lon=120), zoom=4),
        updatemenus=[dict(buttons=[
            dict(label='Raw', method='update', args=[{'visible': [True, False]}]),
            dict(label='Smoothed', method='update', args=[{'visible': [False, True]}])
        ], direction='left', x=0.1, y=1.05)],
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Zonasi cluster
    st.subheader("🗺️ Zonasi Dominan Cluster")
    fig2 = px.scatter_mapbox(df, lat='lat', lon='lon', color='cluster_smooth',
                             color_continuous_scale=px.colors.qualitative.Dark24,
                             hover_data=['lat', 'lon', 'cluster_smooth'], zoom=4, height=650)
    fig2.update_layout(mapbox_style='carto-positron', margin=dict(l=0, r=0, b=0, t=40), mapbox_center=dict(lat=-2, lon=120))
    st.plotly_chart(fig2, use_container_width=True)

    # U-Matrix
    st.subheader("📏 U-Matrix (SOM distance)")
    m, n, d = weights.shape
    U = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            w = weights[i, j]
            neigh = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < n:
                    neigh.append(weights[ii, jj])
            if neigh:
                U[i, j] = np.mean([np.linalg.norm(w - nb) for nb in neigh])
    fig_u, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(U.T, origin='lower')
    ax.set_title("SOM U-Matrix")
    plt.colorbar(im, ax=ax)
    st.pyplot(fig_u)

    # Profil bulanan
    st.subheader("📈 Profil Bulanan: Top 6 Cluster")
    top_clusters = stats['cluster_smooth'].head(6).tolist()
    months = np.arange(1, 13)
    for cid in top_clusters:
        row = stats[stats['cluster_smooth'] == cid].iloc[0]
        vals = [row[f'Bulan_{m}'] for m in months]
        figp, axp = plt.subplots(figsize=(6, 3))
        axp.plot(months, vals, marker='o')
        axp.set_xticks(months)
        axp.set_xlabel('Bulan')
        axp.set_ylabel('Curah Hujan (mm)')
        axp.set_title(f'Cluster {cid} (n={int(row["n_points"])})')
        st.pyplot(figp)

    st.success("✅ Analisis selesai. File CSV & Statistik bisa diunduh.")

