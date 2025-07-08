import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
import geopandas as gpd
import plotly.express as px
from functools import reduce

# ==============================================================================
# Konfigurasi Halaman dan Judul
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Analisis Klaster MBG",
    page_icon="📊",
    layout="wide"
)

st.title("Dashboard Analisis Klaster MBG")
st.write("Visualisasi Interaktif untuk Analisis Klaster Wilayah Prioritas Distribusi Makan Bergizi Gratis (MBG)")

# ==============================================================================
# Inisialisasi Session State
# ==============================================================================
# Ini untuk memastikan state tidak hilang saat interaksi
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# ==============================================================================
# Fungsi-fungsi Inti
# ==============================================================================

# Fungsi ini hanya memuat data mentah, tanpa proses apa pun.
@st.cache_data
def load_raw_data():
    try:
        kemiskinan = pd.read_excel('kemiskinan.xlsx')
        sekolah = pd.read_excel('Sekolah.xlsx')
        anak_sekolah = pd.read_excel('JumlahAnakSekolah.xlsx')
        ikp = pd.read_excel('IndeksKetahananPangan.xlsx')
        return kemiskinan, sekolah, anak_sekolah, ikp
    except FileNotFoundError as e:
        st.error(f"Error: File tidak ditemukan -> {e}. Pastikan semua file Excel ada di direktori aplikasi.")
        return None, None, None, None

@st.cache_data
def load_geo_data():
    try:
        geo_df = gpd.read_file('gadm41_IDN_2.json')
        geo_df.rename(columns={'CC_2': 'Kode Wilayah'}, inplace=True)
        geo_df['Kode Wilayah'] = geo_df['Kode Wilayah'].astype(str)
        return geo_df
    except Exception:
        st.warning("Gagal memuat file peta `gadm41_IDN_2.json`. Fitur peta tidak akan tersedia.")
        return None

# Fungsi ini menjalankan SEMUA proses, dari persiapan data hingga analisis.
def run_full_analysis(raw_data, selected_vars, cluster_method, num_clusters):
    kemiskinan_raw, sekolah_raw, anak_sekolah_raw, ikp_raw = raw_data
    
    # 1. Persiapan Data (Logika dari Colab)
    dataframes_to_merge = []
    # Selalu sertakan Kode Wilayah dan Nama Wilayah
    df_base = kemiskinan_raw[['Kode Wilayah', 'Nama Wilayah']]
    dataframes_to_merge.append(df_base)

    # Pilih kolom berdasarkan variabel yang dicentang
    if 'Persentase_Penduduk_Miskin' in selected_vars:
        dataframes_to_merge.append(kemiskinan_raw[['Kode Wilayah', 'Persentase_Penduduk_Miskin']])
    if 'Indeks_Kedalaman_Kemiskinan' in selected_vars:
        dataframes_to_merge.append(kemiskinan_raw[['Kode Wilayah', 'Indeks_Kedalaman_Kemiskinan']])
    if 'IKP' in selected_vars:
        dataframes_to_merge.append(ikp_raw[['Kode Wilayah', 'IKP']])
    if 'Total_Jumlah' in selected_vars:
        dataframes_to_merge.append(sekolah_raw[['Kode Wilayah', 'Total_Jumlah']])
    if 'SD_Jumlah' in selected_vars:
        dataframes_to_merge.append(sekolah_raw[['Kode Wilayah', 'SD_Jumlah']])
    if 'SMP_Jumlah' in selected_vars:
        dataframes_to_merge.append(sekolah_raw[['Kode Wilayah', 'SMP_Jumlah']])
    if 'Total_LP' in selected_vars:
        dataframes_to_merge.append(anak_sekolah_raw[['Kode Wilayah', 'Total_LP']])
    if 'SD_LP' in selected_vars:
        dataframes_to_merge.append(anak_sekolah_raw[['Kode Wilayah', 'SD_LP']])
    if 'SMP_LP' in selected_vars:
        dataframes_to_merge.append(anak_sekolah_raw[['Kode Wilayah', 'SMP_LP']])

    # Gabungkan hanya dataframe yang relevan
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='Kode Wilayah', how='inner'), dataframes_to_merge)
    
    # Bersihkan data
    df_merged.replace(0, np.nan, inplace=True)
    df_clean = df_merged.dropna().reset_index(drop=True)

    # 2. Proses Clustering
    data_numerik = df_clean[selected_vars]
    scaler = StandardScaler()
    data_norm = pd.DataFrame(scaler.fit_transform(data_numerik), columns=selected_vars, index=df_clean.index)

    linkage_matrix = linkage(data_norm, method=cluster_method)
    clusters = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')
    df_clean['Cluster'] = clusters

    # 3. Hitung Metrik Validasi
    silhouette_avg = silhouette_score(data_norm, clusters)
    
    data_norm_temp = data_norm.copy()
    data_norm_temp['Cluster'] = clusters
    
    # PERBAIKAN LOGIKA Sw: Menghitung std dari semua nilai dalam cluster, lalu dirata-rata
    sw_per_cluster = [cluster_data.values.std() for _, cluster_data in data_norm_temp.groupby('Cluster')[selected_vars] if len(cluster_data) > 1]
    Sw = np.mean(sw_per_cluster) if sw_per_cluster else np.nan

    # PERBAIKAN LOGIKA Sb: Menghitung std dari semua nilai centroid
    cluster_centroids_norm = data_norm_temp.groupby('Cluster')[selected_vars].mean()
    Sb = cluster_centroids_norm.values.std() if len(cluster_centroids_norm) > 1 else np.nan
    
    ratio = Sw / Sb if not np.isnan(Sw) and not np.isnan(Sb) and Sb != 0 else np.nan

    # 4. Simpan semua hasil ke dictionary
    return {
        "df_final": df_clean,
        "linkage_matrix": linkage_matrix,
        "silhouette": silhouette_avg,
        "Sw": Sw,
        "Sb": Sb,
        "ratio": ratio,
        "centroids": df_clean.groupby('Cluster')[selected_vars].mean()
    }

# ==============================================================================
# Sidebar dan Kontrol Utama
# ==============================================================================
st.sidebar.header("Parameter Analisis")

# Variabel yang tersedia untuk dipilih
ALL_VARS = [
    'Persentase_Penduduk_Miskin', 'Indeks_Kedalaman_Kemiskinan', 'Total_Jumlah',
    'SD_Jumlah', 'SMP_Jumlah', 'Total_LP', 'SD_LP', 'SMP_LP', 'IKP'
]
# Variabel default yang dicentang
DEFAULT_VARS = ['Persentase_Penduduk_Miskin', 'Total_Jumlah', 'Total_LP', 'IKP']

with st.sidebar.expander("Pilih Variabel untuk Analisis", expanded=True):
    selected_vars = [var for var in ALL_VARS if st.checkbox(var, value=(var in DEFAULT_VARS), key=f"cb_{var}")]

cluster_method = st.sidebar.selectbox("Pilih Metode Clustering", ["average", "ward", "complete", "single"], index=0)
num_clusters = st.sidebar.slider("Pilih Jumlah Cluster", 2, 10, 5)

if st.sidebar.button("Jalankan Analisis", type="primary"):
    if len(selected_vars) < 2:
        st.sidebar.error("Pilih minimal 2 variabel.")
    else:
        raw_data = load_raw_data()
        if all(d is not None for d in raw_data):
            with st.spinner("Menjalankan analisis..."):
                st.session_state.results = run_full_analysis(raw_data, selected_vars, cluster_method, num_clusters)
                st.session_state.analysis_run = True
        else:
            st.session_state.analysis_run = False

# ==============================================================================
# Tampilan Hasil
# ==============================================================================
if st.session_state.analysis_run:
    results = st.session_state.results
    df_final = results["df_final"]
    
    st.success(f"Analisis selesai. Ditemukan {len(df_final)} wilayah yang dikelompokkan menjadi {num_clusters} cluster.")

    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Peta Cluster", "🌳 Dendrogram", "📊 Validasi & Karakteristik", "📋 Detail Anggota"])

    with tab1:
        st.header("Peta Sebaran Cluster Wilayah")
        geo_df = load_geo_data()
        if geo_df is not None:
            df_final['Kode Wilayah'] = df_final['Kode Wilayah'].astype(str)
            geo_df['Kode Wilayah'] = geo_df['Kode Wilayah'].astype(str)
            geo_result = geo_df.merge(df_final[['Kode Wilayah', 'Nama Wilayah', 'Cluster']], on='Kode Wilayah', how='inner')
            
            fig = px.choropleth_mapbox(geo_result, geojson=geo_result.geometry, locations=geo_result.index,
                                       color='Cluster', color_continuous_scale='viridis',
                                       mapbox_style="carto-positron", zoom=4, center={"lat": -2.5, "lon": 118},
                                       opacity=0.7, hover_name="Nama Wilayah", hover_data={'Cluster': True})
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header(f"Dendrogram - Metode {cluster_method.capitalize()}")
        fig, ax = plt.subplots(figsize=(15, 8))
        dendrogram(results["linkage_matrix"], truncate_mode='lastp', p=50, leaf_rotation=90, ax=ax)
        plt.title("Dendrogram (Menampilkan 50 Gabungan Terakhir)", fontsize=16)
        st.pyplot(fig)

        # Checkbox ini sekarang berfungsi tanpa mereset aplikasi
        if st.checkbox("Tampilkan dendrogram lengkap", key="show_full_dendrogram"):
            with st.spinner("Menggambar dendrogram lengkap..."):
                fig_full, ax_full = plt.subplots(figsize=(20, 10))
                dendrogram(
                    results["linkage_matrix"],
                    labels=df_final['Nama Wilayah'].values,
                    leaf_rotation=90,
                    leaf_font_size=8,
                    ax=ax_full
                )
                plt.title("Dendrogram Lengkap", fontsize=16)
                st.pyplot(fig_full)

    with tab3:
        st.header("Validasi dan Karakteristik Cluster")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Silhouette Score", f"{results['silhouette']:.4f}")
        col2.metric("Sw (Variasi Internal)", f"{results['Sw']:.4f}")
        col3.metric("Sb (Variasi Eksternal)", f"{results['Sb']:.4f}")
        col4.metric("Rasio Sw/Sb", f"{results['ratio']:.4f}", help="Nilai lebih kecil lebih baik")
        
        st.subheader("Karakteristik Rata-rata per Cluster (Centroid)")
        st.dataframe(results["centroids"].style.background_gradient(cmap='viridis', axis=0))

    with tab4:
        st.header("Detail Anggota per Cluster")
        for cluster_id in sorted(df_final['Cluster'].unique()):
            cluster_data = df_final[df_final['Cluster'] == cluster_id]
            with st.expander(f"Cluster {cluster_id} ({len(cluster_data)} wilayah)"):
                st.dataframe(cluster_data[['Kode Wilayah', 'Nama Wilayah']].sort_values('Nama Wilayah'),
                             use_container_width=True, hide_index=True)
else:
    st.info("Pilih parameter di sidebar dan klik 'Jalankan Analisis' untuk memulai.")
