pip install matplotlib pandas numpy seaborn scikit-learn scipy geopandas plotly

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
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.utils import resample

# Set page config
st.set_page_config(
    page_title="MBG Clustering Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Page title
st.title("MBG Clustering Analysis Dashboard")
st.write("Analisis Clustering Wilayah Prioritas Pendistribusian Makan Bergizi Gratis (MBG)")

# Function to load and prepare data
@st.cache_data
def load_data():
    # Load datasets
    kemiskinan = pd.read_excel('kemiskinan.xlsx')
    sekolah = pd.read_excel('Sekolah.xlsx')
    anak_sekolah = pd.read_excel('JumlahAnakSekolah.xlsx')
    IKP = pd.read_excel('IndeksKetahananPangan.xlsx')

    # Merge datasets
    df_merged = kemiskinan.merge(
        sekolah, on='Kode Wilayah', how='outer', suffixes=('_kemiskinan', '_sekolah')
    ).merge(
        anak_sekolah, on='Kode Wilayah', how='outer', suffixes=('', '_anak_sekolah')
    ).merge(
        IKP, on='Kode Wilayah', how='outer', suffixes=('', '_ikp')
    )

    # Replace 0 with NaN
    df_merged = df_merged.replace(0, np.nan)

    # Load Indonesia shapefile (you need to have this file)
    # Load Indonesia shapefile (you need to have this file) 
    # Load Indonesia shapefile (you need to have this file)
    try:
        geo_df = gpd.read_file('gadm41_IDN_2.json')
        # Ganti kolom CC_2 jadi Kode Wilayah
        geo_df.rename(columns={'CC_2': 'Kode Wilayah'}, inplace=True)
        # Pastikan 'Kode Wilayah' adalah string untuk penggabungan yang tepat
        geo_df['Kode Wilayah'] = geo_df['Kode Wilayah'].astype(str)
        
        # Pastikan df_merged['Kode Wilayah'] adalah string untuk penggabungan yang tepat
        df_merged['Kode Wilayah'] = df_merged['Kode Wilayah'].astype(str)

    except Exception as e:
        st.error(f"Error loading shapefile: {e}")
        geo_df = None

    return df_merged, geo_df


# Load data
try:
    df_merged, geo_df = load_data()

    # Display basic data information
    st.write(f"Total Data: {df_merged.shape[0]} wilayah dengan {df_merged.shape[1]} variabel")

    # Get numeric columns (excluding ID columns)
    exclude_cols = ['Kode Wilayah', 'Nama Wilayah_kemiskinan', 'Nama Wilayah_sekolah', 'Nama Wilayah', 'Nama Wilayah_ikp']
    numeric_cols = [col for col in df_merged.columns if col not in exclude_cols and df_merged[col].dtype in ['int64', 'float64']]

    # Sidebar for variable selection
    st.sidebar.header("Pemilihan Variabel")
    st.sidebar.write("Pilih variabel yang akan digunakan untuk analisis clustering:")

    # Group variables by source dataset for better organization
    var_kemiskinan = [col for col in numeric_cols if '_kemiskinan' in col]
    var_sekolah = [col for col in numeric_cols if '_sekolah' in col and '_anak_sekolah' not in col]
    var_anak_sekolah = [col for col in numeric_cols if '_anak_sekolah' in col]
    var_ikp = [col for col in numeric_cols if '_ikp' in col]
    var_other = [col for col in numeric_cols if col not in var_kemiskinan + var_sekolah + var_anak_sekolah + var_ikp]

    # Create checkboxes for variable selection by group
    selected_vars = []

    with st.sidebar.expander("Variabel Kemiskinan", expanded=False):
        for col in var_kemiskinan:
            if st.checkbox(col, value=True, key=f"cb_{col}"):
                selected_vars.append(col)

    with st.sidebar.expander("Variabel Sekolah", expanded=False):
        for col in var_sekolah:
            if st.checkbox(col, value=True, key=f"cb_{col}"):
                selected_vars.append(col)

    with st.sidebar.expander("Variabel Jumlah Anak Sekolah", expanded=False):
        for col in var_anak_sekolah:
            if st.checkbox(col, value=True, key=f"cb_{col}"):
                selected_vars.append(col)

    with st.sidebar.expander("Variabel Indeks Ketahanan Pangan", expanded=False):
        for col in var_ikp:
            if st.checkbox(col, value=True, key=f"cb_{col}"):
                selected_vars.append(col)

    with st.sidebar.expander("Variabel Lainnya", expanded=False):
        for col in var_other:
            if st.checkbox(col, value=True, key=f"cb_{col}"):
                selected_vars.append(col)

    # Clustering parameters
    st.sidebar.header("Parameter Clustering")
    cluster_method = st.sidebar.selectbox(
        "Metode Clustering",
        ["average", "ward", "complete", "single"],
        index=0
    )

    num_clusters = st.sidebar.slider(
        "Jumlah Cluster",
        min_value=2,
        max_value=10,
        value=3
    )

    # Analysis button
    run_analysis = st.sidebar.button("Jalankan Analisis", type="primary")

    # Main content area
    if run_analysis:
        if len(selected_vars) < 2:
            st.error("Silakan pilih minimal 2 variabel untuk clustering!")
        else:
            st.write(f"Menjalankan analisis dengan {len(selected_vars)} variabel terpilih.")

            # Prepare data for clustering
            df_clean = df_merged.dropna(subset=selected_vars)

            # Display number of records after cleaning
            st.write(f"Data bersih: {df_clean.shape[0]} wilayah (setelah menghapus missing values)")

            # Standardize data
            scaler = StandardScaler()
            data_norm = pd.DataFrame(
                scaler.fit_transform(df_clean[selected_vars]),
                columns=selected_vars,
                index=df_clean.index
            )

            # Hierarchical clustering
            with st.spinner("Melakukan clustering..."):
                # Compute linkage matrix
                linkage_matrix = linkage(data_norm, method=cluster_method)

                # Form clusters
                clusters = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

                # Add cluster information to dataframe
                df_clean['Cluster'] = clusters

                # Calculate centroids
                centroids = df_clean.groupby('Cluster')[selected_vars].mean()

                # Calculate cluster validation metrics
                silhouette_avg = silhouette_score(data_norm, clusters)

                # Calculate Sw (within-cluster variation)
                sw_per_cluster = []
                for clust_id in np.unique(clusters):
                    cluster_data = data_norm.iloc[np.where(clusters == clust_id)[0]]
                    if len(cluster_data) > 1:
                        cluster_std = cluster_data.std().mean()
                        sw_per_cluster.append(cluster_std)
                Sw = np.mean(sw_per_cluster) if sw_per_cluster else np.nan

                # Calculate Sb (between-cluster variation)
                cluster_centroids_norm = pd.DataFrame(
                    scaler.transform(centroids),
                    index=centroids.index,
                    columns=centroids.columns
                )
                Sb = cluster_centroids_norm.std().mean() if len(cluster_centroids_norm) > 1 else np.nan

                # Calculate ratio
                ratio = Sw / Sb if not np.isnan(Sw) and not np.isnan(Sb) and Sb != 0 else np.nan

            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Peta Cluster", "Dendrogram", "Visualisasi PCA", "Validasi", "Detail Cluster"])

            with tab1:
                st.header("Peta Cluster Wilayah")

                if geo_df is not None:
                    # Merge cluster results with geographic data
                    # Immediately before the merge operation
                    df_clean['Kode Wilayah'] = df_clean['Kode Wilayah'].astype(str)
                    geo_df['Kode Wilayah'] = geo_df['Kode Wilayah'].astype(str)

                    geo_result = geo_df.merge(
                        df_clean[['Kode Wilayah', 'Cluster', 'Nama Wilayah']],
                        on='Kode Wilayah',
                        how='left'
                    )
                    # Plot using plotly
                    fig = px.choropleth_mapbox(
                        geo_result,
                        geojson=geo_result.geometry,
                        locations=geo_result.index,
                        color='Cluster',
                        color_continuous_scale='viridis',
                        mapbox_style="carto-positron",
                        zoom=4,
                        center={"lat": -2.5, "lon": 118},
                        opacity=0.7,
                        labels={'Cluster':'Cluster'},
                        hover_data=['Nama Wilayah', 'Cluster']
                    )
                    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Map visualization requires Indonesia shapefile.")

                # Show cluster counts
                cluster_counts = df_clean['Cluster'].value_counts().sort_index()
                st.write("Jumlah wilayah per cluster:")
                st.bar_chart(cluster_counts)

            with tab2:
                st.header("Dendrogram")

                # Generate dendrogram
                fig, ax = plt.subplots(figsize=(12, 8))
                dendrogram(
                    linkage_matrix,
                    truncate_mode='lastp',
                    p=50,  # Show only the last p merged clusters
                    leaf_rotation=90,
                    ax=ax
                )
                plt.title(f"Dendrogram (Metode {cluster_method.capitalize()})")
                plt.xlabel("Index")
                plt.ylabel("Distance")
                plt.tight_layout()
                st.pyplot(fig)

                # Option to show full dendrogram
                if st.checkbox("Tampilkan dendrogram lengkap"):
                    fig, ax = plt.subplots(figsize=(20, 10))
                    dendrogram(
                        linkage_matrix,
                        labels=df_clean['Kode Wilayah'].astype(str).values,
                        leaf_rotation=90,
                        ax=ax
                    )
                    plt.title(f"Dendrogram Lengkap (Metode {cluster_method.capitalize()})")
                    plt.xlabel("Kode Wilayah")
                    plt.ylabel("Distance")
                    plt.tight_layout()
                    st.pyplot(fig)

            with tab3:
                st.header("Visualisasi PCA")

                # Perform PCA
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data_norm)

                # Create DataFrame for plotting
                pca_df = pd.DataFrame(
                    data=pca_result,
                    columns=['PC1', 'PC2']
                )
                pca_df['Cluster'] = clusters
                pca_df['Nama Wilayah'] = df_clean['Nama Wilayah'].values
                pca_df['Kode Wilayah'] = df_clean['Kode Wilayah'].values

                # Plot with Plotly
                fig = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    hover_data=['Nama Wilayah', 'Kode Wilayah'],
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    title=f"PCA - Visualisasi Cluster ({cluster_method.capitalize()})",
                    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                           'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'}
                )

                # Add centroid markers
                for cluster in pca_df['Cluster'].unique():
                    cluster_data = pca_df[pca_df['Cluster'] == cluster]
                    centroid_x = cluster_data['PC1'].mean()
                    centroid_y = cluster_data['PC2'].mean()
                    fig.add_trace(
                        go.Scatter(
                            x=[centroid_x],
                            y=[centroid_y],
                            mode='markers',
                            marker=dict(
                                symbol='x',
                                size=15,
                                color='black',
                                line=dict(width=2)
                            ),
                            name=f'Centroid Cluster {cluster}',
                            showlegend=True
                        )
                    )

                st.plotly_chart(fig, use_container_width=True)

                # Explained variance ratio
                st.write(f"PC1 explained variance: {pca.explained_variance_ratio_[0]:.2%}")
                st.write(f"PC2 explained variance: {pca.explained_variance_ratio_[1]:.2%}")

                # Feature importance
                loading_scores = pd.DataFrame(
                    data=pca.components_.T * np.sqrt(pca.explained_variance_),
                    columns=['PC1', 'PC2'],
                    index=selected_vars
                )
                loading_scores['abs_pc1'] = abs(loading_scores['PC1'])
                loading_scores = loading_scores.sort_values('abs_pc1', ascending=False)

                st.write("Feature importance (loading scores):")
                st.dataframe(loading_scores[['PC1', 'PC2']])

            with tab4:
                st.header("Validasi Cluster")

                # Display validation metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Silhouette Score", f"{silhouette_avg:.4f}")
                with col2:
                    st.metric("Within-cluster variation (Sw)", f"{Sw:.4f}" if not np.isnan(Sw) else "N/A")
                with col3:
                    st.metric("Between-cluster variation (Sb)", f"{Sb:.4f}" if not np.isnan(Sb) else "N/A")

                st.write(f"Sw/Sb Ratio: {ratio:.4f}" if not np.isnan(ratio) else "N/A")
                st.write("Semakin kecil nilai rasio Sw/Sb menunjukkan kualitas cluster yang lebih baik")

                # Python implementation of bootstrap validation
                st.write("### Bootstrap Validation")
                st.info("Catatan: Proses ini mungkin membutuhkan waktu yang cukup lama tergantung ukuran data")

                if st.button("Jalankan Bootstrap Validation"):
                    with st.spinner("Menjalankan Bootstrap... (ini mungkin membutuhkan waktu)"):
                        # Function to compute Jaccard similarity between two cluster assignments
                        def jaccard_similarity(clusters1, clusters2):
                            # Create pairs for each clustering
                            def create_pairs(clusters):
                                n = len(clusters)
                                pairs = set()
                                for i in range(n):
                                    for j in range(i+1, n):
                                        if clusters[i] == clusters[j]:
                                            pairs.add((i, j))
                                return pairs

                            pairs1 = create_pairs(clusters1)
                            pairs2 = create_pairs(clusters2)

                            # Calculate Jaccard similarity
                            intersection = len(pairs1.intersection(pairs2))
                            union = len(pairs1.union(pairs2))

                            return intersection / union if union > 0 else 0

                        # Parameters
                        n_bootstrap = 100  # Number of bootstrap samples

                        # Original clustering
                        orig_clusters = clusters

                        # Initialize storage for stability scores
                        stability_scores = []

                        # Progress bar
                        progress_bar = st.progress(0)

                        # Run bootstrap iterations
                        for i in range(n_bootstrap):
                            # Bootstrap resample
                            bootstrap_indices = resample(range(len(data_norm)), replace=True, n_samples=len(data_norm))
                            bootstrap_data = data_norm.iloc[bootstrap_indices]

                            # Compute linkage for bootstrap sample
                            try:
                                bootstrap_linkage = linkage(bootstrap_data, method=cluster_method)

                                # Form clusters
                                bootstrap_clusters = fcluster(bootstrap_linkage, t=num_clusters, criterion='maxclust')

                                # Map bootstrap clusters back to original indices
                                mapped_clusters = np.zeros(len(data_norm))
                                mapped_clusters[:] = np.nan  # Initialize with NaN

                                for j, orig_idx in enumerate(bootstrap_indices):
                                    mapped_clusters[orig_idx] = bootstrap_clusters[j]

                                # Only compare elements that were selected in the bootstrap
                                valid_indices = ~np.isnan(mapped_clusters)
                                if sum(valid_indices) > 1:  # Need at least 2 valid points
                                    orig_subset = orig_clusters[valid_indices]
                                    bootstrap_subset = mapped_clusters[valid_indices]

                                    # Calculate Jaccard similarity
                                    similarity = jaccard_similarity(orig_subset, bootstrap_subset)
                                    stability_scores.append(similarity)
                            except Exception as e:
                                st.warning(f"Bootstrap iteration {i+1} failed: {e}")

                            # Update progress
                            progress_bar.progress((i + 1) / n_bootstrap)

                        # Calculate average stability score
                        if stability_scores:
                            avg_stability = np.mean(stability_scores)

                            # Plot histogram of stability scores
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.hist(stability_scores, bins=20, alpha=0.7)
                            ax.axvline(avg_stability, color='red', linestyle='dashed', linewidth=2)
                            ax.set_xlabel('Jaccard Similarity')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Distribution of Cluster Stability Scores')
                            plt.tight_layout()

                            st.pyplot(fig)

                            # Interpret results
                            st.write(f"### Hasil Bootstrap Validation")
                            st.write(f"Rata-rata Jaccard Similarity: {avg_stability:.4f}")

                            # Interpretation guide
                            if avg_stability >= 0.75:
                                st.success("Cluster sangat stabil (>= 0.75)")
                            elif avg_stability >= 0.6:
                                st.info("Cluster cukup stabil (0.6 - 0.75)")
                            elif avg_stability >= 0.4:
                                st.warning("Cluster kurang stabil (0.4 - 0.6)")
                            else:
                                st.error("Cluster tidak stabil (< 0.4)")

                            st.write("""
                            **Interpretasi Jaccard Similarity:**
                            - 1.0 = Identik (sempurna)
                            - > 0.75 = Sangat mirip
                            - 0.5 - 0.75 = Cukup mirip
                            - < 0.5 = Berbeda signifikan
                            """)
                        else:
                            st.error("Tidak ada hasil bootstrap yang valid")

            with tab5:
                st.header("Detail Cluster")

                # Show summary of each cluster
                for cluster_id in sorted(df_clean['Cluster'].unique()):
                    with st.expander(f"Cluster {cluster_id} ({(df_clean['Cluster'] == cluster_id).sum()} wilayah)"):
                        cluster_data = df_clean[df_clean['Cluster'] == cluster_id]

                        # Show list of regions in this cluster
                        st.write("### Daftar Wilayah")
                        st.dataframe(
                            cluster_data[['Kode Wilayah', 'Nama Wilayah']].sort_values('Nama Wilayah'),
                            hide_index=True
                        )

                        # Characteristics of this cluster (compared to other clusters)
                        st.write("### Karakteristik Cluster")

                        # Calculate z-scores of centroids relative to overall mean
                        overall_mean = df_clean[selected_vars].mean()
                        overall_std = df_clean[selected_vars].std()

                        cluster_mean = cluster_data[selected_vars].mean()
                        z_scores = (cluster_mean - overall_mean) / overall_std

                        # Display as a horizontal bar chart
                        z_df = pd.DataFrame({'z-score': z_scores}).sort_values('z-score')

                        fig = px.bar(
                            z_df,
                            x='z-score',
                            y=z_df.index,
                            orientation='h',
                            title=f"Karakteristik Cluster {cluster_id} (Z-scores)",
                            color='z-score',
                            color_continuous_scale=px.colors.diverging.RdBu_r,
                            range_color=[-3, 3]
                        )
                        fig.add_vline(x=0, line_dash="dash", line_color="black")
                        st.plotly_chart(fig, use_container_width=True)

                        # Raw centroid values
                        st.write("### Nilai Rata-rata")
                        centroid_df = pd.DataFrame({
                            'Nilai Rata-rata': cluster_mean,
                            'Rata-rata Keseluruhan': overall_mean,
                            'Perbedaan (%)': ((cluster_mean - overall_mean) / overall_mean * 100).round(2)
                        })
                        st.dataframe(centroid_df)

                # Download options
                st.write("### Download Hasil")

                @st.cache_data
                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df(df_clean)
                st.download_button(
                    "Download hasil clustering (CSV)",
                    csv,
                    "mbg_clustering_results.csv",
                    "text/csv",
                    key='download-csv'
                )
    else:
        # Display default information when first loading
        st.info("Pilih variabel dan parameter clustering di sidebar, lalu klik 'Jalankan Analisis' untuk melihat hasil.")

        # Display data preview
        st.write("### Preview Data")
        st.dataframe(df_merged.head())

        # Missing values information
        st.write("### Informasi Missing Values")
        missing = df_merged.isnull().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if len(missing) > 0:
            missing_df = pd.DataFrame({
                'Variabel': missing.index,
                'Jumlah Missing': missing.values,
                'Persentase (%)': (missing.values / len(df_merged) * 100).round(2)
            })
            st.dataframe(missing_df)
        else:
            st.write("Tidak ada missing values dalam dataset.")

except Exception as e:
    st.error(f"Error: {e}")
    st.write("Pastikan semua file data (kemiskinan.xlsx, Sekolah.xlsx, JumlahAnakSekolah.xlsx, IndeksKetahananPangan.xlsx) tersedia.")
