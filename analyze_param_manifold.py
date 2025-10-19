import json
import os
# ABM Parameter Space Analysis - Enhanced Version
# Adds: Convex hull volume, distance to centroid, clustering analysis, and better visualization

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, distance
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def get_config(directory,prefix_pattern,suffix_pattern,index):
    config_file = f'{directory}/{prefix_pattern}{index}{suffix_pattern}.json'
    #print(config_file)
    if not os.path.isfile(config_file):
        #print('trying by accumulating all files')
        all_files = os.listdir(directory)
        possible_files = []
        for f in all_files:
            #print(f)
            if f'sim_{index}.json' in f:
                #print(f'{index} found in {f}')
                possible_files.append(f)
        #print(f'found possible {len(possible_files)} files')
        if len(possible_files)==1:
            config_file = f'{directory}/{possible_files[0]}'
        else:
            return None
            
    
    param_names = [
            "param_migration_bias",
            "param_distance_decay", 
            "param_discount_rate",
            "param_risk_growth_rate",
            "param_threshold_hi",
            "param_lambda1",
            "param_lambda2",
            "refugee_among_displaced"
    ]
    
    with open(config_file) as f:
        config = json.load(f)
    
    param_vectors = [config[p] for p in config if p in param_names]
    return param_vectors
    
# ----------------------------- Utilities -----------------------------

def scale_arrays(A, B, scaling_params, scale_together=True):
    """Standard scale columns. If scale_together=True, fit scaler on A then transform both A and B.
    Otherwise scale each array independently."""
    if scale_together:
        scaler = StandardScaler().fit(scaling_params)
        return scaler.transform(A), scaler.transform(B)
    else:
        return StandardScaler().fit_transform(A), StandardScaler().fit_transform(B)


def pairwise_stats(X, metric='euclidean'):
    D = pdist(X, metric=metric)
    statsd = {
        'n_pairs': len(D),
        'mean': D.mean(),
        'median': np.median(D),
        'std': D.std(),
        'min': D.min(),
        'max': D.max(),
        'cv': D.std()/D.mean() if D.mean()!=0 else np.nan,
        'Dvec': D
    }
    return statsd

# ---------------------- Distance-based comparisons ----------------------

def compare_pairwise(A, B, metric='euclidean', plot=True, fig_prefix='fig'):
    """Compute pairwise summaries, KS, Wasserstein, Spearman between flattened pairwise distances.
    Returns dictionary of results and (optionally) plots."""
    sA = pairwise_stats(A, metric=metric)
    sB = pairwise_stats(B, metric=metric)

    # two-sample KS
    ks = stats.ks_2samp(sA['Dvec'], sB['Dvec'])
    # Wasserstein
    w = wasserstein_distance(sA['Dvec'], sB['Dvec'])
    # Spearman between pairs
    rho, pval = stats.spearmanr(sA['Dvec'], sB['Dvec'])
    
    # Effect size: Cohen's d for distance distributions
    cohens_d = (sB['mean'] - sA['mean']) / np.sqrt((sA['std']**2 + sB['std']**2) / 2)

    results = {
        'A_stats': sA, 
        'B_stats': sB, 
        'ks': ks, 
        'wasserstein': w, 
        'spearman': (rho, pval),
        'cohens_d': cohens_d,
        'compression_ratio': sB['mean'] / sA['mean']  # How much smaller B is
    }

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # KDE plot
        axes[0].hist(sA['Dvec'], bins=30, alpha=0.5, label='Initial', density=True)
        axes[0].hist(sB['Dvec'], bins=30, alpha=0.5, label='Fitted', density=True)
        axes[0].axvline(sA['mean'], color='blue', linestyle='--', label=f"Initial mean: {sA['mean']:.3f}")
        axes[0].axvline(sB['mean'], color='orange', linestyle='--', label=f"Fitted mean: {sB['mean']:.3f}")
        axes[0].legend()
        axes[0].set_xlabel('Pairwise Distance')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distance Distributions')
        
        # Scatter plot
        axes[1].scatter(sA['Dvec'], sB['Dvec'], s=5, alpha=0.3)
        axes[1].plot([0, max(sA['Dvec'])], [0, max(sA['Dvec'])], 'r--', label='y=x')
        axes[1].set_xlabel('Initial distances')
        axes[1].set_ylabel('Fitted distances')
        axes[1].set_title(f'Distance correlation (ρ={rho:.3f})')
        axes[1].legend()
        
        # Box plot comparison
        axes[2].boxplot([sA['Dvec'], sB['Dvec']], labels=['Initial', 'Fitted'])
        axes[2].set_ylabel('Distance')
        axes[2].set_title('Distance Distribution Comparison')
        
        plt.tight_layout()
        plt.savefig(f'Figures/{fig_prefix}_pairwise.png', dpi=150)
        plt.close()
    
    return results

# ------------------------ NEW: Convex Hull Volume Analysis ------------------------

def convex_hull_volume(X):
    """Compute convex hull volume. Handle edge cases."""
    try:
        hull = ConvexHull(X)
        return hull.volume
    except:
        # If can't compute (e.g., degenerate), return 0
        return 0.0

def volume_compression_analysis(A, B):
    """Compare convex hull volumes between A and B"""
    vol_A = convex_hull_volume(A)
    vol_B = convex_hull_volume(B)
    
    compression_ratio = vol_B / vol_A if vol_A > 0 else np.nan
    
    return {
        'volume_A': vol_A,
        'volume_B': vol_B,
        'compression_ratio': compression_ratio
    }

# ------------------------ NEW: Distance to Centroid ------------------------

def centroid_analysis(A, B):
    """Analyze distances from points to centroid"""
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    
    dist_to_centroid_A = np.linalg.norm(A - centroid_A, axis=1)
    dist_to_centroid_B = np.linalg.norm(B - centroid_B, axis=1)
    
    # Statistics
    stats_A = {
        'mean': dist_to_centroid_A.mean(),
        'std': dist_to_centroid_A.std(),
        'median': np.median(dist_to_centroid_A),
        'distances': dist_to_centroid_A
    }
    
    stats_B = {
        'mean': dist_to_centroid_B.mean(),
        'std': dist_to_centroid_B.std(),
        'median': np.median(dist_to_centroid_B),
        'distances': dist_to_centroid_B
    }
    
    # Test if B is more concentrated
    ks_test = stats.ks_2samp(dist_to_centroid_A, dist_to_centroid_B)
    
    # Compression ratio
    compression = stats_B['mean'] / stats_A['mean']
    
    return {
        'A_stats': stats_A,
        'B_stats': stats_B,
        'ks_test': ks_test,
        'compression_ratio': compression,
        'centroid_displacement': np.linalg.norm(centroid_B - centroid_A)
    }

# ------------------------ NEW: Clustering Analysis ------------------------

def clustering_analysis(A, B, eps_factor=0.5):
    """Use DBSCAN to detect if fitted params form distinct clusters"""
    # Adaptive epsilon based on median k-NN distance
    def adaptive_eps(X, k=3):
        nn = NearestNeighbors(n_neighbors=k+1).fit(X)
        distances, _ = nn.kneighbors(X)
        return np.median(distances[:, k])
    
    eps_A = adaptive_eps(A) * eps_factor
    eps_B = adaptive_eps(B) * eps_factor
    
    db_A = DBSCAN(eps=eps_A, min_samples=2).fit(A)
    db_B = DBSCAN(eps=eps_B, min_samples=2).fit(B)
    
    n_clusters_A = len(set(db_A.labels_)) - (1 if -1 in db_A.labels_ else 0)
    n_clusters_B = len(set(db_B.labels_)) - (1 if -1 in db_B.labels_ else 0)
    
    noise_ratio_A = np.sum(db_A.labels_ == -1) / len(db_A.labels_)
    noise_ratio_B = np.sum(db_B.labels_ == -1) / len(db_B.labels_)
    
    return {
        'n_clusters_A': n_clusters_A,
        'n_clusters_B': n_clusters_B,
        'noise_ratio_A': noise_ratio_A,
        'noise_ratio_B': noise_ratio_B,
        'labels_A': db_A.labels_,
        'labels_B': db_B.labels_
    }

# ------------------------ Nearest-neighbor overlap ------------------------

def knn_overlap(A, B, k=5):
    """Compute, for each point i, the Jaccard overlap between the k-NN sets in A and B."""
    n = A.shape[0]
    nnA = NearestNeighbors(n_neighbors=k+1).fit(A)
    nnB = NearestNeighbors(n_neighbors=k+1).fit(B)
    idxA = nnA.kneighbors(return_distance=False)[:,1:]
    idxB = nnB.kneighbors(return_distance=False)[:,1:]

    overlaps = []
    for i in range(n):
        setA = set(idxA[i].tolist())
        setB = set(idxB[i].tolist())
        inter = len(setA & setB)
        union = len(setA | setB)
        overlaps.append(inter/union if union>0 else 0.0)
    return {
        'overlaps': np.array(overlaps), 
        'mean_overlap': np.mean(overlaps), 
        'median_overlap': np.median(overlaps)
    }

# ------------------------ PCA & effective dimension ------------------------

def effective_dim_via_pca(X, n_components=None):
    pca = PCA(n_components=n_components).fit(X)
    lamb = pca.explained_variance_
    d_eff = (lamb.sum()**2) / (np.sum(lamb**2))
    return {
        'pca': pca, 
        'd_eff': d_eff, 
        'explained_ratio': pca.explained_variance_ratio_
    }

# ------------------------ Intrinsic dimension (Levina & Bickel MLE) ------------------------

def levina_bickel_intrinsic_dim(X, k=10):
    """Estimate intrinsic dimension using Levina & Bickel (2004) MLE method."""
    n, d = X.shape
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    dist_idx = nbrs.kneighbors(return_distance=True)
    distances = dist_idx[0][:,1:]
    
    eps = 1e-12
    mles = []
    for i in range(n):
        r = distances[i]
        r_k = r[-1] + eps
        logs = np.log(r_k / (r[:-1] + eps))
        if np.sum(logs) == 0:
            continue
        mle_i = (k-1) / np.sum(logs)
        mles.append(mle_i)
    
    return {
        'id_est': np.mean(mles), 
        'id_std': np.std(mles),
        'local_mles': np.array(mles)
    }

# ------------------------ Mahalanobis distances ------------------------

def pairwise_mahalanobis(X, VI=None):
    if VI is None:
        cov = np.cov(X, rowvar=False)
        try:
            VI = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            VI = np.linalg.pinv(cov)
    
    from scipy.spatial.distance import mahalanobis
    n = X.shape[0]
    Dm = []
    for i in range(n):
        for j in range(i+1, n):
            Dm.append(mahalanobis(X[i], X[j], VI))
    return np.array(Dm)

# ------------------------ NEW: Comprehensive Plotting ------------------------

def create_comprehensive_plots(results, fig_prefix='analysis'):
    """Create publication-quality plots"""
    
    # Plot 1: PCA comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scree plots
    pA = results['pca_A']
    pB = results['pca_B']
    
    axes[0, 0].plot(np.cumsum(pA['explained_ratio'])[:8], 'o-', label='Initial', linewidth=2)
    axes[0, 0].plot(np.cumsum(pB['explained_ratio'])[:8], 's-', label='Fitted', linewidth=2)
    axes[0, 0].axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90% variance')
    axes[0, 0].axhline(0.95, color='purple', linestyle='--', alpha=0.5, label='95% variance')
    axes[0, 0].set_xlabel('Number of Components')
    axes[0, 0].set_ylabel('Cumulative Explained Variance')
    axes[0, 0].set_title('Dimensionality Reduction')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Distance distributions
    sA = results['pairwise_compare']['A_stats']
    sB = results['pairwise_compare']['B_stats']
    
    axes[0, 1].violinplot([sA['Dvec'], sB['Dvec']], positions=[0, 1], showmeans=True)
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(['Initial', 'Fitted'])
    axes[0, 1].set_ylabel('Pairwise Distance')
    axes[0, 1].set_title(f"Compression Ratio: {results['pairwise_compare']['compression_ratio']:.3f}")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Centroid distances
    cent = results['centroid_analysis']
    axes[1, 0].violinplot([cent['A_stats']['distances'], cent['B_stats']['distances']], 
                          positions=[0, 1], showmeans=True)
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(['Initial', 'Fitted'])
    axes[1, 0].set_ylabel('Distance to Centroid')
    axes[1, 0].set_title(f"Centroid Compression: {cent['compression_ratio']:.3f}")
    axes[1, 0].grid(True, alpha=0.3)
    
    # k-NN overlap
    knn = results['knn_overlap']
    axes[1, 1].hist(knn['overlaps'], bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(knn['mean_overlap'], color='red', linestyle='--', 
                       label=f"Mean: {knn['mean_overlap']:.3f}", linewidth=2)
    axes[1, 1].set_xlabel('k-NN Jaccard Overlap')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Neighborhood Preservation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Figures/{fig_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Summary metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = {
        'Pairwise Distance': results['pairwise_compare']['compression_ratio'],
        'Centroid Distance': results['centroid_analysis']['compression_ratio'],
        'Convex Hull Volume': results['volume_analysis']['compression_ratio'],
        'Effective Dimension': results['pca_B']['d_eff'] / results['pca_A']['d_eff'],
        'Intrinsic Dimension': results['id_B']['id_est'] / results['id_A']['id_est']
    }
    
    colors = ['#2ecc71' if v < 1 else '#e74c3c' for v in metrics.values()]
    bars = ax.barh(list(metrics.keys()), list(metrics.values()), color=colors, alpha=0.7)
    ax.axvline(1.0, color='black', linestyle='--', linewidth=2, label='No change')
    ax.set_xlabel('Ratio (Fitted / Initial)')
    ax.set_title('Parameter Space Compression Metrics\n(Values < 1.0 indicate compression)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                ha='left' if width < 1 else 'right', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'Figures/{fig_prefix}_metrics_summary.png', dpi=300, bbox_inches='tight')
    #plt.close()

# ------------------------ Run all analyses ------------------------

def run_all(A, B, all_param, scale_together=True, fig_prefix='abm'):
    """Run comprehensive parameter space analysis"""
    print("="*60)
    print("PARAMETER SPACE COMPRESSION ANALYSIS")
    print("="*60)
    
    A_s, B_s = scale_arrays(A, B, all_param, scale_together=scale_together)
    
    print("\n1. Computing pairwise distance statistics...")
    comp = compare_pairwise(A_s, B_s, plot=True, fig_prefix=fig_prefix)
    
    print("2. Computing convex hull volumes...")
    vol = volume_compression_analysis(A_s, B_s)
    
    print("3. Analyzing distances to centroid...")
    cent = centroid_analysis(A_s, B_s)
    
    print("4. Computing k-NN overlap...")
    knn = knn_overlap(A_s, B_s, k=3)
    
    print("5. Running PCA and effective dimension...")
    pA = effective_dim_via_pca(A_s)
    pB = effective_dim_via_pca(B_s)
    
    print("6. Estimating intrinsic dimensions...")
    idA = levina_bickel_intrinsic_dim(A_s, k=3)
    idB = levina_bickel_intrinsic_dim(B_s, k=3)
    
    print("7. Clustering analysis...")
    clust = clustering_analysis(A_s, B_s)
    
    print("8. Computing Mahalanobis distances...")
    DmA = pairwise_mahalanobis(A_s)
    DmB = pairwise_mahalanobis(B_s)
    
    results = {
        'pairwise_compare': comp,
        'volume_analysis': vol,
        'centroid_analysis': cent,
        'knn_overlap': knn,
        'pca_A': pA, 
        'pca_B': pB,
        'id_A': idA, 
        'id_B': idB,
        'clustering': clust,
        'mahalanobis_A': DmA, 
        'mahalanobis_B': DmB
    }
    
    print("\n9. Creating comprehensive plots...")
    create_comprehensive_plots(results, fig_prefix=fig_prefix)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    return results

def print_summary(results):
    """Print a concise summary of key findings"""
    print("\n" + "="*60)
    print("KEY FINDINGS SUMMARY")
    print("="*60)
    
    print("\n📊 SPACE COMPRESSION METRICS:")
    print(f"   • Pairwise distance compression: {results['pairwise_compare']['compression_ratio']:.3f}x")
    print(f"   • Centroid distance compression: {results['centroid_analysis']['compression_ratio']:.3f}x")
    print(f"   • Convex hull volume compression: {results['volume_analysis']['compression_ratio']:.6f}x")
    print(f"     (Fitted space is {(1-results['volume_analysis']['compression_ratio'])*100:.1f}% smaller)")
    
    print("\n📐 DIMENSIONALITY:")
    print(f"   • Effective dimension (Initial): {results['pca_A']['d_eff']:.2f}")
    print(f"   • Effective dimension (Fitted): {results['pca_B']['d_eff']:.2f}")
    print(f"   • Intrinsic dimension (Initial): {results['id_A']['id_est']:.2f} ± {results['id_A']['id_std']:.2f}")
    print(f"   • Intrinsic dimension (Fitted): {results['id_B']['id_est']:.2f} ± {results['id_B']['id_std']:.2f}")
    
    pA_cumvar = np.cumsum(results['pca_A']['explained_ratio'])
    pB_cumvar = np.cumsum(results['pca_B']['explained_ratio'])
    n_comp_A = np.argmax(pA_cumvar >= 0.9) + 1
    n_comp_B = np.argmax(pB_cumvar >= 0.9) + 1
    print(f"   • Components for 90% variance (Initial): {n_comp_A}/8")
    print(f"   • Components for 90% variance (Fitted): {n_comp_B}/8")
    n_comp_A = np.argmax(pA_cumvar >= 0.95) + 1
    n_comp_B = np.argmax(pB_cumvar >= 0.95) + 1
    print(f"   • Components for 95% variance (Initial): {n_comp_A}/8")
    print(f"   • Components for 95% variance (Fitted): {n_comp_B}/8")
    
    print("\n🔍 STRUCTURE:")
    print(f"   • k-NN overlap (k=3): {results['knn_overlap']['mean_overlap']:.3f}")
    print(f"   • Clusters detected (Initial): {results['clustering']['n_clusters_A']}")
    print(f"   • Clusters detected (Fitted): {results['clustering']['n_clusters_B']}")
    
    print("\n📈 STATISTICAL TESTS:")
    ks = results['pairwise_compare']['ks']
    print(f"   • KS test (distances): D={ks.statistic:.4f}, p={ks.pvalue:.2e}")
    print(f"   • Wasserstein distance: {results['pairwise_compare']['wasserstein']:.4f}")
    print(f"   • Cohen's d (effect size): {results['pairwise_compare']['cohens_d']:.3f}")
    
    # Interpretation
    print("\n💡 INTERPRETATION:")
    if results['volume_analysis']['compression_ratio'] < 0.1:
        print("   ✓ STRONG EVIDENCE: Fitted space is >90% smaller than initial space")
    elif results['volume_analysis']['compression_ratio'] < 0.3:
        print("   ✓ MODERATE EVIDENCE: Fitted space is 70-90% smaller than initial space")
    else:
        print("   ⚠ WEAK EVIDENCE: Fitted space reduction is <70%")
    
    if results['pca_B']['d_eff'] < results['pca_A']['d_eff'] * 0.7:
        print("   ✓ Parameters converge to lower-dimensional manifold")
    
    if results['clustering']['n_clusters_B'] <= 2:
        print("   ✓ Calibration converges to single solution region (1-2 clusters)")
    else:
        print(f"   ⚠ Multiple solution regions detected ({results['clustering']['n_clusters_B']} clusters)")
    
    print("="*60 + "\n")


if __name__ == "__main__":

    fitted_params = []
    
    #{'66--done', '4--done', '65', '67', '20--done', '73--done', '53--done', '91--done', '5--done', '85', '13', '32', '21--done', '93--done', '7', '199', '17--done'}
    #{need - 7,32,65,67,85} -- 32 and 67 have performed quite poorly
    #final_models = [339,4075,5230,13282,17150,20286,21230,53230,66153,91230,93197]
    final_models = [339,4075,5230,13282,17150,20286,21230,53230,66153,73230,91230,93197]
    
    for sim in final_models:
        
        parent_dir = 'InferenceConfig/' if sim!=339 else '../intention/CoordinateDescentconfig/'
        param_config = get_config(parent_dir,'inference_seed','',sim)
        print(sim,param_config)
        fitted_params.append(param_config)
    
    print('------------------')
    
    original_params = []
    
    for sim in final_models:
        original_sim_idx = sim//1000 if sim!=339 else 199
        parent_dir = '../intention/LHC_configs/' if original_sim_idx!=199 else '../intention/OFAT_configs/'
        param_config = get_config(parent_dir,'inference_seed','',original_sim_idx)
        print(original_sim_idx,param_config)
        original_params.append(param_config)
    
    
    all_params = []
    
    for sim in range(0,100):
        param_config = get_config(parent_dir,'../intention/LHC_configs/','',sim)
        #print(original_sim_idx,param_config)
        all_params.append(param_config)
    
    print('collected everything')
    
    all_params = np.array(all_params)  # Large reference set
    A = np.array(original_params)  # Initial
    B = np.array(fitted_params) # Fitted (compressed and shifted)
    
    print(f"Data shapes: all_params={all_params.shape}, A={A.shape}, B={B.shape}")
    
    # Run analysis
    results = run_all(A, B, all_params, scale_together=True, fig_prefix='example')
    
    # Print summary
    print_summary(results)
    
    # Access individual results
    print("\nDetailed results available in 'results' dictionary:")
    print(f"  - Pairwise KS p-value: {results['pairwise_compare']['ks'].pvalue}")
    print(f"  - Mean k-NN overlap: {results['knn_overlap']['mean_overlap']:.3f}")
    print(f"  - Volume compression: {results['volume_analysis']['compression_ratio']:.6f}")
    
    print('results dict')
    print(results)
    
    
    