"""
=============================================================================
  Vegetation/Forest Health Monitoring Using Vegetation Indices & ML Regression
=============================================================================
  Author   : Forest Health Analytics Project
  Date     : 2025
  Python   : 3.8+
  Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn

  Description:
    This project simulates multispectral remote sensing data for forest parcels,
    computes multiple vegetation indices (NDVI, EVI, SAVI, NDRE, GNDVI, NBR),
    trains and compares multiple ML regression models to predict a continuous
    "Forest Health Score" (FHS), evaluates model performance, and produces
    publication-quality visualizations.

  Outputs:
    1. vegetation_health_results.png  – Comprehensive visualization dashboard
    2. model_comparison.png           – Model performance bar charts
    3. feature_importance.png         – Feature importance / coefficient plots
    4. vegetation_health_report.csv   – Full dataset with predictions
=============================================================================
"""
import os
# Create output folder inside current project
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, explained_variance_score)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                  ElasticNet, BayesianRidge)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_multispectral_data(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Simulate multispectral satellite band reflectances for forest parcels.

    Bands (reflectance values 0-1):
      Blue  (B2) : ~450–520 nm
      Green (B3) : ~520–600 nm
      Red   (B4) : ~630–680 nm
      Red-Edge (B5): ~705–745 nm
      NIR   (B8) : ~835–875 nm
      SWIR1 (B11): ~1565–1655 nm

    Forest Health Score (FHS) is a composite 0–100 index derived from bands
    with added noise, mimicking ground-truth field measurements.
    """
    rng = np.random.default_rng(random_state)

    # Health classes: Healthy (0), Stressed (1), Diseased (2), Burned (3)
    n_classes = 4
    labels = ['Healthy', 'Stressed', 'Diseased', 'Burned']
    health_class = rng.integers(0, n_classes, size=n_samples)

    # Base spectral signatures per class
    class_params = {
        # class: (blue_m, green_m, red_m, rededge_m, nir_m, swir1_m)
        0: dict(blue=(0.04, 0.01), green=(0.10, 0.015), red=(0.05, 0.01),
                rededge=(0.15, 0.02), nir=(0.50, 0.05), swir1=(0.12, 0.02)),  # Healthy
        1: dict(blue=(0.06, 0.015), green=(0.09, 0.015), red=(0.10, 0.02),
                rededge=(0.20, 0.03), nir=(0.35, 0.05), swir1=(0.20, 0.03)),  # Stressed
        2: dict(blue=(0.07, 0.02), green=(0.08, 0.02), red=(0.15, 0.025),
                rededge=(0.18, 0.03), nir=(0.25, 0.06), swir1=(0.28, 0.04)),  # Diseased
        3: dict(blue=(0.05, 0.01), green=(0.06, 0.01), red=(0.18, 0.03),
                rededge=(0.10, 0.02), nir=(0.10, 0.03), swir1=(0.45, 0.06)),  # Burned
    }

    bands = {}
    for band, idx in [('blue', 0), ('green', 1), ('red', 2),
                      ('rededge', 3), ('nir', 4), ('swir1', 5)]:
        values = np.zeros(n_samples)
        for cls in range(n_classes):
            mask = health_class == cls
            m, s = class_params[cls][band]
            values[mask] = np.clip(rng.normal(m, s, mask.sum()), 0.01, 0.99)
        bands[band] = values

    df = pd.DataFrame(bands)
    df['health_class'] = health_class
    df['health_label'] = [labels[c] for c in health_class]

    # Environmental covariates
    df['elevation_m']   = rng.uniform(200, 3500, n_samples)
    df['slope_deg']     = rng.uniform(0, 45, n_samples)
    df['precip_mm']     = rng.uniform(300, 2500, n_samples)
    df['temp_celsius']  = rng.uniform(-5, 30, n_samples)
    df['canopy_cover']  = rng.uniform(0, 1, n_samples)
    df['stand_age']     = rng.uniform(5, 200, n_samples)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. VEGETATION INDEX COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_vegetation_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute six standard vegetation / spectral indices."""
    eps = 1e-8  # avoid division by zero

    nir   = df['nir']
    red   = df['red']
    green = df['green']
    blue  = df['blue']
    re    = df['rededge']
    swir1 = df['swir1']

    # 1. NDVI – Normalized Difference Vegetation Index
    df['NDVI'] = (nir - red) / (nir + red + eps)

    # 2. EVI – Enhanced Vegetation Index (Huete et al. 2002)
    df['EVI'] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)

    # 3. SAVI – Soil-Adjusted Vegetation Index (L = 0.5)
    L = 0.5
    df['SAVI'] = ((nir - red) / (nir + red + L + eps)) * (1 + L)

    # 4. NDRE – Normalized Difference Red-Edge (sensitive to chlorophyll)
    df['NDRE'] = (nir - re) / (nir + re + eps)

    # 5. GNDVI – Green NDVI (sensitive to chlorophyll content)
    df['GNDVI'] = (nir - green) / (nir + green + eps)

    # 6. NBR – Normalized Burn Ratio (fires & burn severity)
    df['NBR'] = (nir - swir1) / (nir + swir1 + eps)

    # ── Forest Health Score (FHS) – synthetic ground truth ──────────────────
    # Weighted composite of indices + environmental factors + noise
    fhs = (
        35 * np.clip(df['NDVI'], -1, 1) +
        20 * np.clip(df['EVI'],  -1, 1) +
        15 * np.clip(df['NDRE'], -1, 1) +
        10 * np.clip(df['GNDVI'], -1, 1) +
        10 * np.clip(df['NBR'],   -1, 1) +
        5  * df['canopy_cover'] +
        3  * (df['precip_mm'] / 2500) +
        2  * (1 - df['temp_celsius'] / 30)
    )
    # Rescale to 0–100
    fhs_min, fhs_max = fhs.min(), fhs.max()
    df['FHS'] = 100 * (fhs - fhs_min) / (fhs_max - fhs_min + eps)
    # Add realistic noise
    rng = np.random.default_rng(99)
    df['FHS'] = np.clip(df['FHS'] + rng.normal(0, 3, len(df)), 0, 100)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    """Return a dictionary of regression models to compare."""
    return {
        'Linear Regression':    LinearRegression(),
        'Ridge Regression':     Ridge(alpha=1.0),
        'Lasso Regression':     Lasso(alpha=0.05, max_iter=5000),
        'ElasticNet':           ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000),
        'Bayesian Ridge':       BayesianRidge(),
        'Decision Tree':        DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest':        RandomForestRegressor(n_estimators=150, max_depth=12,
                                                       n_jobs=-1, random_state=42),
        'Extra Trees':          ExtraTreesRegressor(n_estimators=150, n_jobs=-1, random_state=42),
        'Gradient Boosting':    GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                           max_depth=5, random_state=42),
        'AdaBoost':             AdaBoostRegressor(n_estimators=100, random_state=42),
        'SVR (RBF)':            SVR(kernel='rbf', C=10, epsilon=0.5),
        'KNN Regressor':        KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
        'MLP Neural Net':       MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                                              activation='relu', max_iter=1000,
                                              random_state=42),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test,
                       feature_names: list) -> pd.DataFrame:
    """Train all models, compute metrics, return results DataFrame."""
    models  = get_models()
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    results = []
    trained_models = {}

    for name, model in models.items():
        # Models that benefit from scaling
        needs_scale = name in ('SVR (RBF)', 'KNN Regressor', 'MLP Neural Net',
                               'Lasso Regression', 'ElasticNet', 'Ridge Regression',
                               'Bayesian Ridge', 'Linear Regression')
        Xtr = X_tr_sc if needs_scale else X_train
        Xte = X_te_sc if needs_scale else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        evs  = explained_variance_score(y_test, y_pred)

        # 5-fold CV on training data
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, Xtr, y_train, cv=cv,
                                    scoring='r2', n_jobs=-1)

        results.append({
            'Model':   name,
            'RMSE':    round(rmse, 3),
            'MAE':     round(mae, 3),
            'R²':      round(r2, 4),
            'Exp.Var': round(evs, 4),
            'CV R² Mean': round(cv_scores.mean(), 4),
            'CV R² Std':  round(cv_scores.std(), 4),
        })
        trained_models[name] = (model, scaler if needs_scale else None,
                                 X_te_sc if needs_scale else X_test, y_pred)

        print(f"  {name:<25} RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

    return pd.DataFrame(results), trained_models


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    'Healthy':  '#2ecc71',
    'Stressed': '#f39c12',
    'Diseased': '#e74c3c',
    'Burned':   '#7f8c8d',
}


def plot_dashboard(df: pd.DataFrame, results_df: pd.DataFrame,
                   trained_models: dict, feature_names: list,
                   X_test, y_test):

    """Generate a comprehensive 3×3 dashboard figure."""

    fig = plt.figure(figsize=(22, 18), facecolor='#0d1117')
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.38)

    text_c = '#e6edf3'
    spine_c = '#30363d'

    def style_ax(ax, title):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors=text_c, labelsize=9)
        ax.set_title(
            title,
            color=text_c,
            fontsize=11,
            fontweight='bold',
            pad=8
        )

        for sp in ax.spines.values():
            sp.set_color(spine_c)

        ax.xaxis.label.set_color(text_c)
        ax.yaxis.label.set_color(text_c)

    # ==========================
    # (0,0) NDVI Distribution
    # ==========================
    ax1 = fig.add_subplot(gs[0, 0])

    for lbl, grp in df.groupby('health_label'):
        ax1.hist(
            grp['NDVI'],
            bins=30,
            alpha=0.7,
            label=lbl,
            color=PALETTE[lbl],
            edgecolor='none',
            density=True
        )

    style_ax(ax1, 'NDVI Distribution by Health Class')

    ax1.set_xlabel('NDVI')
    ax1.set_ylabel('Density')

    ax1.legend(
        fontsize=8,
        facecolor='#21262d',
        edgecolor=spine_c,
        labelcolor=text_c
    )

    # ==========================
    # (0,1) Correlation Heatmap
    # ==========================
    ax2 = fig.add_subplot(gs[0, 1])

    vi_cols = [
        'NDVI', 'EVI', 'SAVI',
        'NDRE', 'GNDVI',
        'NBR', 'FHS'
    ]

    corr = df[vi_cols].corr()

    im = ax2.imshow(
        corr.values,
        cmap='RdYlGn',
        vmin=-1,
        vmax=1,
        aspect='auto'
    )

    ax2.set_xticks(range(len(vi_cols)))
    ax2.set_xticklabels(
        vi_cols,
        rotation=45,
        ha='right'
    )

    ax2.set_yticks(range(len(vi_cols)))
    ax2.set_yticklabels(vi_cols)

    for i in range(len(vi_cols)):
        for j in range(len(vi_cols)):
            ax2.text(
                j,
                i,
                f'{corr.values[i, j]:.2f}',
                ha='center',
                va='center',
                fontsize=7
            )

    plt.colorbar(im, ax=ax2)

    style_ax(ax2, 'Vegetation Indices Correlation')

    # ==========================
    # (0,2) NDVI vs NBR
    # ==========================
    ax3 = fig.add_subplot(gs[0, 2])

    for lbl, grp in df.groupby('health_label'):
        ax3.scatter(
            grp['NDVI'],
            grp['NBR'],
            c=PALETTE[lbl],
            s=8,
            alpha=0.5,
            label=lbl
        )

    style_ax(ax3, 'NDVI vs NBR')

    ax3.set_xlabel('NDVI')
    ax3.set_ylabel('NBR')

    ax3.legend()

    # ==========================
    # (1,0) Model R²
    # ==========================
    ax4 = fig.add_subplot(gs[1, 0])

    res_sorted = results_df.sort_values(
        'R²',
        ascending=True
    )

    bars = ax4.barh(
        res_sorted['Model'],
        res_sorted['R²']
    )

    for bar, val in zip(
            bars,
            res_sorted['R²']):

        ax4.text(
            val + 0.002,
            bar.get_y() + bar.get_height()/2,
            f'{val:.3f}',
            va='center'
        )

    style_ax(ax4, 'Model R² Comparison')

    # ==========================
    # (1,1) RMSE vs MAE
    # ==========================
    ax5 = fig.add_subplot(gs[1, 1])

    x = np.arange(len(results_df))
    w = 0.4

    ax5.bar(
        x - w/2,
        results_df['RMSE'],
        w,
        label='RMSE'
    )

    ax5.bar(
        x + w/2,
        results_df['MAE'],
        w,
        label='MAE'
    )

    ax5.set_xticks(x)

    ax5.set_xticklabels(
        results_df['Model'],
        rotation=45,
        ha='right'
    )

    style_ax(ax5, 'RMSE & MAE')

    ax5.legend()

    # ==========================
    # (1,2) Actual vs Predicted
    # ==========================
    ax6 = fig.add_subplot(gs[1, 2])

    best_name = results_df.loc[
        results_df['R²'].idxmax(),
        'Model'
    ]

    _, _, _, y_pred_best = trained_models[
        best_name
    ]

    ax6.scatter(
        y_test,
        y_pred_best,
        s=12,
        alpha=0.5
    )

    style_ax(
        ax6,
        f'Actual vs Predicted\n{best_name}'
    )

    ax6.set_xlabel('Actual')
    ax6.set_ylabel('Predicted')

    # ==========================
    # (2,0) FHS Distribution
    # ==========================
    ax7 = fig.add_subplot(gs[2, 0])

    for lbl, grp in df.groupby(
            'health_label'):

        ax7.hist(
            grp['FHS'],
            bins=25,
            alpha=0.7,
            label=lbl,
            density=True
        )

    style_ax(ax7, 'FHS Distribution')

    ax7.legend()

    # ==========================
    # (2,1) Feature Importance
    # ==========================
    ax8 = fig.add_subplot(gs[2, 1])

    rf_model, _, _, _ = trained_models[
        'Random Forest'
    ]

    feat_imp = pd.Series(
        rf_model.feature_importances_,
        index=feature_names
    ).sort_values()

    ax8.barh(
        feat_imp.index,
        feat_imp.values
    )

    style_ax(
        ax8,
        'Feature Importance'
    )

    # ==========================
    # (2,2) CV Scores
    # ==========================
    ax9 = fig.add_subplot(gs[2, 2])

    res_cv = results_df.sort_values(
        'CV R² Mean'
    )

    ax9.barh(
        res_cv['Model'],
        res_cv['CV R² Mean']
    )

    style_ax(
        ax9,
        'Cross Validation'
    )

    # ==========================
    # Save (FIXED)
    # ==========================
    fig.suptitle(
        '🌿 Vegetation Health Dashboard',
        fontsize=18,
        fontweight='bold',
        color=text_c
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "vegetation_health_dashboard.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches='tight',
        facecolor=fig.get_facecolor()
    )

    plt.close()

    print(
        f"✓ Dashboard saved → {output_file}"
    )



def plot_health_map(df: pd.DataFrame):
    """Simulate a 2-D spatial health map using NDVI & NBR."""

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 5),
        facecolor='#0d1117'
    )

    indices = ['NDVI', 'NBR', 'FHS']
    cmaps = ['RdYlGn', 'RdYlGn', 'RdYlGn']
    labels = [
        'NDVI',
        'Normalized Burn Ratio',
        'Forest Health Score'
    ]

    text_c = '#e6edf3'

    # Create pseudo-spatial grid
    for ax, idx, cmap, lbl in zip(
            axes,
            indices,
            cmaps,
            labels):

        grid = df[idx].values[:1200].reshape(
            50,
            24
        )

        im = ax.imshow(
            grid,
            cmap=cmap,
            aspect='auto',
            vmin=df[idx].quantile(0.02),
            vmax=df[idx].quantile(0.98)
        )

        plt.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

        ax.set_title(
            f'Spatial Map: {lbl}',
            color=text_c,
            fontsize=11,
            fontweight='bold'
        )

        ax.set_facecolor('#161b22')

        ax.tick_params(
            colors=text_c
        )

        for sp in ax.spines.values():
            sp.set_color('#30363d')

    fig.suptitle(
        '🛰️ Simulated Spatial Distribution of Vegetation Indices',
        fontsize=14,
        fontweight='bold',
        color=text_c
    )

    fig.patch.set_facecolor('#0d1117')

    plt.tight_layout()

    # WINDOWS SAFE SAVE
    output_file = os.path.join(
        OUTPUT_DIR,
        "spatial_health_map.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches='tight',
        facecolor='#0d1117'
    )

    plt.close()

    print(
        f"✓ Spatial map saved → {output_file}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():

    print("=" * 65)
    print(" Vegetation / Forest Health Monitoring – ML Regression Pipeline")
    print("=" * 65)

    # Step 1
    print("\n[1] Generating synthetic multispectral data ...")

    df = generate_multispectral_data(
        n_samples=1200,
        random_state=42
    )

    print(f"    Dataset shape : {df.shape}")

    # Step 2
    print("\n[2] Computing vegetation indices ...")

    df = compute_vegetation_indices(df)

    vi_stats = df[
        ['NDVI', 'EVI', 'SAVI',
         'NDRE', 'GNDVI',
         'NBR', 'FHS']
    ].describe().round(3)

    print(vi_stats.to_string())

    # Step 3
    print("\n[3] Preparing feature matrix ...")

    feature_cols = [
        'NDVI', 'EVI', 'SAVI',
        'NDRE', 'GNDVI', 'NBR',
        'blue', 'green', 'red',
        'rededge', 'nir', 'swir1',
        'elevation_m', 'slope_deg',
        'precip_mm', 'temp_celsius',
        'canopy_cover', 'stand_age'
    ]

    X = df[feature_cols].values
    y = df['FHS'].values

    print(f"    Features : {len(feature_cols)}")
    print(f"    Samples  : {len(y)}")

    # Step 4
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(
        f"\n[4] Train/Test split → "
        f"{len(y_train)} train / {len(y_test)} test"
    )

    # Step 5
    print("\n[5] Training models ...")

    results_df, trained_models = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols
    )

    # Step 6
    print("\n[6] Results Summary")

    summary = results_df.sort_values(
        'R²',
        ascending=False
    ).reset_index(drop=True)

    summary.index += 1

    print(summary.to_string())

    best = summary.iloc[0]

    print(
        f"\n🏆 Best model: "
        f"{best['Model']} | "
        f"R²={best['R²']}"
    )

    # Step 7
    print("\n[7] Generating visualizations ...")

    plot_dashboard(
        df,
        results_df,
        trained_models,
        feature_cols,
        X_test,
        y_test
    )

    plot_health_map(df)

    # Step 8 (FIXED)
    print("\n[8] Saving reports ...")

    df_out = df.copy()

    report_file = os.path.join(
        OUTPUT_DIR,
        "vegetation_health_report.csv"
    )

    results_file = os.path.join(
        OUTPUT_DIR,
        "model_results.csv"
    )

    df_out.to_csv(
        report_file,
        index=False
    )

    results_df.sort_values(
        'R²',
        ascending=False
    ).to_csv(
        results_file,
        index=False
    )

    print(
        f"✓ Report saved → {report_file}"
    )

    print(
        f"✓ Results saved → {results_file}"
    )

    print("\n" + "=" * 65)
    print(" Pipeline complete. All outputs saved.")
    print("=" * 65)

    return df, results_df, trained_models


if __name__ == '__main__':
    df, results_df, trained_models = main()