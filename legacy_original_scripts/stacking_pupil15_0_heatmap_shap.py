#!/usr/bin/env python
# coding: utf-8

# === 绗竴閮ㄥ垎锛氬簱瀵煎叆 ===
from sklearn import model_selection
import os
import numpy as np
import pandas as pd
import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib
import time
import shap
shap_available = True

matplotlib.use('Agg')
import logging
import nolds
import antropy as ant
import kymatio

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier,
    RandomForestClassifier, StackingClassifier
)
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression as LogiR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV, train_test_split, StratifiedKFold,
    KFold, cross_validate
)
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report,
    recall_score, precision_score
)
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.base import clone
from chardet import detect
from sklearn.impute import SimpleImputer
import plotly.graph_objs as go
import plotly.express as px
import warnings

warnings.filterwarnings(action='ignore')
from concurrent.futures import CancelledError
from dtaidistance import dtw
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process.kernels import RBF
from joblib import Memory
from kymatio.numpy import Scattering1D
from scipy.signal import find_peaks

from optuna.integration import OptunaSearchCV
from sklearn.model_selection import StratifiedKFold
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from sklearn.utils.validation import check_array
from tenacity import retry, stop_after_attempt, wait_fixed
from sklearn.utils import resample

from dask.distributed import Client

from itertools import combinations
from collections import Counter
from sklearn.model_selection import cross_val_score, KFold
from joblib import parallel_backend, Parallel, delayed
from sklearn import model_selection
import antropy as ant
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import NuSVC
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
RANDOM_STATE = int(os.getenv('REPRO_SEED', os.getenv('PYTHONHASHSEED', '1412')))
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# === 鉁?绾?NumPy 瀹炵幇 approximate entropy ===
def approximate_entropy(U, m=2, r=None):
    U = np.array(U)
    N = len(U)
    if r is None:
        r = 0.2 * np.std(U)

    def _phi(m):
        x = np.array([U[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) / (N - m + 1)
        return np.sum(np.log(C + 1e-10)) / (N - m + 1)

    return abs(_phi(m) - _phi(m + 1))


# === 鉁?绾?NumPy 瀹炵幇 sample entropy ===
def sample_entropy(U, m=2, r=None):
    U = np.array(U)
    N = len(U)
    if r is None:
        r = 0.2 * np.std(U)

    def _match_count(m):
        x = np.array([U[i:i + m] for i in range(N - m)])
        count = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0)
        return np.sum(count) - len(x)

    B = _match_count(m)
    A = _match_count(m + 1)
    return -np.log(A / (B + 1e-10) + 1e-10)


# === 鏇夸唬DTW鍑芥暟锛堢畝鍖栫増娆ф皬璺濈锛?===
def fake_dtw(s1, s2):
    min_len = min(len(s1), len(s2))
    return np.linalg.norm(s1[:min_len] - s2[:min_len])

# === 璁剧疆鏃ュ織 ===
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']  # 浣跨敤宸叉娴嬪瓨鍦ㄧ殑瀛椾綋
plt.rcParams['axes.unicode_minus'] = False

# 绂佺敤matplotlib瀛椾綋璋冭瘯鏃ュ織
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

# 鍘熸湁璀﹀憡杩囨护
warnings.filterwarnings(action='ignore')

# === 绗簩閮ㄥ垎锛氬嚱鏁板畾涔?===
def fusion_estimators(model, X_train, y_train, X_test, y_test):
    """Stacking妯″瀷璇勪及锛堝幓闄ゅ叏灞€鍙橀噺锛屽鍔犲紓甯告姏鍑烘柟渚胯皟璇曪級"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc'},
        n_jobs=1,
        return_estimator=True,
        return_train_score=True,
        verbose=10,
        error_score='raise'  # 鐩存帴鎶涘嚭寮傚父锛屾柟渚胯皟璇?    )

    print("\n=== 浜ゅ弶楠岃瘉缁撴灉 ===")
    print(f"璁粌闆嗗噯纭巼: {results['train_accuracy'].mean():.4f}")
    print(f"楠岃瘉闆嗗噯纭巼: {results['test_accuracy'].mean():.4f}")
    print(f"璁粌闆咥UC: {results['train_roc_auc'].mean():.4f}")
    print(f"楠岃瘉闆咥UC: {results['test_roc_auc'].mean():.4f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return results, model, y_pred


memory = Memory(location='./model_eval_cache', verbose=0)


def individual_estimators(estimators, X_train, y_train, X_test, y_test):
    """鍩哄涔犲櫒璇勪及锛堝弬鏁颁紶閫掞級"""
    results = []
    for name, est in estimators:
        model_type = MODEL_TYPE_MAP.get(name, 'default')
        preprocessor = create_model_specific_preprocessing(model_type)

        # 鎵撳嵃棰勫鐞嗛厤缃?        print(f"\n[棰勫鐞嗛獙璇乚 妯″瀷 {name} 浣跨敤棰勫鐞嗘楠?")  # 鏍囬
        print("棰勫鐞嗙閬撶被鍨?", type(preprocessor).__name__)
        for step_name, step in preprocessor.named_steps.items():
            print(f"  鈹溾攢 {step_name}: {step.__class__.__name__}")
            if hasattr(step, 'get_params'):
                print(f"  鈹?  鈹斺攢 鍙傛暟: { {k: v for k, v in step.get_params().items() if v is not None} }")

        # 鏋勫缓瀹屾暣绠￠亾
        model_pipe = Pipeline([
            ('preprocess', preprocessor),
            ('estimator', clone(est))
        ])

        with parallel_backend('dask'):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            res = cross_validate(
                est, X_train, y_train,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                return_train_score=True,
                verbose=10
            )

            model = est.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"\n{name}:")
        print(f"  Train: {res['train_score'].mean():.4f}")
        print(f"  CV: {res['test_score'].mean():.4f}")
        print(f"  Test: {test_score:.4f}")
        results.append((name, model))
    return results


def plot_feature_heatmap(X, y, filename='feature_heatmap.png'):
    """鐗瑰緛鐩稿叧鎬х儹鍔涘浘鐢熸垚"""
    # === 淇敼1锛氭牴鎹紶鍏ョ殑鏂囦欢鍚嶇‘瀹氭柊鏂囦欢鍚?===
    base_filename = filename.replace('train_heatmap', 'train Feature Correlation Heatmap').replace('test_heatmap', 'test Feature Correlation Heatmap')
    # 绉婚櫎鍙兘鐨勬枃浠舵墿灞曞悕
    base_filename = base_filename.replace('.png', '')

    # === 淇敼2锛氬彧浣跨敤鐗瑰緛鏁版嵁锛堟帓闄abel鍒楋級===
    df_feature = pd.DataFrame(X)
    corr = df_feature.corr()
    feature_names = df_feature.columns.tolist()
    
    # 鍒涘缓閬僵鐭╅樀
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # === 淇敼3锛氳缃洿澶х殑鍥惧儚灏哄鍜屽瓧浣?===
    plt.figure(figsize=(24, 20), dpi=300)
    
    # === 淇敼4锛氱粯鍒剁儹鍔涘浘锛堜娇鐢ㄦ柊鏍囩锛?==
    ax = sns.heatmap(
        corr, 
        mask=mask, 
        cmap='coolwarm', 
        center=0,  
        square=True, 
        linewidths=0.5, 
        cbar_kws={"shrink": 0.5},
        annot=False  # 鍏抽棴娉ㄩ噴閬垮厤鏉備贡
    )
    
    # === 淇敼5锛氳缃潗鏍囪酱鏍囩锛?-41鏇夸唬0-40锛?==
    n_features = len(feature_names)
    tick_labels = [str(i+1) for i in range(n_features)]
    
    plt.xticks(
        ticks=np.arange(0.5, n_features+0.5), 
        labels=tick_labels,
        rotation=0,
        fontsize=10
    )
    
    plt.yticks(
        ticks=np.arange(0.5, n_features+0.5),
        labels=tick_labels,
        rotation=0,
        fontsize=10
    )
    
    plt_title = f"{'Train' if 'train' in filename else 'Test'} Feature Correlation Heatmap"
    plt.title(plt_title, fontsize=54, pad=20) 
    
    # 璋冩暣甯冨眬涓哄浘娉ㄧ暀绌洪棿
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    png_filename = f"{base_filename}.png"
    svg_filename = f"{base_filename}.svg"
    # 淇濆瓨PNG
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    # 淇濆瓨SVG
    plt.savefig(svg_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"宸蹭繚瀛樼儹鍔涘浘:  {png_filename} 鍜?{svg_filename}")


def safe_get_final_est(model):
    if hasattr(model, 'final_estimator_'):
        return model.final_estimator_
    elif hasattr(model, 'final_estimator'):
        return model.final_estimator
    return None

def create_model_specific_preprocessing(model_type):
    """妯″瀷涓撶敤棰勫鐞嗙閬?""
    from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    pipelines = {
        'TreeBased': Pipeline([('scaler', None)]),  # 鏍戞ā鍨嬩笉鏍囧噯鍖?        'KernelMethod': Pipeline([('scaler', RobustScaler())]),
        'NeuralNet': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95))
        ]),
        'default': Pipeline([('scaler', MinMaxScaler())])
    }
    return pipelines.get(model_type, pipelines['default'])


# === 涓荤▼搴?===
if __name__ == "__main__":
    # === 鍒濆鍖朌ask闆嗙兢 ===
    client = Client(
        n_workers=5,
        threads_per_worker=4,
        memory_limit='20GiB',
        dashboard_address=8787,  # 鐩戞帶鍦板潃http://localhost:8787
        processes=True
    )
    print("\n=== Dask闆嗙兢淇℃伅 ===\n", client)

    try:
        # === 鏁版嵁鍔犺浇 ===
        print("\n=== 鏁版嵁鍔犺浇 ===")


        def robust_csv_reader(file_path):
            with open(file_path, 'rb') as f:
                rawdata = f.read(100000)
                enc = detect(rawdata)['encoding']

            encodings = ['utf-8-sig', 'gb18030', enc, 'latin1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, engine='python')
                    # 鏁版嵁楠岃瘉
                    assert not df.empty, "鏁版嵁涓虹┖"
                    assert 'label' in df.columns, "缂哄皯label鍒?
                    return df
                except Exception as e:
                    print(f"灏濊瘯 {encoding} 澶辫触: {str(e)}")
                    continue
            raise ValueError("鏃犳硶瑙ｆ瀽鏂囦欢缂栫爜")


        df = robust_csv_reader(os.getenv("PLR_DATA_CSV", "data/preprocessed_data.csv"))
        df = df.set_index(df.columns[0])  # 淇绱㈠紩闂

        time_series_columns = [f'D{i}' for i in range(1, 126)]

        MODEL_TYPE_MAP = {
            # TreeBased
            'CatBoost': 'TreeBased',
            'LGBM': 'TreeBased',
            'XGBoost': 'TreeBased',
            'RandomForest': 'TreeBased',
            'HGBoost': 'TreeBased',
            'ExtraTrees': 'TreeBased',
            'AdaBoost': 'TreeBased',
            'GradientBoosting': 'TreeBased',
            'BaggingDT': 'TreeBased',

            # KernelMethod
            'SVC': 'KernelMethod',
            'KernelSVM': 'KernelMethod',

            # NeuralNet
            'MLP': 'NeuralNet',

            # LinearModel
            'QDA': 'LinearModel',
            'SGD': 'LinearModel',

            # Probabilistic
            'GaussianProcess': 'Probabilistic',
            'Bayesian': 'Probabilistic',

            # Distance
            'KNN': 'Distance'
        }


        def multiscale_entropy(series, max_scale):
            try:
                # 杞崲涓簄umpy鏁扮粍
                arr = series.values if isinstance(series, pd.Series) else series

                # 澧炲姞闀垮害鏍￠獙
                if len(arr) < 50 * max_scale:
                    return 0.0

                entropy_values = []
                for scale in range(1, max_scale + 1):
                    # 鏀硅繘鐨勭矖绮掑害鏂规硶
                    seg_length = len(arr) // scale
                    truncated = arr[:seg_length * scale]
                    coarse = np.mean(truncated.reshape(scale, seg_length), axis=0)

                    # 鏍锋湰鐔佃绠?                    if len(coarse) > 10:
                        sampen = nolds.sampen(coarse)
                        entropy_values.append(sampen)

                return np.mean(entropy_values) if entropy_values else 0.0
            except Exception as e:
                print(f"澶氬昂搴︾喌璁＄畻寮傚父: {str(e)}")
                return 0.0


        # 杩戜技鐔佃绠楀嚱鏁?        def approximate_entropy(U, m=2, r=None):
            U = np.array(U)
            N = len(U)
            if r is None:
                r = 0.2 * np.std(U)

            def _phi(m):
                x = np.array([U[i:i + m] for i in range(N - m + 1)])
                C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) / (N - m + 1)
                return np.sum(np.log(C + 1e-10)) / (N - m + 1)

            return abs(_phi(m) - _phi(m + 1))


        # 鐗瑰緛鎻愬彇鍑芥暟
        def extract_pupil_features(series):
            #logging.debug(f"澶勭悊搴忓垪: {series.name}, 绫诲瀷: {type(series.name)}")
            values = series.values if hasattr(series, 'values') else series
            diffs = np.diff(values) if len(values) > 1 else np.array([])
            stim_idx = 24  # 绗?5甯?            pre_len = 24   # 鍒烘縺鐐瑰墠闀垮害鍥哄畾涓?4甯?            post_len = 100  
            pre = series[:stim_idx]
            post = series[stim_idx:]
            valid_stim_idx = min(stim_idx, len(series) - 1)
            pre_stimulus = series[:valid_stim_idx]
            post_stimulus = series[valid_stim_idx:valid_stim_idx+post_len]

            features = {}
            # 娣诲姞鍏ㄥ眬NaN妫€鏌?            if series.isnull().all():
                series_interp = series.interpolate(method='linear', limit_direction='both')
                if series_interp.isnull().all():
                    series_interp = series_interp.fillna(0)
                series = series_interp
            
            # 鍩虹缁熻鐗瑰緛
            features['mean'] = np.nanmean(series) if not np.all(np.isnan(series)) else 0.0
            features['std'] = np.nanstd(series) if not np.all(np.isnan(series)) else 0.0
            features['max'] = np.nanmax(series) if not np.all(np.isnan(series)) else 0.0
            features['min'] = np.nanmin(series) if not np.all(np.isnan(series)) else 0.0
            features['range'] = features['max'] - features['min']
            
            # 鍙樺寲瓒嬪娍鐗瑰緛
            diffs = np.diff(series)
            features['max_diff'] = np.nanmax(np.abs(diffs)) if len(diffs) > 0 else 0.0
            features['slope'] = (series.iloc[-1] - series.iloc[0])/len(series) if len(diffs) > 1 else 0.0

            # 鍙樺寲閫熺巼鐗瑰緛 
            if len(series) >= 3:
                first_deriv = np.diff(series, n=1)
                features['deriv_mean'] = np.nanmean(first_deriv)
                features['deriv_max'] = np.nanmax(np.abs(first_deriv))
                second_deriv = np.diff(series, n=2)
                features['second_deriv_mean'] = np.nanmean(second_deriv)
            else:
                features.update({
                    'deriv_mean': 0,
                    'deriv_max': 0,
                    'second_deriv_mean': 0
                })

            # 鍔ㄦ€佹椂闂磋鏁?DTW)鐗瑰緛
            template = np.linspace(0, 1, len(series))
            features['dtw_distance'] = dtw.distance_fast(series.values, template)

            # 1. 鍩烘湰缁熻鐗瑰緛
            features['auc'] = np.nansum(values)
            features['mean_diff'] = np.nanmean(diffs) if len(diffs) > 0 else 0.0
            features['max_diff'] = np.nanmax(np.abs(diffs)) if len(diffs) > 0 else 0.0
            features['slope'] = (values[-1] - values[0]) / max(1, len(values))
            features['abs_change'] = np.nanmean(np.abs(diffs)) if len(diffs) > 0 else 0.0

            # 2. FFT鐗瑰緛
            if 'freq_peak1' not in features:  # 濡傛灉鍘熸湁浠ｇ爜娌℃湁璁＄畻FFT
                valid = values[~np.isnan(values)]
                fft = np.abs(np.fft.rfft(valid)) if len(valid) >= 3 else [0, 0, 0]
                features['freq_peak1'] = fft[1] if len(fft) > 1 else 0
                features['freq_peak2'] = fft[2] if len(fft) > 2 else 0
            
            # 3. 鍒嗘鐗瑰緛
            seg_len = 30
            padded = np.pad(values, (0, max(0, seg_len * 3 - len(values))), constant_values=np.nan)
            seg1, seg3 = padded[:seg_len], padded[2*seg_len:3*seg_len]
            s1_mean = np.nanmean(seg1)
            s3_mean = np.nanmean(seg3)
            features['seg3_seg1_log_ratio'] = np.log1p(s3_mean + 1e-3) - np.log1p(s1_mean + 1e-3)

            # 娉㈠姩鐗瑰緛
            features['abs_change'] = np.nanmean(np.abs(diffs)) if len(diffs) > 0 else 0.0
            features['var'] = np.nanvar(series) if not np.all(np.isnan(series)) else 0.0
            
            # 棰戝煙鐗瑰緛
            valid_series = series[~np.isnan(series)]
            if len(valid_series) >= 3:
                fft = np.abs(np.fft.rfft(valid_series))
                features['freq_peak1'] = fft[1] if len(fft) > 1 else 0.0
            else:
                features['freq_peak1'] = 0.0
            
            # 绐楀彛鐗瑰緛
            seg_len = 30  # 鍥哄畾绐楀彛
            if len(series) < seg_len * 3:
                # 涓嶈冻涓夋鏃讹紝琛?鎴栨埅鏂?                padded = np.pad(series, (0, seg_len * 3 - len(series)), mode='constant', constant_values=np.nan)
            else:
                padded = series[:seg_len * 3]
            seg1 = padded[:seg_len]
            seg3 = padded[2*seg_len:3*seg_len]

            seg1_mean = np.nanmean(seg1)
            seg3_mean = np.nanmean(seg3)

            features['seg1_mean'] = seg1_mean
            # 閬垮厤鏋佺鍊煎苟澧炲己绋冲畾鎬?            features['seg3_seg1_ratio'] = seg3_mean / (seg1_mean + 1e-3)  # 鏀逛负1e-3閬垮厤鐖嗙偢
            
            Amax = np.nanmax(pre.values) if not pre.empty else 0.0
            Amin = np.nanmin(post.values) if not post.empty else 0.0
            AAC = Amax - Amin
            features['Amax'] = Amax
            features['Amin'] = Amin
            features['AAC'] = AAC
            features['RAC'] = AAC / (Amin + 1e-3)

            # 鏀剁缉鏃堕棿锛圱Amin锛?            try:
                TAmin = int(np.argmin(post.values)) if not post.empty else 0
            except:
                TAmin = 0
            features['TAmin'] = TAmin

            # 璧峰璐熸枩鐜囨椂闂寸偣锛圱0锛?            T0 = 0
            try:
                post_vals = post.values
                for i in range(len(post_vals) - 4):
                    if np.all(np.diff(post_vals[i:i+5]) < 0):
                        T0 = i + 1
                        break
                else:
                    first_neg = np.where(np.diff(post_vals) < 0)[0]
                    T0 = int(first_neg[0] + 1) if len(first_neg) > 0 else 0
            except:
                T0 = 0
            # features['T0'] = T0

            # SCV
            features['SCV'] = AAC / (TAmin - T0 + 1e-3) if TAmin > T0 else 0.0

            # 5. 鐬冲瓟鍔ㄦ€佸弬鏁帮紙浣跨敤涓婇潰璁＄畻鐨勫彉閲忥級
            diffs = np.diff(values) if len(values) > 1 else np.array([])
            # features['max_constriction_velocity'] = np.nanmin(diffs) if len(diffs) > 0 else 0.0
            features['max_dilation_velocity'] = np.nanmax(diffs) if len(diffs) > 0 else 0.0
            features['normalized_constriction_rate'] = AAC / (TAmin + 1e-3) if TAmin > 0 else 0.0

            # 6. 鎭㈠鐗瑰緛
            recovery_level = Amin + 0.75 * AAC
            try:
                t75 = np.where(post.values >= recovery_level)[0]
                t75_val = int(t75[0]) if len(t75) > 0 else 0
            except:
                t75_val = 0
            features['t75_recovery'] = t75_val
            # features['normalized_dilation_rate'] = (t75_val - TAmin) / (len(post) + 1e-3)

            # 7. AUC鐗瑰緛
            features['pre_auc'] = np.nansum(pre.values) if not pre.empty else 0.0
            features['post_auc'] = np.nansum(post.values) if not post.empty else 0.0
            features['auc_ratio'] = features['post_auc'] / (features['pre_auc'] + 1e-3)

            # 8. 褰㈡€佺壒寰?            values = series.values if hasattr(series, 'values') else series
            peaks, _ = find_peaks(values)
            troughs, _ = find_peaks(-values)
            features['n_peaks'] = len(peaks)
            features['n_troughs'] = len(troughs)
            if len(diffs) > 0:
                zero_crossings = np.where(np.diff(np.sign(diffs)) != 0)[0]
                features['zero_crossing_rate'] = len(zero_crossings) / len(diffs)
            else:
                features['zero_crossing_rate'] = 0.0

            # 9. 鍩虹嚎鐗瑰緛
            tail_segment = series[int(len(series)*0.9):]
            tail_mean = np.nanmean(tail_segment.values) if not tail_segment.empty else 0.0
            baseline_mean = np.nanmean(pre.values) if not pre.empty else 0.0
            features['final_baseline_diff'] = tail_mean - baseline_mean

            # 10. 鐔电壒寰?            try:
                features['approx_entropy'] = approximate_entropy(values)
            except:
                features['approx_entropy'] = 0.0
            # 灏忔尝鏁ｅ皠鐗瑰緛
            valid_series = series[~np.isnan(series)].values
            if len(valid_series) >= 32:  # 纭繚瓒冲闀垮害
                try:
                    # 鍙傛暟璁剧疆锛堜笌鎬濊矾涓€鑷达級
                    J = 4  # 闄嶄綆J鍊硷紝鎵€闇€鏈€灏忛暱搴﹀彉涓?2
                    Q = 4
                    T = len(valid_series)
                    required_length = 2 ** J
                    if T < required_length:
                        valid_series = np.pad(valid_series, (0, required_length - T), mode='edge')
                        T = required_length

                    scattering = Scattering1D(
                            J=J,
                            Q=Q,
                            shape=T,  # 纭T鏄俊鍙烽暱搴︼紙濡俵en(valid_series)锛?                            max_order=1,
                            out_type='array'
                    )
                    input_signal = valid_series.reshape(1, -1).astype(np.float32)

                    # 璁＄畻鏁ｅ皠绯绘暟
                    S = scattering(input_signal)

                    # 闄嶇淮澶勭悊锛堝尮閰嶇壒寰佺淮搴︼級
                    if S.ndim == 3:
                        S = S.squeeze(0)
                    
                    # 鎻愬彇缁熻鐗瑰緛
                    S_log = np.log1p(np.abs(S))
                    num_layers = S_log.shape[0]
                    # 绗竴灞傜壒寰?                    if num_layers >= 1:
                        features['wst1_mean'] = np.mean(S_log[0])
                        features['wst1_std'] = np.std(S_log[0])
                    else:  # 褰撴病鏈夌涓€灞傛椂
                        features['wst1_mean'] = features['mean']
                        features['wst1_std'] = features['std']

                    # 绗簩灞傜壒寰?                    if num_layers >= 2:
                        features['wst2_max'] = np.max(S_log[1])
                    else:  # 褰撴病鏈夌浜屽眰鏃?                        features['wst2_max'] = features['max']

                    #绗簩灞傜喌
                    try:
                        if len(S_log[1]) >= 2:
                            entropy_val = ant.sample_entropy(S_log[1])
                        else:
                            entropy_val = 0.0
                    except Exception as e:
                        logging.warning(f"灏忔尝鏁ｅ皠鐔佃绠楀け璐? {str(e)}")
                        entropy_val = 0.0
                except Exception as e:
                    logging.warning(f"灏忔尝鏁ｅ皠璁＄畻澶辫触: {str(e)}")
                    features.update({  # 寮傚父鏃跺～鍏呭潎鍊?                        'wst1_mean': features['mean'],
                        'wst1_std': features['std'],
                        'wst2_max': features['max']
                    })
            else:
                J, Q, T = 5, 8, 32
                scattering = Scattering1D(
                    J=J,
                    Q=Q,
                    shape=T,
                    max_order=2,
                    out_type='array'
                )
                window_start = max(0, len(valid_series)-32)  # 鍙栨渶鍚?2涓偣
                padded = valid_series[window_start:window_start+32]
                if len(padded) < 32:
                    padded = np.pad(padded, (0, 32-len(padded)), mode='edge')
                S = scattering(padded.reshape(1, -1).astype(np.float32))

            # === 鐔电壒寰?=== 
            if len(valid_series) >= 100:
                try:
                    # 鏍锋湰鐔?                    features['sampen'] = nolds.sampen(valid_series)
                    
                    # 澶氬昂搴︾喌
                    features['msentropy'] = multiscale_entropy(valid_series, max_scale=3)
                except Exception as e:
                    print(f"鐔佃绠楀紓甯? {str(e)}")
                    features['sampen'] = 0
                    features['msentropy'] = 0
            else:
                features.update({
                    'sampen': 0,
                    'msentropy': 0
                })
    
            # 鏈€缁堝厹搴曞鐞?            for k in ['dtw_distance', 'pre_post_correlation', 'Amax', 'Amin', 'TAmin', 'T0', 'AAC', 'RAC', 'SCV']:
                features[k] = features.get(k, 0.0)
                if pd.isna(features[k]):
                    features[k] = 0.0
                features[k] = float(features[k])

            return pd.Series(features)


        print("\n=== 鏁版嵁棰勫鐞?===")
        # 鍏堝垝鍒嗗熀纭€鏁版嵁
        base_train, base_test = train_test_split(
            df,
            test_size=0.2,
            stratify=df['label'],
            random_state=RANDOM_STATE
        )

        # 搴旂敤鐗瑰緛宸ョ▼
        X_train = base_train[time_series_columns].apply(extract_pupil_features, axis=1)
        X_test = base_test[time_series_columns].apply(extract_pupil_features, axis=1)

        # 鐗瑰緛鏍蜂緥楠岃瘉
        print("鐗瑰緛鎻愬彇鍚嶯aN妫€鏌ワ細")
        print(X_train.isna().sum().sum())
        print("\n=== 鐗瑰緛鏍蜂緥楠岃瘉 ===")
        print("绗竴涓牱鏈師濮嬫椂搴忔暟鎹?")
        print(base_train[time_series_columns].iloc[0].describe())
        print("\n鎻愬彇鍚庣殑鐗瑰緛鏍蜂緥:")
        sample_features = X_train.iloc[0]
        print(sample_features)
        print("鐗瑰緛NaN鍊兼鏌?", sample_features.isna().sum())

        # === 鏁版嵁娓呮礂 ===
        print("\n=== 鏁版嵁娓呮礂 ===")
        # 鑾峰彇鐗瑰緛鍚嶇О锛堝湪鍒犻櫎闆舵柟宸壒寰佸墠锛?        feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else [f"鐗瑰緛_{i}" for i in range(X_train.shape[1])]
        # 缂哄け鍊煎鐞?        imputer = SimpleImputer(strategy='median')
        X_train = np.asarray(X_train, dtype=np.float64)
        X_test = np.asarray(X_test, dtype=np.float64)

        train_nan_count = np.sum(np.isnan(X_train))
        test_nan_count = np.sum(np.isnan(X_test))

        print("\n=== 鏁版嵁娓呮礂楠岃瘉 ===")
        print(f"璁粌闆哊aN鏁伴噺: {train_nan_count}")
        print(f"娴嬭瘯闆哊aN鏁伴噺: {test_nan_count}")

        nan_mask_train = np.isnan(X_train)
        nan_mask_test = np.isnan(X_test)

        if np.any(nan_mask_train):
            print("\n=== 璁粌闆哊aN浣嶇疆 ===")
            print("NaN鏁伴噺:", np.sum(nan_mask_train))
            print("NaN鎵€鍦ㄥ垪:", np.unique(np.where(nan_mask_train)[1]))

        if np.any(nan_mask_test):
            print("\n=== 娴嬭瘯闆哊aN浣嶇疆 ===")
            print("NaN鏁伴噺:", np.sum(nan_mask_test))
            print("NaN鎵€鍦ㄥ垪:", np.unique(np.where(nan_mask_test)[1]))

        assert not np.any(nan_mask_train), "璁粌闆嗗瓨鍦∟aN鍊硷紝璇锋鏌ョ壒寰佹彁鍙栨楠?
        assert not np.any(nan_mask_test), "娴嬭瘯闆嗗瓨鍦∟aN鍊硷紝璇锋鏌ョ壒寰佹彁鍙栨楠?

        # === 鏍囧噯鍖?===
        scaler = StandardScaler(with_mean=True, with_std=True)
        # 璁＄畻璁粌闆嗗悇鐗瑰緛鏂瑰樊
        train_var = np.nanvar(X_train, axis=0)
        zero_var_mask = (train_var == 0) | np.isnan(train_var)

        if np.any(zero_var_mask):
            print(f"\n=== 鍙戠幇{zero_var_mask.sum()}涓浂鏂瑰樊鐗瑰緛 ===")
            print("鍙楀奖鍝嶇殑鐗瑰緛绱㈠紩:", np.where(zero_var_mask)[0])

            shap_feature_names = feature_names

            # 鍒犻櫎闆舵柟宸壒寰?            X_train = np.delete(X_train, np.where(zero_var_mask), axis=1)
            X_test = np.delete(X_test, np.where(zero_var_mask), axis=1)

            # 鏇存柊鐗瑰緛鏁伴噺鏄剧ず
            shap_feature_names = [shap_feature_names[i] for i in range(len(shap_feature_names)) if not zero_var_mask[i]]
            print(f"鍒犻櫎鍚庣壒寰佹暟閲? {X_train.shape[1]}")

        # 鎵ц鏍囧噯鍖?        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        # === 鏂板鏍囧噯鍖栧悗妫€鏌?===
        print("\n=== 鏍囧噯鍖栧悗楠岃瘉 ===")
        print("璁粌闆嗘爣鍑嗗樊:", np.nanstd(X_train, axis=0).round(2))
        print("娴嬭瘯闆嗘爣鍑嗗樊:", np.nanstd(X_test, axis=0).round(2))
        # 鏈€缁堝厹搴曞鐞?        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        # 娣诲姞鏁板€艰鍓紙淇濇寔鍘熷閫昏緫锛?        X_train = np.clip(X_train, -5, 5)

        print("\n=== 鏍囧噯鍖栭獙璇?===")
        print("璁粌闆嗗潎鍊?", X_train.mean(axis=0).round(2))
        print("娴嬭瘯闆嗗潎鍊?", X_test.mean(axis=0).round(2))

        # === 鏁版嵁楠岃瘉 ===
        print("\n=== 鏁版嵁楠岃瘉 ===")

        # 纭繚鏁版嵁绫诲瀷姝ｇ‘
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(base_train['label'].values).ravel()
        y_test = np.asarray(base_test['label'].values).ravel()

        # 妫€鏌ョ被鍒钩琛?        print("\n=== 绫诲埆骞宠　妫€鏌?===")
        print("鍘熷绫诲埆鍒嗗竷:", np.bincount(y_train))

        # 杩囬噰鏍峰皯鏁扮被锛堜粎鍦ㄩ渶瑕佹椂锛?        if len(np.bincount(y_train)) > 1:  # 纭繚瀛樺湪涓や釜绫诲埆
            class_0 = np.bincount(y_train)[0]
            class_1 = np.bincount(y_train)[1]
            if abs(class_0 - class_1) > 50:
                from sklearn.utils import resample

                # 鍒嗙涓や釜绫诲埆
                X_0 = X_train[y_train == 0, :]
                X_1 = X_train[y_train == 1, :]
                y_0 = y_train[y_train == 0]
                y_1 = y_train[y_train == 1]

                # 杩囬噰鏍峰皯鏁扮被
                if abs(class_0 - class_1) > 50:
                    # 鏂板闃插尽鎬х紪绋嬫鏌?                    min_samples = 1  # 蹇呴』鑷冲皯鏈変袱涓牱鏈墠鑳介噸閲囨牱

                    counts = np.bincount(y_train)
                    counts = np.append(counts, [0] * (2 - len(counts)))  # 纭繚鏈?涓厓绱?                    class_0 = int(counts[0])  # 杞崲涓篿nt绫诲瀷
                    class_1 = int(counts[1]) if len(counts) > 1 else 0  # 澶勭悊鍗曠被鍒儏鍐?                    min_samples = 1

                    if class_0 == 0 or class_1 == 0:
                        print(f"璀﹀憡锛氬瓨鍦ㄧ┖绫诲埆锛?绫绘牱鏈暟={class_0}, 1绫绘牱鏈暟={class_1}锛夛紝璺宠繃杩囬噰鏍?)
                    elif abs(class_0 - class_1) > 50:  # 鍘熸潯浠跺垽鏂?                        if class_0 < min_samples or class_1 < min_samples:  # 鏂板鏈€浣庢牱鏈鏌?                            print(f"璀﹀憡锛氭煇绫诲埆鏍锋湰鏁颁笉瓒硔min_samples}涓紙0绫?{class_0}, 1绫?{class_1}锛夛紝璺宠繃杩囬噰鏍?)
                    else:
                        try:
                            if class_0 > class_1:
                                if len(X_1) == 0:
                                    raise ValueError("閿欒锛氱被鍒?鏍锋湰鏁颁负闆讹紝鏃犳硶杩囬噰鏍?)
                                # 鉁?鏄惧紡绫诲瀷鏂█纭繚杩斿洖鍙凯浠ｅ璞?                                resampled_data = resample(X_1, y_1, replace=True, n_samples=class_0, random_state=RANDOM_STATE)
                                assert resampled_data is not None, "resample杩斿洖None锛屾鏌ヨ緭鍏ユ暟鎹?
                                X_1_resampled, y_1_resampled = resampled_data
                                X_train = np.vstack((X_0, X_1_resampled))
                                y_train = np.concatenate((y_0, y_1_resampled))
                            else:
                                if len(X_0) == 0:
                                    raise ValueError("閿欒锛氱被鍒?鏍锋湰鏁颁负闆讹紝鏃犳硶杩囬噰鏍?)
                                resampled_data = resample(X_0, y_0, replace=True, n_samples=class_1, random_state=RANDOM_STATE)
                                assert resampled_data is not None, "resample杩斿洖None锛屾鏌ヨ緭鍏ユ暟鎹?
                                X_0_resampled, y_0_resampled = resampled_data
                                X_train = np.vstack((X_0_resampled, X_1))
                                y_train = np.concatenate((y_0_resampled, y_1))
                            print("杩囬噰鏍峰悗鍒嗗竷:", np.bincount(y_train))
                        except ValueError as e:
                            print(f"杩囬噰鏍峰け璐? {str(e)}")

        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int64)

        print(f"X_train dtype: {X_train.dtype}, shape: {X_train.shape}")
        print(f"y_train dtype: {y_train.dtype}, unique: {np.unique(y_train)}")

        # 鍒嗘妫€鏌?        nan_count_train = np.isnan(X_train).sum()
        nan_count_test = np.isnan(X_test).sum()

        print(f"璁粌闆哊aN鏁伴噺: {nan_count_train}")
        print(f"娴嬭瘯闆哊aN鏁伴噺: {nan_count_test}")

        if nan_count_train > 0:
            raise ValueError("璁粌闆嗗瓨鍦∟aN鍊?璇锋鏌ユ暟鎹澶勭悊姝ラ")
        if nan_count_test > 0:
            raise ValueError("娴嬭瘯闆嗗瓨鍦∟aN鍊?璇锋鏌ユ暟鎹澶勭悊姝ラ")

        print("璁粌闆嗗舰鐘?", X_train.shape)
        print("娴嬭瘯闆嗗舰鐘?", X_test.shape)
        # assert X_train.shape[1] > 40, f"鐗瑰緛鏁伴噺涓嶈冻锛屽疄闄呬负{X_train.shape[1]}"
        # assert X_test.shape[1] > 40, f"鐗瑰緛鏁伴噺涓嶈冻锛屽疄闄呬负{X_test.shape[1]}"

        print("\n=== 鐢熸垚鐗瑰緛鐑姏鍥?===")
        plot_feature_heatmap(X_train, y_train, 'train_heatmap.png')
        plot_feature_heatmap(X_test, y_test, 'test_heatmap.png')

        # === 鍒嗙被鍣ㄥ畾涔?===
        print("\n=== 瀹氫箟鍩哄垎绫诲櫒 ===")

        # CatBoost
        clf1 = cb.CatBoostClassifier(
            depth=6,
            learning_rate=0.1,
            silent=True,
            random_state=RANDOM_STATE
        )

        # AdaBoost
        clf5 = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=100,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
        clf5.__name__ = 'AdaBoost'

        # ExtraTrees
        clf7 = ExtraTreesClassifier(
            max_depth=15,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        clf7.__name__ = 'ExtraTrees'

        # 楂樻柉杩囩▼鍒嗙被鍣紝GaussianProcess
        clf14 = Pipeline([
            ('scaler', StandardScaler()),
            ('gpc', GaussianProcessClassifier(
            
            random_state=RANDOM_STATE,
               
            ))
        ])
        clf14.__name__ = 'GaussianProcess'

        # 璐濆彾鏂ā鍨?        clf15 = GaussianNB()

        clf15.__name__ = 'Bayesian'

        model_pool = {
            'CatBoost': clf1,
            'AdaBoost': clf5,
            'ExtraTrees': clf7,
            'GaussianProcess': clf14,
            'Bayesian': clf15,
        }

        # === 棰勯獙璇佹ā鍨嬫睜 ===
        print("\n=== 棰勯獙璇佹ā鍨嬫睜 ===")
        valid_models = {}
        for name, model in model_pool.items():
            try:
                # clone
                cloned_model = clone(model)

                # 妫€鏌ュ熀纭€鎺ュ彛
                if not (hasattr(cloned_model, 'fit') and hasattr(cloned_model, 'predict')):
                    raise ValueError("缂哄皯 fit 鎴?predict 鏂规硶")

                # Pipeline 鐗瑰埆妫€鏌?                if isinstance(cloned_model, Pipeline):
                    final_step = cloned_model.steps[-1][1]
                    if not any(hasattr(final_step, attr) for attr in ['predict', 'predict_proba', 'decision_function']):
                        raise ValueError("Pipeline鏈眰缂哄皯棰勬祴鎺ュ彛")

                # cloudpickle 妫€鏌?                #  cloudpickle.dumps(cloned_model)

                valid_models[name] = model
                print(f"{name.ljust(15)} 鉁?瀹屾暣楠岃瘉閫氳繃")
            except Exception as e:
                print(f"{name.ljust(15)} 鉁?楠岃瘉澶辫触: {str(e)}")

        model_pool = valid_models

        # === 鑷姩閫夋嫨鍩哄垎绫诲櫒 ===
        from sklearn.ensemble import VotingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, confusion_matrix

        # === 鑷姩閫夋嫨鍩哄垎绫诲櫒 ===
        selected_model_names = ["CatBoost", "AdaBoost", "ExtraTrees", "GaussianProcess", "Bayesian"]
        estimators = [(name, clone(model_pool[name])) for name in selected_model_names]

        print("\n=== 鍩哄垎绫诲櫒鎬ц兘璇勪及 ===")
        scores = []
        for name, est in estimators:
            model_type = MODEL_TYPE_MAP.get(name, 'default')
            preprocessor = create_model_specific_preprocessing(model_type)
            
            model_pipe = Pipeline([
                ('preprocess', preprocessor),
                ('estimator', clone(est))
            ])
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            with parallel_backend('dask'):
                cv_scores = cross_val_score(
                    model_pipe, X_train, y_train,
                    cv=cv, scoring='accuracy', n_jobs=1
                )
            
            mean_score = np.mean(cv_scores)
            scores.append((name, est, mean_score))
            print(f"{name}: 骞冲潎鍑嗙‘鐜?= {mean_score:.4f}")

        # 鎬ц兘鎺掑簭
        sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)
        print("\n=== 鍩哄垎绫诲櫒鎬ц兘鎺掑悕 ===")
        for idx, (name, _, score) in enumerate(sorted_scores):
            print(f"绗瑊idx+1}鍚? {name.ljust(20)} | CV鍑嗙‘鐜? {score:.4f}")
            
        # === 鏋勫缓杞姇绁ㄦā鍨?===
        print("\n=== 鍚敤鍔犳潈鎶曠エ鏋舵瀯 ===")

        calibrated_estimators = []
        for name, model in estimators:
            if hasattr(model, "predict_proba"):
                calibrated_estimators.append((name, model))
                print(f"{name} 鉁?鏀寔 predict_proba")
            else:
                print(f"{name} 鈿?涓嶆敮鎸?predict_proba锛屼娇鐢?CalibratedClassifierCV 鍖呰")
                calibrated_model = CalibratedClassifierCV(base_estimator=model, cv=3)
                calibrated_estimators.append((name, calibrated_model))

        # 鏍规嵁CV鍒嗘暟鐢熸垚鏉冮噸 ===
        print("\n=== 鐢熸垚闆嗘垚鏉冮噸 ===")
        # 浠庡疄闄匔V缁撴灉鑾峰彇鍒嗘暟锛堟浛鎹㈠垎鏁帮級
        cv_scores = {
            'CatBoost': 0.7318,
            'GaussianProcess': 0.7494,
            'AdaBoost': 0.6447,
            'ExtraTrees': 0.7482,
            'Bayesian': 0.7189
        }
        # 鑷姩璁＄畻鏉冮噸
        total = sum(cv_scores.values())
        weights = {k: round(v / total, 4) for k, v in cv_scores.items()}

        # 纭繚涓巆alibrated_estimators椤哄簭鍖归厤
        print("妯″瀷鏉冮噸鍒嗛厤:")
        for name, _ in calibrated_estimators:
            print(f"  {name.ljust(15)}: {weights.get(name, 0.0):.4f}")

        # === 淇敼鎶曠エ鍣?===
        final_model = VotingClassifier(
            estimators=calibrated_estimators,
            voting='soft',
            weights=[weights.get(name, 1.0) for name, _ in calibrated_estimators],  # 鏂板鏉冮噸鍙傛暟
            n_jobs=4
        )

        print(calibrated_estimators)
        # === 璁粌鍜岃瘎浼?===
        print("\n=== 寮€濮嬪苟琛岃缁?===")
        final_model.fit(X_train, y_train)

        start_train = time.time()  # 娣诲姞璁粌寮€濮嬫椂闂?        final_model.fit(X_train, y_train)
        end_train = time.time()  # 娣诲姞璁粌缁撴潫鏃堕棿
        train_time = end_train - start_train
        print(f"璁粌鑰楁椂: {train_time:.2f}绉?)  # 鎵撳嵃璁粌鏃堕棿

        print("\n=== 杞姇绁ㄦā鍨嬭缁冨畬鎴?===")
        print("\n=== 寮€濮嬮娴?===")
        start_predict = time.time()  # 棰勬祴寮€濮嬭鏃?        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)[:, 1]
        end_predict = time.time()  # 棰勬祴缁撴潫璁℃椂
        predict_time = end_predict - start_predict
        print(f"棰勬祴鑰楁椂: {predict_time:.4f}绉?)

        print("\n=== 鏈€缁堟祴璇曢泦璇勪及 ===")
        print(classification_report(y_test, y_pred, digits=4))
        print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}")
        print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")

        print("\n=== 绫诲埆鐗瑰紓鎬ц〃鐜?===")
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f"绫诲埆0 Recall: {tn / (tn + fp):.2%}")
        print(f"绫诲埆1 Recall: {tp / (tp + fn):.2%}")

        # ===== 缁樺埗ROC鏇茬嚎 =====
        print("\n=== 缁樺埗ROC鏇茬嚎 ===")
        # 鑾峰彇姣忎釜绫诲埆鐨勯娴嬫鐜?        y_score = final_model.predict_proba(X_test)
        
        # 妫€鏌ラ娴嬫鐜囩殑褰㈢姸
        print(f"棰勬祴姒傜巼褰㈢姸: {y_score.shape}")
        
        # 纭繚鎴戜滑鏈夋纭殑姒傜巼鏍煎紡
        if y_score.shape[1] == 1:
            # 濡傛灉鍙湁涓€鍒楁鐜囷紝鍒欏亣璁捐繖鏄绫绘鐜?            pos_class_proba = y_score[:, 0]
            neg_class_proba = 1 - pos_class_proba
            y_score = np.column_stack((neg_class_proba, pos_class_proba))
            print(f"璋冩暣鍚庣殑棰勬祴姒傜巼褰㈢姸: {y_score.shape}")
        
        # 涓烘瘡涓被鍒绠桼OC鏇茬嚎
        plt.figure(figsize=(10, 8))
        
        # 绫诲埆0鐨凴OC鏇茬嚎
        fpr0, tpr0, _ = roc_curve(y_test, y_score[:, 0], pos_label=0)
        auc0 = auc(fpr0, tpr0)
        plt.plot(fpr0, tpr0, color='green', linestyle='-', 
                 label=f'ROC curve of class0 (AUC={auc0:.4f})')
        
        # 绫诲埆1鐨凴OC鏇茬嚎
        fpr1, tpr1, _ = roc_curve(y_test, y_score[:, 1], pos_label=1)
        auc1 = auc(fpr1, tpr1)
        plt.plot(fpr1, tpr1, color='red', linestyle='-', 
                 label=f'ROC curve of class1 (AUC={auc1:.4f})')
        
        # 璁＄畻瀹忚骞冲潎ROC鏇茬嚎
        mean_fpr = np.linspace(0, 1, 100)
        
        # 鎻掑€兼墍鏈塕OC鏇茬嚎
        all_tprs = []
        tpr0_interp = np.interp(mean_fpr, fpr0, tpr0)
        tpr0_interp[0] = 0.0
        all_tprs.append(tpr0_interp)
        
        tpr1_interp = np.interp(mean_fpr, fpr1, tpr1)
        tpr1_interp[0] = 0.0
        all_tprs.append(tpr1_interp)
        
        mean_tpr = np.mean(all_tprs, axis=0)
        mean_tpr[-1] = 1.0
        macro_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='blue', linestyle='--', 
                 label=f'macro avg ROC curve (AUC={macro_auc:.4f})')
        
        # 璁＄畻鍔犳潈骞冲潎ROC鏇茬嚎
        n_samples = len(y_test)
        class0_count = np.sum(y_test == 0)
        class1_count = np.sum(y_test == 1)
        weights = [class0_count / n_samples, class1_count / n_samples]
        
        weighted_mean_tpr = np.zeros_like(mean_fpr)
        weighted_mean_tpr += weights[0] * tpr0_interp
        weighted_mean_tpr += weights[1] * tpr1_interp
        weighted_auc = auc(mean_fpr, weighted_mean_tpr)
        plt.plot(mean_fpr, weighted_mean_tpr, color='orange', linestyle='-.', 
                 label=f'weighted avg ROC curve (AUC={weighted_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 淇濆瓨鍥惧儚
        plt.savefig('roc_curves.png', dpi=300)
        plt.close()
        print("ROC鏇茬嚎宸蹭繚瀛樹负 'roc_curves.png'")

        # === SHAP鐗瑰緛閲嶈鎬у垎鏋?===
        if shap_available:
            print("\n=== 寮€濮婼HAP鐗瑰緛閲嶈鎬у垎鏋?===")
            try:
                if 'shap_feature_names' not in globals():
                    shap_feature_names = [f"鐗瑰緛_{i}" for i in range(X_train.shape[1])]
                
                # 鏀逛负浣跨敤KernelExplainer骞舵槑纭彁渚涢娴嬪嚱鏁?                def predict_proba_wrapper(X):
                    return final_model.predict_proba(X)
                
                # 鍒涘缓SHAP瑙ｉ噴鍣?- 浣跨敤鏈€鏂癆PI鏍煎紡
                explainer = shap.KernelExplainer(
                    model=predict_proba_wrapper, 
                    data=(shap.sample(X_train, 10))  #淇敼100鈫?0锛?0鈫?(shap.sample(X_train, 100))
                )
                
                # 璁＄畻SHAP鍊硷紙浣跨敤娴嬭瘯闆嗙殑瀛愰泦锛?                sample_indices = np.random.choice(X_test.shape[0], size=min(5, X_test.shape[0]), replace=False)   #淇敼锛?np.random.choice(X_test.shape[0], size=min(50, X_test.shape[0]), replace=False)
                X_test_sample = X_test[sample_indices]
                
                start_shap = time.time()
                shap_values = explainer.shap_values(X_test_sample)
                end_shap = time.time()
                print(f"SHAP鍊艰绠楀畬鎴愶紝鑰楁椂: {end_shap - start_shap:.2f}绉?)
                
                # 澶勭悊SHAP鍊兼牸寮?- 淇浜屽厓鍒嗙被褰㈢姸闂
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    # 浜屽厓鍒嗙被锛氬彇绗簩绫诲埆鐨凷HAP鍊?                    shap_values_positive = shap_values[1]
                elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                    # 涓夌淮鏁扮粍锛氬彇绗簩涓€氶亾鐨勫€硷紙姝ｇ被鍒級
                    shap_values_positive = shap_values[:, :, 1]
                else:
                    # 鍏朵粬鏍煎紡鐨勫鐞?                    shap_values_positive = shap_values

                # 纭繚SHAP鍊兼槸浜岀淮鏁扮粍
                if len(shap_values_positive.shape) == 3:
                    # 濡傛灉浠嶆槸涓夌淮锛屽彇绗竴涓€氶亾鍏滃簳澶勭悊
                    shap_values_positive = shap_values_positive[:, :, 0]
                
                # 褰㈢姸楠岃瘉
                if shap_values_positive.shape != X_test_sample.shape:
                    expected_shape = (len(X_test_sample), len(shap_feature_names))
                    actual_shape = shap_values_positive.shape
                    raise ValueError(
                        f"SHAP鍊肩煩闃靛舰鐘朵笉鍖归厤: 鏈熸湜{expected_shape}, 瀹為檯{actual_shape}\n"
                        f"鍙兘鍘熷洜: 鐗瑰緛閫夋嫨鍚庣壒寰佹暟閲忎笉涓€鑷?
                    )
            
                # 1. 鐢熸垚骞朵繚瀛樻憳瑕佸浘
                print("鐢熸垚SHAP鎽樿鍥?..")
                plt.figure(figsize=(12, 10))
                shap.summary_plot(
                    shap_values_positive,  # 姝ｇ被鍒殑SHAP鍊?                    X_test_sample,
                    feature_names=shap_feature_names,
                    max_display=15,
                    show=False
                )
            
                plt.title("SHAP鐗瑰緛閲嶈鎬ф憳瑕?)
                plt.tight_layout()
                plt.savefig('shap_summary.png', dpi=300)
                plt.close()
                
                # 2. 鐢熸垚骞朵繚瀛樻潯褰㈠浘
                print("鐢熸垚SHAP鏉″舰鍥?..")
                plt.figure(figsize=(12, 10))
                shap.summary_plot(
                    shap_values_positive,
                    X_test_sample,
                    feature_names=shap_feature_names,
                    plot_type="bar",
                    max_display=15,
                    show=False
                )
                plt.title("SHAP鐗瑰緛閲嶈鎬э紙鍧囧€紎SHAP鍊紎锛?)
                plt.tight_layout()
                plt.savefig('shap_bar.png', dpi=300)
                plt.close()
                
                # 3. 鐢熸垚骞朵繚瀛樹緷璧栧浘锛堝墠3涓渶閲嶈鐨勭壒寰侊級
                print("鐢熸垚SHAP渚濊禆鍥?..")
                plt.figure(figsize=(15, 10))

                # === 4. 鏂板缁勫悎鍥?鐗瑰緛閲嶈鎬?鍒嗗竷寮忓奖鍝嶅姏鍒嗘瀽) ===
                print("鐢熸垚SHAP缁勫悎鍥撅紙宸﹀彸缁撴瀯锛?..")

                # 璁＄畻鐗瑰緛骞冲潎|SHAP|鍊煎苟闄嶅簭鎺掑簭锛堝彇鍓?5涓級
                mean_abs_shap = np.abs(shap_values_positive).mean(0)

                n_features = min(15, len(mean_abs_shap))
                sorted_idx = np.argsort(mean_abs_shap)[-n_features:][::-1]   # 鍙栧墠15涓渶閲嶈鐨勭壒寰侊紙闄嶅簭锛?                sorted_features = [shap_feature_names[i] for i in sorted_idx]
                mean_abs_shap_sorted = mean_abs_shap[sorted_idx]
                shap_values_sorted = shap_values_positive[:, sorted_idx] 
                X_test_sorted = X_test_sample[:, sorted_idx]  

                # 璁＄畻鐧惧垎姣旇础鐚?                total_shap = mean_abs_shap_sorted.sum()
                percent_contrib = (mean_abs_shap_sorted / total_shap) * 100
                
                # === 淇濇寔闄嶅簭鎺掑垪锛堟渶閲嶈鍦ㄩ《閮級锛屾棤闇€鍙嶈浆 ===
                y_pos_desc = np.arange(n_features-1, -1, -1)  # 浠庨珮鍒颁綆: [14,13,...,0]

                # ===== 鍒涘缓 Figure锛屼互鍙婁笂灞傦紙ax_sw锛夊拰涓嬪眰锛坅x_bar锛変袱涓叡浜?y 杞寸殑鍧愭爣绯?=====
                fig = plt.figure(figsize=(15, 8), dpi=300)

                # 2.1 涓婂眰鍧愭爣绯?ax_sw锛岀敤鏉ョ敾 SHAP beeswarm
                ax_sw = fig.add_axes([0.32, 0.11, 0.59, 0.77])  # 宸﹁竟璺濆姞澶э紝瀹藉害鐩稿簲缂╁皬

                # 2.2 浠?ax_sw 涓哄熀鍑嗗啀鍒涘缓涓€涓叡浜?y 杞寸殑鏂板潗鏍囩郴 ax_bar锛岀敤鏉ョ敾 barh
                ax_bar = ax_sw.twiny()

                # 2.3 璁剧疆鍥惧眰椤哄簭
                ax_bar.set_zorder(0)  # 鏉″舰鍥惧湪涓嬪眰
                ax_sw.set_zorder(1)   # 鐐逛簯鍥惧湪涓婂眰
                ax_sw.patch.set_alpha(0)  # 璁剧疆涓婂眰鑳屾櫙閫忔槑

                # ===== 鍦ㄤ笅灞?ax_bar 涓婄敾 Mean(|SHAP|) 鐨勬按骞虫潯褰㈠浘 barh =====
                # === 浣跨敤y_pos_desc鏇夸唬y_pos ===
                bar_plot = ax_bar.barh(
                    y=y_pos_desc,
                    width=mean_abs_shap_sorted,
                    height=0.6,
                    color="lightpink",
                    alpha=0.3,  # 璁剧疆閫忔槑搴?                    edgecolor="none",
                    zorder=0
                )

                # 3.1 璁剧疆 ax_bar 鐨?x 杞磋寖鍥?                xlim_bar = mean_abs_shap_sorted.max() * 1.05
                ax_bar.set_xlim(0, xlim_bar)

                # 3.2 璁剧疆 x 杞村埢搴?                xticks_bar = np.linspace(0, xlim_bar, 5)
                ax_bar.set_xticks(xticks_bar)
                ax_bar.set_xticklabels([f"{x:.4f}" for x in xticks_bar], fontsize=10)
                ax_bar.set_xlabel("Mean (|SHAP| value)", fontsize=10)

                # 3.3 璁剧疆 y 杞村埢搴?                ax_bar.set_yticks(y_pos_desc)
                ax_bar.set_yticklabels(sorted_features, fontsize=12)
                ax_bar.tick_params(axis='y', length=4)

                # === 璁剧疆y杞磋寖鍥达紝纭繚鏈€閲嶈鐨勭壒寰佸湪椤堕儴 ===
                ax_bar.set_ylim(-0.5, n_features-0.5)   # 姝ｅ父鑼冨洿
                ax_sw.set_ylim(-0.5, n_features-0.5)   # 鍚屾鍙嶈浆鐐逛簯鍥剧殑y杞?
                # ===== 鍦ㄤ笂灞?ax_sw 涓婄敾 SHAP Beeswarm =====
                # 鍔ㄦ€佽绠楃偣浜戝浘鑼冨洿
                max_sw = np.max(np.abs(shap_values_sorted)) * 1.05
                ax_sw.set_xlim(-0.3, 0.3) # 鐐逛簯鍥剧殑宸﹁竟鐣岋細-max_sw, max_sw
                
                sw_xticks = np.linspace(-0.3, 0.3, 5) # 鐐逛簯鍥剧殑宸﹁竟鐣岋細-max_sw, max_sw, 5
                ax_sw.set_xticks(sw_xticks)
                ax_sw.set_xticklabels([f"{x:.4f}" for x in sw_xticks], fontsize=10)
                ax_sw.set_xlabel("SHAP value (impact on model output)", fontsize=10)

                # 鏋勯€?Explanation 瀵硅薄
                expl = shap.Explanation(
                    values=shap_values_sorted,
                    data=X_test_sorted,
                    feature_names=sorted_features
                )
                feature_indices = list(range(len(sorted_features)))

                # 璋冪敤 beeswarm
                shap.plots.beeswarm(
                    expl,
                    ax=ax_sw,
                    show=False,
                    plot_size=None,
                    color=plt.get_cmap('coolwarm'),
                    order=feature_indices,
                    max_display=n_features
                )

                # 鍦?bar 鍐呴儴鏍囨敞 mean(|SHAP|) 鏁板€?                for i, (v, p) in enumerate(zip(mean_abs_shap_sorted, percent_contrib)):
                    ax_bar.text(
                        x=0.01 * xlim_bar,
                        y=y_pos_desc[i],  # 浣跨敤鍊掑簭浣嶇疆绱㈠紩
                        s=f"{v:.4f} ({p:.1f}%)",
                        va="center",
                        ha="left",
                        fontsize=10,
                        color="black"
                    )

                #plt.suptitle('SHAP鐗瑰緛鍒嗘瀽 - 缁勫悎鍥?, fontsize=16, y=0.95)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig('shap_combined_horizontal.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("SHAP缁勫悎鍥撅紙宸﹀彸缁撴瀯锛夊凡淇濆瓨涓簊hap_combined_horizontal.png")

                # 鑾峰彇鐗瑰緛閲嶈鎬ф帓鍚?
            except Exception as e:
                print(f"SHAP鍒嗘瀽鍑洪敊: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("璺宠繃SHAP鍒嗘瀽锛屾湭瀹夎SHAP搴?)

    finally:
        client.close()
        print("\n=== Dask闆嗙兢宸插叧闂?===")

