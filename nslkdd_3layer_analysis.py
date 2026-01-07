"""
NSL-KDD 3層構造分析：Isolation Forestスコア分布の可視化
- 横軸: Isolation Forestスコア
- 縦軸: データ数（積み上げ棒グラフ：下=異常、上=正常）
- 最適な分位点（q_b=16%, q_c=3%）での境界線を表示
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime

# 日本語フォント設定（macOS）
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 設定
# ============================================================
BASE_DIR = "/Users/shoutarou/Library/Mobile Documents/com~apple~CloudDocs/授業/3年/後期/プロ研/AnomalyDetection3Layer"
DATA_DIR = f"{BASE_DIR}/nsl-kdd"
OUTPUT_DIR = BASE_DIR
RANDOM_STATE = 42

# 最適化結果から取得した分位点
BEST_Q_B = 16  # A層境界（上位16%）
BEST_Q_C = 3   # C層境界（下位3%）

COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# ============================================================
# データ読み込み
# ============================================================
print("=" * 60)
print("NSL-KDD 3層構造分析：スコア分布可視化")
print("=" * 60)

print("\nデータ読み込み中...")
train_df = pd.read_csv(f"{DATA_DIR}/KDDTrain+.txt", names=COLUMN_NAMES)
test_df = pd.read_csv(f"{DATA_DIR}/KDDTest+.txt", names=COLUMN_NAMES)

# ラベル二値化
train_df['is_attack'] = (train_df['label'] != 'normal').astype(int)
test_df['is_attack'] = (test_df['label'] != 'normal').astype(int)

# カテゴリ変数エンコード
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    le.fit(train_df[col])
    train_df[col] = le.transform(train_df[col])
    test_values = test_df[col].copy()
    known_mask = test_values.isin(le.classes_)
    test_df[col] = -1
    test_df.loc[known_mask, col] = le.transform(test_values[known_mask])

feature_cols = [c for c in COLUMN_NAMES if c not in ['label', 'difficulty']]

X_train = train_df[feature_cols].values
y_train = train_df['is_attack'].values
X_test = test_df[feature_cols].values
y_test = test_df['is_attack'].values

# ============================================================
# Isolation Forestでスコア計算
# ============================================================
print("Isolation Forestスコア計算中...")

X_train_normal = X_train[y_train == 0]
if len(X_train_normal) > 50000:
    sample_idx = np.random.choice(len(X_train_normal), 50000, replace=False)
    X_train_normal_sample = X_train_normal[sample_idx]
else:
    X_train_normal_sample = X_train_normal

iso_forest = IsolationForest(contamination=0.1, random_state=RANDOM_STATE, n_jobs=-1)
iso_forest.fit(X_train_normal_sample)

# スコア計算
train_scores = iso_forest.score_samples(X_train)
test_scores = iso_forest.score_samples(X_test)
normal_scores = iso_forest.score_samples(X_train_normal_sample)

# 境界値計算
threshold_b = np.percentile(normal_scores, 100 - BEST_Q_B)  # A層境界
threshold_c = np.percentile(normal_scores, BEST_Q_C)        # C層境界

print(f"A層境界（q_b={BEST_Q_B}%）: {threshold_b:.4f}")
print(f"C層境界（q_c={BEST_Q_C}%）: {threshold_c:.4f}")

# ============================================================
# グラフ作成関数
# ============================================================
def create_score_distribution_chart(scores, labels, threshold_b, threshold_c,
                                     title, filename, n_bins=50):
    """
    Isolation Forestスコアの分布を積み上げ棒グラフで表示
    下部分：異常（攻撃）、上部分：正常
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # スコアの範囲を決定
    score_min = min(scores)
    score_max = max(scores)
    bins = np.linspace(score_min, score_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    # 正常と異常を分離
    normal_scores = scores[labels == 0]
    attack_scores = scores[labels == 1]

    # ヒストグラム計算
    normal_counts, _ = np.histogram(normal_scores, bins=bins)
    attack_counts, _ = np.histogram(attack_scores, bins=bins)

    # 積み上げ棒グラフ（下：異常、上：正常）
    ax.bar(bin_centers, attack_counts, width=bin_width * 0.9,
           color='#E74C3C', label='異常（攻撃）', alpha=0.85)
    ax.bar(bin_centers, normal_counts, width=bin_width * 0.9,
           bottom=attack_counts, color='#3498DB', label='正常', alpha=0.85)

    # 境界線を描画
    y_max = max(normal_counts + attack_counts) * 1.15

    # C層境界（左側）
    ax.axvline(x=threshold_c, color='#E67E22', linewidth=2.5, linestyle='--',
               label=f'C層境界（q_c={BEST_Q_C}%）')
    ax.fill_betweenx([0, y_max], score_min, threshold_c, alpha=0.12, color='#E67E22')

    # A層境界（右側）
    ax.axvline(x=threshold_b, color='#27AE60', linewidth=2.5, linestyle='--',
               label=f'A層境界（q_b={BEST_Q_B}%）')
    ax.fill_betweenx([0, y_max], threshold_b, score_max, alpha=0.12, color='#27AE60')

    # 層ラベルを追加
    ax.text((score_min + threshold_c) / 2, y_max * 0.95, 'C層\n(Anomaly-prone)',
            ha='center', va='top', fontsize=12, fontweight='bold', color='#E67E22')
    ax.text((threshold_c + threshold_b) / 2, y_max * 0.95, 'B層\n(Fluctuation)',
            ha='center', va='top', fontsize=12, fontweight='bold', color='#7F8C8D')
    ax.text((threshold_b + score_max) / 2, y_max * 0.95, 'A層\n(Core Normal)',
            ha='center', va='top', fontsize=12, fontweight='bold', color='#27AE60')

    # 軸設定
    ax.set_xlabel('Isolation Forest スコア', fontsize=14, fontweight='bold')
    ax.set_ylabel('データ数', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, y_max)

    # 凡例
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # グリッド
    ax.grid(axis='y', alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)

    # スタイル調整
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {filename}")

    return normal_counts, attack_counts, bins

# ============================================================
# グラフ生成
# ============================================================
print("\nグラフ生成中...")

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Trainデータの分布
print("\n[1] Trainデータの分布...")
train_normal_counts, train_attack_counts, train_bins = create_score_distribution_chart(
    train_scores, y_train, threshold_b, threshold_c,
    'NSL-KDD Train: Isolation Forestスコア分布',
    f'{OUTPUT_DIR}/nslkdd_score_distribution_train_{TIMESTAMP}.png'
)

# Testデータの分布
print("[2] Testデータの分布...")
test_normal_counts, test_attack_counts, test_bins = create_score_distribution_chart(
    test_scores, y_test, threshold_b, threshold_c,
    'NSL-KDD Test: Isolation Forestスコア分布',
    f'{OUTPUT_DIR}/nslkdd_score_distribution_test_{TIMESTAMP}.png'
)

# ============================================================
# 統計情報の出力
# ============================================================
print("\n" + "=" * 60)
print("統計情報")
print("=" * 60)

def print_layer_stats(scores, labels, threshold_b, threshold_c, data_name):
    """層ごとの統計を出力"""
    normal_mask = labels == 0
    attack_mask = labels == 1

    # 層の割り当て
    layer_a = scores >= threshold_b
    layer_c = scores <= threshold_c
    layer_b = ~layer_a & ~layer_c

    print(f"\n【{data_name}】")
    print(f"  全体: {len(scores):,}件（正常: {normal_mask.sum():,}, 異常: {attack_mask.sum():,}）")

    print(f"\n  A層（スコア >= {threshold_b:.4f}）:")
    print(f"    正常: {(normal_mask & layer_a).sum():,} ({(normal_mask & layer_a).sum() / normal_mask.sum() * 100:.1f}%)")
    print(f"    異常: {(attack_mask & layer_a).sum():,} ({(attack_mask & layer_a).sum() / attack_mask.sum() * 100:.1f}%)")

    print(f"\n  B層（{threshold_c:.4f} < スコア < {threshold_b:.4f}）:")
    print(f"    正常: {(normal_mask & layer_b).sum():,} ({(normal_mask & layer_b).sum() / normal_mask.sum() * 100:.1f}%)")
    print(f"    異常: {(attack_mask & layer_b).sum():,} ({(attack_mask & layer_b).sum() / attack_mask.sum() * 100:.1f}%)")

    print(f"\n  C層（スコア <= {threshold_c:.4f}）:")
    print(f"    正常: {(normal_mask & layer_c).sum():,} ({(normal_mask & layer_c).sum() / normal_mask.sum() * 100:.1f}%)")
    print(f"    異常: {(attack_mask & layer_c).sum():,} ({(attack_mask & layer_c).sum() / attack_mask.sum() * 100:.1f}%)")

print_layer_stats(train_scores, y_train, threshold_b, threshold_c, "Train")
print_layer_stats(test_scores, y_test, threshold_b, threshold_c, "Test")

print(f"\n" + "=" * 60)
print("完了！")
print(f"出力ファイル:")
print(f"  - {OUTPUT_DIR}/nslkdd_score_distribution_train_{TIMESTAMP}.png")
print(f"  - {OUTPUT_DIR}/nslkdd_score_distribution_test_{TIMESTAMP}.png")
