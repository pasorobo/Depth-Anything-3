# ユーティリティツール

点群の統合、ダウンサンプリング、フィルタリング、形式変換などを行うユーティリティスクリプト集です。

## 📦 含まれるツール

### pointcloud_tools.py

点群処理のための包括的なコマンドラインツールです。

## 🚀 クイックスタート

```bash
# 統計情報を表示
python pointcloud_tools.py stats --input cloud.ply

# 複数の点群を統合
python pointcloud_tools.py merge --input *.ply --output merged.ply

# ダウンサンプリング
python pointcloud_tools.py downsample --input cloud.ply --output small.ply --ratio 0.5
```

## 📋 機能一覧

### 1. merge - 複数の点群を統合

複数のPLYまたはNPZファイルを1つの点群に統合します。

```bash
python pointcloud_tools.py merge \
    --input file1.ply file2.ply file3.ply \
    --output merged.ply
```

**オプション:**
- `--input`, `-i`: 入力ファイルのリスト（必須）
- `--output`, `-o`: 出力ファイルパス（必須）

**対応形式:** PLY, NPZ

---

### 2. downsample - 点群のダウンサンプリング

点群のサイズを削減します。2つの方法があります：

#### 方法A: ランダムサンプリング

```bash
python pointcloud_tools.py downsample \
    --input cloud.ply \
    --output downsampled.ply \
    --ratio 0.5  # 50%にダウンサンプリング
```

#### 方法B: ボクセルグリッドダウンサンプリング

```bash
python pointcloud_tools.py downsample \
    --input cloud.ply \
    --output voxel_downsampled.ply \
    --voxel-size 0.05  # 5cmのボクセルサイズ
```

**オプション:**
- `--input`, `-i`: 入力ファイル（必須）
- `--output`, `-o`: 出力ファイル（必須）
- `--ratio`: ダウンサンプリング率（0-1）
- `--voxel-size`: ボクセルサイズ（メートル単位、ratioよりも優先）

---

### 3. filter - 点群のフィルタリング

距離や統計的外れ値で点群をフィルタリングします。

#### 距離フィルタリング

```bash
# 原点から10m以内の点のみ
python pointcloud_tools.py filter \
    --input cloud.ply \
    --output filtered.ply \
    --max-distance 10.0

# 最小距離と最大距離を指定
python pointcloud_tools.py filter \
    --input cloud.ply \
    --output filtered.ply \
    --min-distance 0.5 \
    --max-distance 10.0
```

#### 統計的外れ値除去

```bash
python pointcloud_tools.py filter \
    --input cloud.ply \
    --output cleaned.ply \
    --remove-outliers
```

**オプション:**
- `--input`, `-i`: 入力ファイル（必須）
- `--output`, `-o`: 出力ファイル（必須）
- `--max-distance`: 最大距離（メートル）
- `--min-distance`: 最小距離（メートル）
- `--remove-outliers`: 統計的外れ値を除去

---

### 4. convert - 形式変換

PLYとNPZの間で形式を変換します。

```bash
# NPZ → PLY
python pointcloud_tools.py convert \
    --input cloud.npz \
    --output cloud.ply

# PLY → NPZ
python pointcloud_tools.py convert \
    --input cloud.ply \
    --output cloud.npz
```

**オプション:**
- `--input`, `-i`: 入力ファイル（必須）
- `--output`, `-o`: 出力ファイル（必須）

**対応形式:** PLY ↔ NPZ

---

### 5. stats - 統計情報の表示

点群の統計情報を表示します。

```bash
python pointcloud_tools.py stats --input cloud.ply
```

**出力例:**
```
Point Cloud Statistics:
==================================================
Total points: 1,234,567

Bounding box:
  X: [-5.234, 5.678]
  Y: [-3.456, 4.123]
  Z: [0.123, 8.901]

Center: [0.222, 0.334, 4.512]

Distance from center:
  Mean: 3.456
  Std:  1.234
  Max:  7.890

Color range: 0-255 (uint8)
```

**オプション:**
- `--input`, `-i`: 入力ファイル（必須）

---

## 💡 実用的な使用例

### ワークフロー例1: 複数スキャンの統合と最適化

```bash
# 1. 複数のライブカメラセッションを統合
python pointcloud_tools.py merge \
    --input scan1.ply scan2.ply scan3.ply \
    --output merged.ply

# 2. 統計情報を確認
python pointcloud_tools.py stats --input merged.ply

# 3. ボクセルダウンサンプリングで最適化
python pointcloud_tools.py downsample \
    --input merged.ply \
    --output optimized.ply \
    --voxel-size 0.02

# 4. 外れ値を除去
python pointcloud_tools.py filter \
    --input optimized.ply \
    --output final.ply \
    --remove-outliers

# 5. 最終的な統計を確認
python pointcloud_tools.py stats --input final.ply
```

### ワークフロー例2: 大規模点群の軽量化

```bash
# 1. 元のサイズを確認
python pointcloud_tools.py stats --input huge_cloud.ply

# 2. 遠くの点を除去（10m以内のみ）
python pointcloud_tools.py filter \
    --input huge_cloud.ply \
    --output nearby.ply \
    --max-distance 10.0

# 3. 50%にダウンサンプリング
python pointcloud_tools.py downsample \
    --input nearby.ply \
    --output light.ply \
    --ratio 0.5

# 4. 結果を確認
python pointcloud_tools.py stats --input light.ply
```

### ワークフロー例3: 品質チェックパイプライン

```bash
#!/bin/bash
# quality_check.sh

INPUT=$1
OUTPUT=$2

echo "=== 品質チェック開始 ==="

# ステップ1: 元の統計
echo "[1/4] 元の点群の統計..."
python pointcloud_tools.py stats --input $INPUT

# ステップ2: 外れ値除去
echo "[2/4] 外れ値を除去中..."
python pointcloud_tools.py filter \
    --input $INPUT \
    --output temp_cleaned.ply \
    --remove-outliers

# ステップ3: 距離フィルタリング
echo "[3/4] 距離でフィルタリング中..."
python pointcloud_tools.py filter \
    --input temp_cleaned.ply \
    --output temp_filtered.ply \
    --max-distance 15.0

# ステップ4: ダウンサンプリング
echo "[4/4] ダウンサンプリング中..."
python pointcloud_tools.py downsample \
    --input temp_filtered.ply \
    --output $OUTPUT \
    --voxel-size 0.03

# クリーンアップ
rm temp_cleaned.ply temp_filtered.ply

echo "=== 完了 ==="
python pointcloud_tools.py stats --input $OUTPUT
```

使用例:
```bash
chmod +x quality_check.sh
./quality_check.sh raw_scan.ply final_scan.ply
```

## 🔧 トラブルシューティング

### メモリ不足

巨大な点群を処理する場合：

```bash
# 段階的にダウンサンプリング
python pointcloud_tools.py downsample --input huge.ply --output step1.ply --ratio 0.5
python pointcloud_tools.py downsample --input step1.ply --output step2.ply --ratio 0.5
python pointcloud_tools.py downsample --input step2.ply --output final.ply --ratio 0.5
```

### plyfileがインストールされていない

```bash
pip install plyfile
```

### scipyがインストールされていない（外れ値除去用）

```bash
pip install scipy
```

## 📊 パフォーマンス目安

| 点数 | merge | downsample | filter (outliers) |
|------|-------|------------|------------------|
| 100K | <1秒 | <1秒 | ~2秒 |
| 1M | ~2秒 | ~2秒 | ~10秒 |
| 10M | ~15秒 | ~15秒 | ~2分 |

*Intel i7-10700K, 32GB RAMでの測定値

## 📚 関連サンプル

- [07. 大規模3D再構成](../07_advanced_3d_reconstruction/) - 大規模点群の生成
- [08. ライブカメラ3D再構成](../08_live_camera_reconstruction/) - リアルタイム点群生成

## 💾 データ形式

### PLYファイル

```
ply
format ascii 1.0
element vertex 1000000
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
[点データ...]
```

### NPZファイル

```python
{
    'points': np.ndarray shape=(N, 3),  # XYZ座標
    'colors': np.ndarray shape=(N, 3),  # RGB色（0-255）
}
```
