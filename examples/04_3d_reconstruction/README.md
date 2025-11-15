# 04. 3D再構成

深度マップから3Dポイントクラウドやメッシュを生成します。GLB、PLY、NPZ形式でエクスポート可能です。

## 🚀 クイックスタート

```bash
python 3d_reconstruction.py --input image.jpg --output ./output_3d
```

## 📋 使用方法

### 基本的な使用

```bash
# 単一画像から3D再構成
python 3d_reconstruction.py --input image.jpg --output ./output_3d

# 複数画像から3D再構成
python 3d_reconstruction.py --input images/ --output ./output_3d

# 複数形式でエクスポート
python 3d_reconstruction.py --input image.jpg --format glb-npz

# すべての形式でエクスポート
python 3d_reconstruction.py --input image.jpg --export-all
```

## ⚙️ オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 入力画像またはディレクトリ | 必須 |
| `--output`, `-o` | 出力ディレクトリ | `./output_3d` |
| `--model`, `-m` | モデル名またはパス | `depth-anything/DA3-LARGE` |
| `--format` | エクスポート形式（glb, npz, mini_npz）<br>`-`で複数指定可能 | `glb` |
| `--export-all` | すべての形式でエクスポート | False |
| `--point-size` | 3D可視化用のポイントサイズ | `0.01` |
| `--process-res` | 処理解像度 | `504` |
| `--device` | 使用デバイス (cuda/cpu) | 自動検出 |

## 📤 出力

- `{name}.glb` - 3Dメッシュ（GLB形式）
- `{name}.npz` - 完全なデータセット
- `{name}_mini.npz` - コンパクト版データセット

## 💡 使用例

### 例1: GLBとPLYの両方でエクスポート

```bash
python 3d_reconstruction.py \
    --input photo.jpg \
    --format glb-ply \
    --output ./3d_model
```

### 例2: 複数画像から統合モデル

```bash
python 3d_reconstruction.py \
    --input ./multi_view_images/ \
    --export-all
```

### 例3: 高品質3Dモデル

```bash
python 3d_reconstruction.py \
    --input image.jpg \
    --model depth-anything/DA3-GIANT \
    --process-res 672 \
    --format glb
```

## 🖼️ 3Dファイルの表示方法

### オンラインビューア
- https://3dviewer.net/ - ブラウザで直接表示
- https://gltf-viewer.donmccurdy.com/ - GLTFビューア

### デスクトップアプリケーション
- **Blender**: File > Import > glTF 2.0
- **MeshLab**: File > Import Mesh
- **CloudCompare**: 点群処理用
- **Windows 3D Viewer**: Windows標準アプリ

### プログラム的に読み込み

```python
import numpy as np

# NPZファイルを読み込み
data = np.load('scene.npz')
depth = data['depth']
images = data['processed_images']
extrinsics = data['extrinsics']  # カメラポーズ
intrinsics = data['intrinsics']  # カメラ内部パラメータ
```

## 🔧 トラブルシューティング

### GLBエクスポートに失敗する

NPZ形式で保存してから、他のツールで変換してください：
```bash
python 3d_reconstruction.py --input image.jpg --format npz
```

## 📚 関連サンプル

- [06. マルチビュー深度とカメラポーズ推定](../06_multiview_pose_estimation/) - 複数視点からの3D再構成
- [07. 大規模3D再構成](../07_advanced_3d_reconstruction/) - 動画からの大規模再構成
