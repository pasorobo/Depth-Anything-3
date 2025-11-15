# Depth Anything 3 - Examples

このディレクトリには、Depth Anything 3を使用した実用的なサンプルコードが含まれています。各サンプルは、WindowsとLinuxの両方のコマンドラインで動作するように設計されています。

**🆕 新構造**: 各サンプルは個別のフォルダに整理され、それぞれに詳細なREADME.mdとrequirements.txtが含まれています。

## 📋 目次

- [セットアップ](#セットアップ)
- [サンプル一覧](#サンプル一覧)
- [よくある質問](#よくある質問)

## 🚀 セットアップ

### 1. 基本インストール

```bash
# リポジトリのルートディレクトリから実行
pip install -e .

# または、すべての機能を含めてインストール
pip install -e ".[all]"
```

### 2. サンプル用の追加パッケージ

各サンプルフォルダ内の`requirements.txt`を参照してください。

```bash
# 例: 単一画像推論のパッケージをインストール
cd 01_single_image_inference
pip install -r requirements.txt
```

### 3. モデルのダウンロード

サンプルを初めて実行すると、モデルが自動的にHugging Face Hubからダウンロードされます。

## 📚 サンプル一覧

### 基本サンプル

| # | サンプル名 | 説明 | 詳細 |
|---|-----------|------|------|
| 01 | **単一画像推論** | 最もシンプルな使用例 | [📖 README](./01_single_image_inference/) |
| 02 | **バッチ画像処理** | 複数画像の効率的な一括処理 | [📖 README](./02_batch_image_processing/) |
| 03 | **動画処理** | 動画フレームからの深度推定 | [📖 README](./03_video_processing/) |

### 3D再構成サンプル

| # | サンプル名 | 説明 | 詳細 |
|---|-----------|------|------|
| 04 | **3D再構成** | 深度から3Dモデルを生成 | [📖 README](./04_3d_reconstruction/) |
| 06 | **マルチビュー深度** | 複数視点からの3D再構成 | [📖 README](./06_multiview_pose_estimation/) |
| 07 | **大規模3D再構成** ⭐ NEW | 動画から大規模点群を生成 | [📖 README](./07_advanced_3d_reconstruction/) |
| 08 | **ライブカメラ3D再構成** ⭐ NEW | リアルタイムカメラスキャン | [📖 README](./08_live_camera_reconstruction/) |

### 高度なサンプル

| # | サンプル名 | 説明 | 詳細 |
|---|-----------|------|------|
| 05 | **メトリック深度推定** | 実スケールの距離測定 | [📖 README](./05_metric_depth/) |

### ユーティリティツール

| ツール名 | 説明 | 詳細 |
|---------|------|------|
| **点群処理ツール** ⭐ NEW | 点群の統合・最適化 | [📖 README](./utils/) |

## 🎯 クイックスタート

### シンプルな深度推定

```bash
cd 01_single_image_inference
python single_image_inference.py --input image.jpg
```

### 動画から3D再構成

```bash
cd 07_advanced_3d_reconstruction
python advanced_3d_reconstruction.py --input video.mp4
```

### ライブカメラでスキャン

```bash
cd 08_live_camera_reconstruction
python live_camera_reconstruction.py
# SPACE でキャプチャ, 's' で保存, 'q' で終了
```

## 📖 各サンプルの詳細

各サンプルフォルダには以下が含まれています：

- **スクリプト**: 実行可能なPythonスクリプト
- **README.md**: 詳細な使用方法とオプション説明
- **requirements.txt**: 追加の依存パッケージ

### ディレクトリ構造

```
examples/
├── README.md (このファイル)
├── 01_single_image_inference/
│   ├── single_image_inference.py
│   ├── README.md
│   └── requirements.txt
├── 02_batch_image_processing/
│   ├── batch_image_processing.py
│   ├── README.md
│   └── requirements.txt
├── 03_video_processing/
│   ├── video_processing.py
│   ├── README.md
│   └── requirements.txt
├── 04_3d_reconstruction/
│   ├── 3d_reconstruction.py
│   ├── README.md
│   └── requirements.txt
├── 05_metric_depth/
│   ├── metric_depth.py
│   ├── README.md
│   └── requirements.txt
├── 06_multiview_pose_estimation/
│   ├── multiview_pose_estimation.py
│   ├── README.md
│   └── requirements.txt
├── 07_advanced_3d_reconstruction/
│   ├── advanced_3d_reconstruction.py
│   ├── README.md
│   └── requirements.txt
├── 08_live_camera_reconstruction/
│   ├── live_camera_reconstruction.py
│   ├── README.md
│   └── requirements.txt
└── utils/
    ├── pointcloud_tools.py
    ├── README.md
    └── requirements.txt
```

## 🔧 よくある質問

### Q: どのサンプルから始めればいいですか？

A: 用途に応じて選択してください：

- **初めての方**: [01. 単一画像推論](./01_single_image_inference/)
- **大量の画像を処理したい**: [02. バッチ画像処理](./02_batch_image_processing/)
- **3Dモデルを作りたい**: [04. 3D再構成](./04_3d_reconstruction/)
- **動画から3Dを作りたい**: [07. 大規模3D再構成](./07_advanced_3d_reconstruction/)
- **リアルタイムスキャン**: [08. ライブカメラ3D再構成](./08_live_camera_reconstruction/)

### Q: GPU メモリ不足エラーが発生します

A: 以下の方法を試してください：

1. バッチサイズを減らす: `--batch-size 1`
2. 処理解像度を下げる: `--process-res 336`
3. より小さいモデルを使用: `--model depth-anything/DA3-BASE`

### Q: どのモデルを使用すべきですか？

A: 用途に応じて選択してください：

- **一般的な深度推定**: `depth-anything/DA3-LARGE`（推奨）
- **高速処理**: `depth-anything/DA3-BASE` または `DA3-SMALL`
- **メトリック深度**: `depth-anything/DA3METRIC-LARGE`
- **単眼深度のみ**: `depth-anything/DA3MONO-LARGE`
- **すべての機能**: `depth-anything/DA3NESTED-GIANT-LARGE`

### Q: CPUでも動作しますか？

A: はい、`--device cpu` を指定することで動作しますが、GPUよりも大幅に遅くなります。

### Q: Windows で実行する際の注意点は？

A: すべてのサンプルはWindowsとLinuxの両方で動作するように設計されています。PowerShellまたはコマンドプロンプトで実行してください。

```powershell
# Windows (PowerShell) での実行例
cd 01_single_image_inference
python single_image_inference.py --input C:\Users\YourName\Pictures\image.jpg
```

### Q: 大規模3D再構成で点群が巨大になりすぎます

A: 以下の方法で点群サイズを削減できます：

1. ダウンサンプリング: `--downsample-points 0.5`（50%に削減）
2. 信頼度フィルタリング: `--min-confidence 0.7`（信頼度の高い点のみ）
3. 処理後に[ユーティリティツール](./utils/)で削減

### Q: ライブカメラ再構成でカメラが認識されません

A: 以下を確認してください：

1. カメラが正しく接続されているか確認
2. 他のアプリケーションがカメラを使用していないか確認
3. 異なるカメラIDを試す: `--camera 0`, `--camera 1`, `--camera 2`
4. Linuxの場合: `/dev/video*` デバイスのパーミッションを確認

### Q: 点群の形式を変換したい

A: [ユーティリティツール](./utils/)を使用してください：

```bash
cd utils
# PLY → NPZ
python pointcloud_tools.py convert --input cloud.ply --output cloud.npz

# NPZ → PLY
python pointcloud_tools.py convert --input cloud.npz --output cloud.ply
```

### Q: 複数の点群を統合したい

A: [ユーティリティツール](./utils/)のmergeコマンドを使用：

```bash
cd utils
python pointcloud_tools.py merge \
    --input session1.ply session2.ply session3.ply \
    --output merged.ply
```

## 🎓 学習パス

### 初級者向け
1. [01. 単一画像推論](./01_single_image_inference/) - 基本を理解
2. [02. バッチ画像処理](./02_batch_image_processing/) - 効率的な処理
3. [04. 3D再構成](./04_3d_reconstruction/) - 3Dモデル作成

### 中級者向け
4. [05. メトリック深度推定](./05_metric_depth/) - 実スケール測定
5. [06. マルチビュー深度](./06_multiview_pose_estimation/) - 複数視点処理
6. [03. 動画処理](./03_video_processing/) - 動画からの深度推定

### 上級者向け
7. [07. 大規模3D再構成](./07_advanced_3d_reconstruction/) - 大規模シーン
8. [08. ライブカメラ3D再構成](./08_live_camera_reconstruction/) - リアルタイム処理
9. [ユーティリティツール](./utils/) - 点群の最適化

## 🛠️ 開発者向け情報

### 新しいサンプルの追加

1. 新しいフォルダを作成: `09_new_example/`
2. スクリプトを追加: `new_example.py`
3. README.mdを作成（他のサンプルを参考に）
4. requirements.txtを作成
5. このREADME.mdに追加

### コントリビューション

改善提案やバグ報告は、GitHubのIssuesで受け付けています。

## 📝 ライセンス

これらのサンプルコードは、Depth Anything 3プロジェクトのライセンスに従います。詳細はプロジェクトのルートディレクトリにあるLICENSEファイルを参照してください。

## 📚 参考資料

- [メインドキュメント](../README.md)
- [API リファレンス](../docs/API.md)
- [CLI リファレンス](../docs/CLI.md)
- [Hugging Face Hub](https://huggingface.co/depth-anything)

## 🌟 サンプルギャラリー

各サンプルの詳細な使用方法は、それぞれのフォルダ内のREADME.mdをご覧ください。すべてのサンプルには、実行例、オプション説明、トラブルシューティングガイドが含まれています。

---

**Happy Depth Estimation! 🎉**
