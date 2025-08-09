# SDG (Scalable Data Generator) - Claude Memory

## プロジェクト概要

このプロジェクトは、vLLMベースの日本語LLM推論環境と、プロンプト進化/データ生成パイプラインを提供するCLIスクリプト群です。

## 主要な構成

### コアファイル
- `main.py` - メインの実行スクリプト
- `test_run.py` - 動作確認用の簡易テスト
- `settings.yaml` - 全ての設定を集中管理する設定ファイル

### 設定ファイル群
- `settings.yaml` - メイン設定（モデルパス、パラメータなど）
- `settings_api.yaml` - API用設定
- `settings_ollama.yaml` - Ollama用設定
- `settings_copy.yaml` - 設定のバックアップ

### ソースコード (`src/`)
- `vllm_inf.py` - vLLM推論エンジン
- `api_openai_inf.py` - OpenAI API互換推論
- `api_openai_comp_inf.py` - OpenAI互換API推論
- `api_google_inf.py` - Google API推論
- `ollama_inf.py` - Ollama推論
- `e5.py` - E5埋め込みモデル処理
- `funs.py` - 共通関数
- `util.py` - ユーティリティ関数

### プロンプトテンプレート (`prompts/`)
- `bare.txt` - 基本プロンプト
- `curation.txt` - キュレーション用プロンプト
- `curation_prompt_4_base.txt` - ベースモデル用キュレーション
- `evo_answer_prompt.txt` - 回答進化プロンプト
- `evo_question_prompt.txt` - 質問進化プロンプト
- `inst_seed.txt` - 指示シードプロンプト
- `test.txt` - テスト用プロンプト

### RunPod専用ファイル
- `runpod_setup.py` - RunPod用自動セットアップスクリプト
- `runpod_quick_setup.sh` - RunPod用クイックセットアップシェルスクリプト
- `README-RunPod.md` - RunPod専用の詳細ガイド

## 重要な設定項目

### モデル設定 (settings.yaml)
```yaml
# 指示追従モデル
Instruct_model_name: "./data/model/TinySwallow-1.5B-Instruct"
Instruct_gpu_memory_utilization: 0.9

# ベースモデル
base_model_name: "./data/model/sarashina2.2-3b"  
base_gpu_memory_utilization: 0.7

# 思考モデル
think_model_name: "./data/model/DeepSeek-R1-Distill-Qwen-1.5B"
think_gpu_memory_utilization: 0.9

# 埋め込みモデル
E5_path: "./data/model/multilingual-e5-large"
```

### パフォーマンス設定
```yaml
tensor_parallel_size: 1  # GPU並列数
batch_size: 32          # バッチサイズ
max_tokens: 4096        # 最大トークン数
trust_remote_code: True # リモートコード実行許可
seed: 42               # 乱数シード
```

### データ生成パイプライン設定
```yaml
# 質問生成
Seed_generation_method: "inst"
Prompt_evolution: False
Number_of_questions_generated: 500

# 回答生成
Answer_evolution: False
Using_think_models_for_answer: False

# キュレーション
Data_curation: False
```

## RunPod対応

### GPU構成別推奨モデル

**RTX 4090 (24GB) - $0.69/hr**
- TinySwallow-1.5B-Instruct (軽量日本語)
- Llama-3.1-8B-Instruct (高性能8B)
- Qwen2.5-14B-Instruct (14B高性能)

**RTX A6000 (48GB) - $0.76/hr**
- Qwen2.5-32B-Instruct (32B超高性能)
- DeepSeek-Coder-V2-Lite-Instruct (コーディング特化)

**A100/H100 (80GB) - $1.69-3.29/hr**
- Llama-3.1-70B-Instruct (70B最高性能)
- Qwen2.5-72B-Instruct (72B最高性能)
- DeepSeek-V2.5 (MoEアーキテクチャ)

### 量子化オプション
- 70Bモデルも INT4量子化により 24GB GPU で実行可能
- FP16 → INT8 で約50%メモリ削減
- FP16 → INT4 で約75%メモリ削減

## 使用方法

### 基本的な使用フロー
1. `runpod_quick_setup.sh` で環境セットアップ
2. `python3 runpod_setup.py` でモデル選択・ダウンロード
3. `python3 test_run.py` で動作確認
4. `python3 main.py` でメイン処理実行

### テストコマンド
```bash
# 基本動作確認
python3 test_run.py

# メイン処理実行
python3 main.py
```

### lintとタイプチェック
プロジェクトには特定のlint/typecheckコマンドは設定されていません。Pythonの基本的なsyntaxチェックのみ。

## データディレクトリ構造
```
data/
├── model/           # ダウンロードしたモデル
├── output/          # 生成されたデータ出力
└── [その他の一時ファイル]
```

## 依存関係 (requirements.txt)
- vllm 0.9.1 (メイン推論エンジン)
- その他の必要ライブラリ（詳細はrequirements.txt参照）

## 注意事項
- モデルダウンロードには大容量の帯域とストレージが必要
- GPU要件は選択するモデルサイズに依存
- Hugging Face認証が必要な場合あり（`huggingface-cli login`）
- RunPodでは従量課金のためGPU使用時間に注意

## カスタムスラッシュコマンド

### 新機能 - RunPod最適化コマンド

プロジェクトに以下のカスタムスラッシュコマンドが追加されました：

**基本コマンド（runpod_optimizer.py）:**
- `/runpod-optimize <RunPod情報>` - リソース情報からモデル推奨
- `/runpod-generate <モデル名> <RunPod構成>` - vLLM実行スクリプト生成

**高度コマンド（runpod_optimizer_web.py）:**
- `/runpod-search <GPU名>` - 最新価格・性能情報検索
- `/runpod-optimize <要件>` - 予算・用途に応じた最適構成推奨
- `/runpod-generate <モデル名> <構成>` - 高度最適化スクリプト生成

### 使用例

```bash
# RunPod情報に基づいてモデル推奨
python runpod_optimizer.py /runpod-optimize "RTX 4090 24GB VRAM, $0.69/hr"

# 特定モデル用のvLLMスクリプト生成
python runpod_optimizer.py /runpod-generate "Llama-3.1-70B-Instruct" "A100 80GB"

# 要件ベースの構成最適化
python runpod_optimizer_web.py /runpod-optimize "budget: $1.5/hr, model: 32B, use: coding"

# 高度最適化スクリプト生成
python runpod_optimizer_web.py /runpod-generate "Qwen2.5-72B" "H100 speed-optimized"
```

### 生成されるスクリプトの特徴

1. **基本スクリプト** - シンプルなvLLM実行
2. **高度スクリプト** - 以下の機能を含む：
   - 動的量子化判定
   - 性能ベンチマーク
   - 複数実行モード（benchmark/interactive/batch）
   - プリセット別サンプリング設定
   - 詳細ログとエラーハンドリング

### テスト実行

```bash
# カスタムコマンドの動作テスト
python test_custom_commands.py
```

詳細な使用方法は `README-CustomCommands.md` を参照してください。

## トラブルシューティング
1. メモリ不足 → `gpu_memory_utilization`を下げる
2. CUDA関連エラー → GPUドライバー・CUDAバージョン確認
3. モデルダウンロード失敗 → HF認証・ディスク容量確認
4. 推論エラー → モデルパス・設定ファイル確認
5. カスタムコマンドエラー → `python test_custom_commands.py`でデバッグ