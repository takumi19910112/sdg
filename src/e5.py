import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Union, Optional

# --- 設定 ---
# 注意: このモデルパスは、ローカル環境にモデルが存在する場合のパスです。
# 存在しない場合は、Hugging Face Hubのモデル名（例: 'intfloat/multilingual-e5-large'）
# を指定すると、自動的にダウンロードされます。
DEFAULT_MODEL_PATH = 'intfloat/multilingual-e5-large'


def reduce_data_by_diversity(
    data: Union[List[str], List[Dict]],
    data_format: str,
    cut_rate: float,
    data_key: Optional[str] = None,
    model_path: str = DEFAULT_MODEL_PATH
) -> Union[List[str], List[Dict]]:
    """
    データの多様性に基づいて、類似度の高いデータから指定された割合を削減する関数。

    Args:
        data (Union[List[str], List[Dict]]): 削減対象のデータ。
        data_format (str): データの形式。"list"または"dict"を指定。
        cut_rate (float): 削減するデータの割合（0〜100）。
        data_key (Optional[str], optional): data_formatが"dict"の場合、
                                            処理対象のテキストが含まれるキー。
                                            Defaults to None.
        model_path (str, optional): 使用するSentenceTransformerモデルのパスまたは
                                    Hugging Face Hubのモデル名。
                                    Defaults to DEFAULT_MODEL_PATH.

    Returns:
        Union[List[str], List[Dict]]: 削減後のデータ。
    """
    # --- 1. 入力値の検証 ---
    if not (0 <= cut_rate <= 100):
        raise ValueError("カット率（cut_rate）は0から100の間で指定してください。")
    if data_format not in ['list', 'dict']:
        raise ValueError("データ形式（data_format）は 'list' または 'dict' を指定してください。")
    if data_format == 'dict' and not data_key:
        raise ValueError("データ形式が 'dict' の場合、データキー（data_key）の指定は必須です。")
    if not data or len(data) < 2:
        return data

    # --- 2. 処理対象の文章リストを抽出 ---
    if data_format == 'list':
        sentences = data
    else: # 'dict'
        try:
            sentences = [item[data_key] for item in data]
        except KeyError:
            raise KeyError(f"指定されたキー '{data_key}' が一部の辞書に存在しません。")
        except TypeError:
            raise TypeError("データ形式が 'dict' の場合、入力データは辞書のリストである必要があります。")

    # --- 3. 削減するデータ数を計算 ---
    num_to_cut = int(len(sentences) * (cut_rate / 100))
    if num_to_cut == 0:
        return data
    if num_to_cut >= len(sentences):
        return []

    # --- 4. モデルの読み込みと文章のエンコード ---
    print(f"モデル '{model_path}' を読み込んでいます...")
    try:
        model = SentenceTransformer(model_path)
    except Exception as e:
        raise RuntimeError(f"モデル '{model_path}' の読み込みに失敗しました。パスが正しいか確認してください。エラー: {e}")

    print("文章のベクトル化（エンコード）を実行中...")
    # 参考コードと同様に、検索タスク用のプレフィックスを付与
    embeddings = model.encode(['passage: ' + s for s in sentences], show_progress_bar=False)

    # --- 5. 類似度に基づいて削減対象を選定 ---
    print("類似度を計算し、削減対象を選定中...")
    # コサイン類似度を計算
    similarity_matrix = cosine_similarity(embeddings)

    # 各データについて、自身を除く他のデータとの最大類似度を求める
    # 対角成分（自身との類似度=1.0）を計算から除外するために-1で埋める
    np.fill_diagonal(similarity_matrix, -1)
    # 各行の最大値が「最も似ているデータとの類似度スコア」になる
    redundancy_scores = np.max(similarity_matrix, axis=1)

    # 冗長性スコアが高い順にインデックスをソートし、削減対象を取得
    # np.argsortは昇順で返すため、スコアを負にして降順ソートと同じ効果を得る
    indices_to_cut = np.argsort(-redundancy_scores)[:num_to_cut]
    indices_to_cut_set = set(indices_to_cut)

    # --- 6. 削減後のデータを生成 ---
    reduced_data = [item for i, item in enumerate(data) if i not in indices_to_cut_set]
    
    print(f"処理が完了しました。{len(data)}件のデータから{len(indices_to_cut)}件を削減し、{len(reduced_data)}件になりました。")
    return reduced_data
