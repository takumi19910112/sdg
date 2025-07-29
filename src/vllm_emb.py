import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Any
import sys

def get_embeddings(sentences: List[str], settings: Any) -> np.ndarray:
    model_path = settings.E5_path
    print(f"モデル '{model_path}' を読み込んでいます...")
    try:
        model = SentenceTransformer(model_path)
        return model.encode(['passage: ' + s for s in sentences], show_progress_bar=False)
    except Exception as e:
        print(f"モデル '{model_path}' の読み込みに失敗しました。パスが正しいか確認してください。エラー: {e}")
        sys.exit(1)