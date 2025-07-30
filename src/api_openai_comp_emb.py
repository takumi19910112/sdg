import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from typing import List, Any
import sys

def get_embeddings(sentences: List[str], settings: Any) -> np.ndarray:
    model_name = getattr(settings, 'google_E5_model_name', 'models/embedding-001')
    print(f"Google Gemini埋め込みモデル '{model_name}' を使用してベクトルを生成します。")
    try:
        # Google Geminiの埋め込みモデルは、テキストのリストを受け取って埋め込みを生成する
        # genai.embed_content を使用
        response = genai.embed_content(
            model=model_name,
            content=sentences,
            task_type="RETRIEVAL_DOCUMENT"
        )
        # 埋め込みは response.embedding にリストとして格納されている
        if response and response['embedding']:
            return np.array(response['embedding'])
        else:
            print(f"Google Geminiからの予期せぬレスポンス形式です: {response}")
            sys.exit(1)
    except GoogleAPIError as e:
        print(f"Google APIでの埋め込み生成中にエラーが発生しました: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        sys.exit(1)