import numpy as np
import ollama
import sys
from ollama import ResponseError
from typing import List, Any

def get_embeddings(sentences: List[str], settings: Any) -> np.ndarray:
    model_name = getattr(settings, 'E5_model_name', 'mxbai-embed-large')
    print(f"Ollama埋め込みモデル '{model_name}' を使用してベクトルを生成します。")
    try:
        response = ollama.embed(model=model_name, input=sentences)
        
        embedding_list = getattr(response, 'embeddings', None)

        if embedding_list is None and isinstance(response, dict) and 'embeddings' in response:
            embedding_list = response['embeddings']

        if embedding_list:
            return np.array(embedding_list)
        else:
            print(f"Ollamaからの予期せぬレスポンス形式です: {response}")
            sys.exit(1)
    except ResponseError as e:
        print(f"Ollamaでの埋め込み生成中にエラーが発生しました: {e.error}")
        if e.status_code == 404:
            print(f"エラー: 埋め込みモデル '{model_name}' が見つかりません。")
            print(f"解決策: 'ollama pull {model_name}' を実行してモデルをダウンロードしてください。")
            sys.exit(1)
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        sys.exit(1)