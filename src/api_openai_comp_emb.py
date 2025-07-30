import sys
from typing import List, Any
import openai
from openai import OpenAI
from openai import APIError
import numpy as np

def get_embeddings(sentences: List[str], settings: Any) -> np.ndarray:
    model_name = getattr(settings, 'openai_comp_E5_model_name', 'text-embedding-ada-002')
    print(f"OpenAI互換埋め込みモデル '{model_name}' を使用してベクトルを生成します。")
    try:
        client = OpenAI(
            base_url=getattr(settings, "openai_comp_endpoint", "https://localhost/v1"),
            api_key=getattr(settings, "openai_comp_api_key"),
        )
        response = client.embeddings.create(
            model=model_name,
            input=sentences
        )
        if response and response.data:
            embeddings = [d.embedding for d in response.data]
            for i, emb in enumerate(embeddings):
                print(f"Embedding {i} length: {len(emb)}")
            return np.array(embeddings)
        else:
            print(f"OpenAI互換APIからの予期せぬレスポンス形式です: {response}")
            sys.exit(1)
    except APIError as e:
        print(f"OpenAI互換APIでの埋め込み生成中にエラーが発生しました: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        sys.exit(1)