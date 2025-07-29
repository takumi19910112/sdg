import sys
from typing import List, Any
from openai import OpenAI
from openai import OpenAIError

client = OpenAI()

# --- ヘルパー関数 -----------------------------------------------------------------

def _get_model_name(settings: Any, prefix: str) -> str:
    """
    settingsオブジェクトとプレフィックスに基づき、モデル名を取得する。
    """
    return getattr(settings, f"openai_{prefix}_model_name")

def _get_sampling_params(settings: Any, prefix: str) -> dict:
    """
    settingsオブジェクトとプレフィックスに基づき、Ollama用のサンプリングパラメータを生成する。
    """
    temperature = getattr(settings, f"openai_{prefix}_temperature", 0.7)
    top_p = getattr(settings, f"openai_{prefix}_top_p", 0.9)
    max_tokens = getattr(settings, "max_tokens", 4096)
    seed = getattr(settings, "seed", 42)
    
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
    }

def _execute_inference(model_name: str, prompts: List[str], options: dict, is_chat: bool) -> List[str]:
    """
    Ollamaを使用して推論を実行する内部関数。
    """
    print(f"OpenAIモデル '{model_name}' を使用して{len(prompts)}件の推論を実行中...")
    results = []
    try:
        for prompt in prompts:
            if is_chat:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=options.get("temperature"),
                    top_p=options.get("top_p"),
                    max_tokens=options.get("max_tokens"),
                    seed=options.get("seed"),
                    stop=options.get("stop")
                )
                results.append(response.choices[0].message.content)
            else:
                response = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    temperature=options.get("temperature"),
                    top_p=options.get("top_p"),
                    max_tokens=options.get("max_tokens"),
                    seed=options.get("seed"),
                    stop=options.get("stop")
                )
                results.append(response.choices[0].text)
        print("推論が完了しました。")
        return results
    except OpenAIError as e:
        print(f"OpenAIでの推論中にエラーが発生しました: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        sys.exit(1)

# --- 1. モデルロード/アンロード系関数（ダミー） --------------------------------------------------
# OpenAIではクライアント側での明示的なロード/アンロードは不要なため、ダミー関数を定義します。

def inst_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "Instruct")
    print(f"OpenAIモデル '{model_name}' を使用します。")
    return model_name

def base_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "base")
    print(f"OpenAIモデル '{model_name}' を使用します。")
    return model_name

def think_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "think")
    print(f"OpenAIモデル '{model_name}' を使用します。")
    return model_name

def unload_model(model_path_for_log: str):
    print(f"OpenAIモデル '{model_path_for_log}' のセッションを終了しました。")
    pass

# --- 2. 推論系関数（5種類） ------------------------------------------------------

def inst_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    Instructモデル（チャット形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "Instruct")
    return _execute_inference(model_name, prompts, options, is_chat=True)

def base_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    ベースモデル（生成形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["stop"] = ['<stop>']
    return _execute_inference(model_name, prompts, options, is_chat=False)

def think_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    長考モデル（チャット形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "think")
    return _execute_inference(model_name, prompts, options, is_chat=True)

def curation_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    キュレーション用の推論を行う。ベースモデルを使用するが、サンプリングパラメータが異なる。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["temperature"] = 0.01
    options["top_p"] = 1.0
    options["stop"] = ['<stop>']
    return _execute_inference(model_name, prompts, options, is_chat=False)

def evolution_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    思考を進化させるための推論を行う。ベースモデルを使用するが、サンプリングパラメータが異なる。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["temperature"] = 0.6
    options["top_p"] = 0.9
    options["stop"] = ['<stop>']
    return _execute_inference(model_name, prompts, options, is_chat=False)
