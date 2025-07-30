import sys
from typing import List, Any
import openai
from openai import OpenAI
from openai import APIError



# --- ヘルパー関数 -----------------------------------------------------------------

def _get_model_name(settings: Any, prefix: str) -> str:
    """
    settingsオブジェクトとプレフィックスに基づき、モデル名を取得する。
    """
    return getattr(settings, f"openai_comp_{prefix}_model_name")

def _get_sampling_params(settings: Any, prefix: str) -> dict:
    """
    settingsオブジェクトとプレフィックスに基づき、OpenAI互換API用のサンプリングパラメータを生成する。
    """
    temperature = getattr(settings, f"openai_comp_{prefix}_temperature", 0.7)
    top_p = getattr(settings, f"openai_comp_{prefix}_top_p", 0.9)
    max_tokens = getattr(settings, "max_tokens", 4096)
    
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

def _execute_inference(model_name: str, prompts: List[str], options: dict, is_chat: bool, settings: Any) -> List[str]:
    """
    OpenAI互換APIを使用して推論を実行する内部関数。
    """
    print(f"OpenAI互換モデル '{model_name}' を使用して{len(prompts)}件の推論を実行中...")
    results = []
    try:
        client = OpenAI(
            base_url=getattr(settings, "openai_comp_endpoint", "https://api.openai.com/v1"),
            api_key=getattr(settings, "openai_comp_api_key"),
        )
        for prompt in prompts:
            if is_chat:
                # For chat models
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=options.get("temperature"),
                    top_p=options.get("top_p"),
                    max_tokens=options.get("max_tokens"),
                    stop=options.get("stop"),
                )
                results.append(response.choices[0].message.content)
            else:
                # For text generation models (legacy, but kept for compatibility)
                response = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    temperature=options.get("temperature"),
                    top_p=options.get("top_p"),
                    max_tokens=options.get("max_tokens"),
                    stop=options.get("stop"),
                )
                results.append(response.choices[0].text)
        print("推論が完了しました。")
        return results
    except APIError as e:
        print(f"OpenAI APIでの推論中にエラーが発生しました: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        sys.exit(1)

# --- 1. モデルロード/アンロード系関数（ダミー） --------------------------------------------------
# OpenAI互換APIではクライアント側での明示的なロード/アンロードは不要なため、ダミー関数を定義します。

def inst_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "Instruct")
    print(f"OpenAI互換モデル '{model_name}' を使用します。")
    return model_name

def base_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "base")
    print(f"OpenAI互換モデル '{model_name}' を使用します。")
    return model_name

def think_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "think")
    print(f"OpenAI互換モデル '{model_name}' を使用します。")
    return model_name

def unload_model(model_path_for_log: str):
    print(f"OpenAI互換モデル '{model_path_for_log}' のセッションを終了しました。")
    pass

# --- 2. 推論系関数（5種類） ------------------------------------------------------

def inst_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    Instructモデル（チャット形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "Instruct")
    return _execute_inference(model_name, prompts, options, is_chat=True, settings=settings)

def base_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    ベースモデル（生成形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["stop"] = ['<stop>']
    return _execute_inference(model_name, prompts, options, is_chat=False, settings=settings)

def think_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    長考モデル（チャット形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "think")
    return _execute_inference(model_name, prompts, options, is_chat=True, settings=settings)

def curation_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    キュレーション用の推論を行う。ベースモデルを使用するが、サンプリングパラメータが異なる。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["temperature"] = 0.01
    options["top_p"] = 1.0
    options["stop"] = ['<stop>']
    return _execute_inference(model_name, prompts, options, is_chat=False, settings=settings)

def evolution_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    思考を進化させるための推論を行う。ベースモデルを使用するが、サンプリングパラメータが異なる。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["temperature"] = 0.6
    options["top_p"] = 0.9
    options["stop"] = ['<stop>']
    return _execute_inference(model_name, prompts, options, is_chat=False, settings=settings)
