"""
quick_test_vllm.py
修正した src/vllm_inf.py が import パス上にある前提
"""

from types import SimpleNamespace
from src.vllm_inf import inst_model_load, inst_model_inference, unload_model
from src.util import load_config

settings = load_config('settings.yaml')

# --- モデルロード & 推論 ----------------------------------------------
llm = inst_model_load(settings)
prompts = [
    "量子力学における「重ね合わせ」と「もつれ」の概念を説明してください。",
    "「夏目漱石」の小説の中で、最も有名な作品は何ですか？"
]

outputs = inst_model_inference(llm, prompts, settings)
for i, out in enumerate(outputs, 1):
    print(f"[{i}] {out.strip()}\n")

unload_model(settings.Instruct_model_name)
