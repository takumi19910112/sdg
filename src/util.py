#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from types import SimpleNamespace

import yaml
import json
from typing import List, Dict, Any

import re

import random
from wordfreq import top_n_list
from fugashi import Tagger

# 形態素解析器を初期化（デフォルトで UniDic-lite を使用）
tagger = Tagger()        

class ConfigLoadError(RuntimeError):
    """設定ファイル読込失敗時に送出する独自例外"""


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """
    再帰的に dict を SimpleNamespace 化するヘルパー関数。

    Parameters
    ----------
    d : dict
        YAML からロードした辞書

    Returns
    -------
    types.SimpleNamespace
        ネスト構造を保ったまま属性アクセス可能なオブジェクト
    """
    def _convert(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _convert(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [_convert(x) for x in obj]
        else:
            return obj

    return _convert(d)


def load_config(yaml_path: str | Path) -> SimpleNamespace:
    """
    YAML ファイルを読み込み、属性アクセスしやすい名前空間にして返す。

    Parameters
    ----------
    yaml_path : str | Path
        YAML ファイルへのパス

    Returns
    -------
    types.SimpleNamespace
        設定オブジェクト
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise ConfigLoadError(f"設定ファイルが見つかりません: {yaml_path}")

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            raw_conf: dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"YAML パースに失敗しました: {e}") from e

    return _dict_to_namespace(raw_conf)

def read_text_file(file_path: str):
    """
    指定のテキストファイルからテキストを読み込む（エラー処理なし）。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def load_jsonl(file_path: str, encoding: str = 'utf-8') -> list[dict]:
    """
    JSONLファイルを読み込み、辞書のリストを返します。

    Args:
        file_path (str): 読み込むJSONLファイルのパス。
        encoding (str): ファイルのエンコーディング（デフォルトは 'utf-8'）。

    Returns:
        list[dict]: 各行のJSONオブジェクトを辞書に変換したもののリスト。
    """
    data = []
    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            # 空行や空白のみの行はスキップ
            line = line.strip()
            if line:
                # JSON文字列を辞書に変換してリストに追加
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str, mode: str = 'w') -> None:
    """
    辞書のリストをJSONL (JSON Lines) 形式でファイルに保存します。

    Args:
        data (List[Dict[str, Any]]): 保存する辞書のリスト。
        file_path (str): 保存先のファイルパス。
        mode (str, optional): ファイルの書き込みモード。
                              'w' (上書き) または 'a' (追記)。
                              デフォルトは 'w' です。
    """
    # modeが 'w' または 'a' 以外の場合はエラーを発生させる
    if mode not in ['w', 'a']:
        raise ValueError("引数 `mode` は 'w' または 'a' を指定してください。")

    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            # 各辞書をJSON文字列に変換し、改行を加えて書き込む
            # ensure_ascii=False で日本語などが文字化けせずに出力される
            json_string = json.dumps(item, ensure_ascii=False)
            f.write(json_string + '\n')

def separate_think_and_answer(data):
    """
    AIの回答から<think>タグで囲まれた思考部分と、その後の回答部分を分離します。

    この関数は、単一の文字列、または文字列のリストを入力として受け取ります。

    - <think>...</think>で囲まれた部分を「思考」
    - </think>より後の部分を「回答」

    として、[思考, 回答] の形式で分離します。

    Args:
        data (str or list[str]): 処理対象の文字列、または文字列のリスト。

    Returns:
        list[str] or list[list[str]]:
        - 入力が文字列の場合: [思考部分, 回答部分] のリストを返します。
        - 入力がリストの場合: 各要素を処理した結果のリスト [[思考1, 回答1], [思考2, 回答2], ...] を返します。
        - <think>タグが見つからない場合は、思考部分を空文字列とし、入力全体を回答部分として返します。

    Raises:
        TypeError: 入力が文字列またはリストでない場合に送出されます。
    """

    # 単一の文字列を処理する内部関数
    def _split_single_string(text: str) -> list[str]:
        # re.DOTALLフラグにより、改行を含むテキストにも対応
        pattern = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)
        match = pattern.search(text)

        if match:
            # マッチした場合、group(1)が思考、group(2)が回答
            think_part = match.group(1).strip()
            answer_part = match.group(2).strip()
            return [think_part, answer_part]
        else:
            # マッチしない場合、思考は空、全体を回答とみなす
            return ["", text.strip()]

    # 入力の型に応じて処理を分岐
    if isinstance(data, str):
        return _split_single_string(data)
    elif isinstance(data, list):
        return [_split_single_string(item) for item in data]
    else:
        raise TypeError("入力は文字列(str)または文字列のリスト(list)である必要があります。")

def parse_cot_reasoning(text: str) -> dict:
    """
    Chain of Thought推論結果を段階的に解析してstructured形式で返します。

    Args:
        text (str): CoT推論の生テキスト

    Returns:
        dict: 構造化されたCoT推論結果
            - steps: 推論ステップのリスト
            - final_answer: 最終回答
            - reasoning_type: 推論タイプ
    """
    steps = []
    final_answer = ""
    reasoning_type = "step_by_step"
    
    # ステップ番号パターンを検索
    step_pattern = re.compile(r'(?:ステップ|Step)\s*(\d+)[:\s]*(.+?)(?=(?:ステップ|Step)\s*\d+|$)', re.DOTALL | re.IGNORECASE)
    step_matches = step_pattern.findall(text)
    
    if step_matches:
        for step_num, content in step_matches:
            steps.append({
                "step": int(step_num),
                "content": content.strip()
            })
    else:
        # 箇条書きパターンを検索
        bullet_pattern = re.compile(r'[・•\-\*]\s*(.+?)(?=[・•\-\*]|$)', re.DOTALL)
        bullet_matches = bullet_pattern.findall(text)
        
        for i, content in enumerate(bullet_matches, 1):
            steps.append({
                "step": i,
                "content": content.strip()
            })
    
    # 最終回答を抽出
    answer_patterns = [
        r'(?:最終的な答え|最終回答|答え|結論)[:\s]*(.+)$',
        r'(?:Therefore|Thus|Hence)[:\s]*(.+)$',
        r'(?:したがって|よって|ゆえに)[:\s]*(.+)$'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            final_answer = match.group(1).strip()
            break
    
    # 最終回答が見つからない場合は最後のステップの内容を使用
    if not final_answer and steps:
        final_answer = steps[-1]["content"]
    
    return {
        "steps": steps,
        "final_answer": final_answer,
        "reasoning_type": reasoning_type,
        "total_steps": len(steps)
    }

def random_japanese_nouns(n: int = 10, pool_size: int = 20000) -> list[str]:
    """
    日本語語彙プールから名詞を n 個ランダムに返す。

    Parameters
    ----------
    n : int
        返す名詞の個数
    pool_size : int
        `wordfreq.top_n_list` で先頭から取得する語彙数（大きいほど語彙が多様化）

    Returns
    -------
    list[str]
        ランダムな日本語名詞リスト
    """
    # 高頻度語を取得
    candidates = top_n_list("ja", pool_size)

    # fugashi で名詞のみ抽出
    nouns = []
    for w in candidates:
        token = tagger(w)[0]          # 1 語なので token は 1 件
        if token.pos.startswith("名詞"):
            nouns.append(w)

    if len(nouns) < n:
        raise ValueError(
            f"名詞候補が {len(nouns)} 語しかありません。pool_size を増やしてください。"
        )

    return random.sample(nouns, n)