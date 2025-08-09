#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from src.funs import setup_pipeline
from src import util

def test_cot_basic():
    """
    基本的なCoT機能のテスト
    """
    print("=== 基本CoT機能テスト ===")
    
    # テスト用設定を作成
    settings = util.load_config("settings.yaml")
    settings.Enable_CoT = True
    settings.CoT_mode = "ultimate"
    settings.Number_of_questions_generated = 3
    
    # テスト用データを準備
    test_data = [
        {"Question": "日本の首都は何ですか？その理由も説明してください。"},
        {"Question": "機械学習と深層学習の違いを説明してください。"},
        {"Question": "地球温暖化の原因と対策について教えてください。"}
    ]
    
    # パイプラインを初期化
    pipe = setup_pipeline("settings.yaml")
    pipe.settings = settings
    pipe.initialize()
    
    try:
        # CoT回答生成をテスト
        print("CoT回答生成をテスト中...")
        results = pipe._generate_cot_answers(test_data)
        
        print(f"テスト完了: {len(results)}件の回答を生成しました")
        
        # 結果を表示
        for i, result in enumerate(results):
            print(f"\n--- 質問 {i+1} ---")
            print(f"質問: {result['Question']}")
            print(f"回答: {result['Answer'][:200]}...")
            if 'reasoning_steps' in result:
                print(f"推論ステップ数: {len(result['reasoning_steps'])}")
        
        return True
        
    except Exception as e:
        print(f"CoT基本テストでエラーが発生: {e}")
        return False

def test_cot_ensemble():
    """
    アンサンブルCoT機能のテスト
    """
    print("\n=== アンサンブルCoT機能テスト ===")
    
    # テスト用設定を作成
    settings = util.load_config("settings.yaml")
    settings.Enable_CoT = True
    settings.Enable_Ensemble = True
    settings.CoT_mode = "ultimate"
    settings.Ensemble_models = ["Instruct", "base"]  # テストのため2モデルに限定
    settings.Number_of_questions_generated = 2
    
    # テスト用データを準備
    test_data = [
        {"Question": "AI技術の将来性について教えてください。"},
        {"Question": "環境保護と経済発展を両立する方法は何ですか？"}
    ]
    
    # パイプラインを初期化
    pipe = setup_pipeline("settings.yaml")
    pipe.settings = settings
    pipe.initialize()
    
    try:
        # アンサンブルCoT回答生成をテスト
        print("アンサンブルCoT回答生成をテスト中...")
        results = pipe._generate_ensemble_cot_answers(test_data)
        
        print(f"アンサンブルテスト完了: {len(results)}件の回答を生成しました")
        
        # 結果を表示
        for i, result in enumerate(results):
            print(f"\n--- アンサンブル質問 {i+1} ---")
            print(f"質問: {result['Question']}")
            print(f"回答: {result['Answer'][:200]}...")
            if 'ensemble_reasoning' in result:
                print(f"使用モデル: {result['ensemble_reasoning']['models_used']}")
                print(f"推論ステップ: {list(result['ensemble_reasoning']['steps'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"アンサンブルCoTテストでエラーが発生: {e}")
        return False

def test_cot_parsing():
    """
    CoT推論解析機能のテスト
    """
    print("\n=== CoT推論解析機能テスト ===")
    
    # テスト用CoT推論テキスト
    test_text = """
ステップ1: この問題は機械学習の基本概念に関するものです。

ステップ2: 機械学習とは、コンピュータがデータから学習してパターンを見つける技術です。

ステップ3: 深層学習は機械学習の一部で、ニューラルネットワークを使用します。

最終回答: 機械学習は広義の概念で、深層学習はその一部である特殊な手法です。
"""
    
    try:
        # CoT解析をテスト
        parsed_result = util.parse_cot_reasoning(test_text)
        
        print("CoT解析テスト完了:")
        print(f"推論ステップ数: {parsed_result['total_steps']}")
        print(f"最終回答: {parsed_result['final_answer']}")
        print(f"推論タイプ: {parsed_result['reasoning_type']}")
        
        for step in parsed_result['steps']:
            print(f"  ステップ{step['step']}: {step['content'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"CoT解析テストでエラーが発生: {e}")
        return False

def main():
    """
    メインテスト実行
    """
    print("Chain of Thought (CoT) 機能テストを開始します...\n")
    
    results = []
    
    # 各テストを実行
    results.append(("CoT解析テスト", test_cot_parsing()))
    
    # 注意: 以下のテストは実際のモデルロードが必要なため、環境によってはスキップ
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        results.append(("CoT基本テスト", test_cot_basic()))
        results.append(("アンサンブルCoTテスト", test_cot_ensemble()))
    else:
        print("注意: 完全なテストを実行するには --full オプションを使用してください")
        print("(実際のモデルロードが必要なテストはスキップされます)\n")
    
    # テスト結果をまとめる
    print("\n" + "="*50)
    print("テスト結果サマリー:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n合計: {passed}/{len(results)} テストが成功しました")
    
    if passed == len(results):
        print("すべてのテストが成功しました！ 🎉")
        return 0
    else:
        print("一部のテストが失敗しました。")
        return 1

if __name__ == "__main__":
    sys.exit(main())