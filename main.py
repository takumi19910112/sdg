#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.funs import setup_pipeline

def main():
    # パイプライン初期化
    pipe = setup_pipeline("settings.yaml")

    try:
        # 質問生成
        questions = pipe.generate_questions()
        # ベースモデル生成時のみキュレーション
        questions = pipe.curate_questions(questions)
        # 多様性フィルタ
        questions = pipe.diversity_filter(questions)
        # 質問進化
        questions = pipe.evolve_questions(questions)
        # 回答生成
        answers = pipe.generate_answers(questions)
        # 回答進化
        answers = pipe.evolve_answers(answers)
        # 最終キュレーション
        final_data = pipe.curate_final(answers)
        # 保存
        output_path = pipe.save_dataset(final_data)
        print(f"データセット生成完了: {output_path.resolve()}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
