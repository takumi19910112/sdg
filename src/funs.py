from pathlib import Path
from typing import Any, List, Dict, Optional, Union
from datetime import datetime
from src import util

# --- 型定義（本来は types.py へ分離推奨） ---
class Pipeline:
    """
    データ生成・キュレーション・進化・保存までを一貫して管理する高レベルAPIクラス。
    各メソッドはパイプラインの主要フェーズを担当し、途中結果を返す。
    """

    def __init__(self, config_path: Union[str, Path] = "settings.yaml"):
        """
        設定ファイルの読み込み・作業ディレクトリの準備など初期化処理を行う。
        """
        self.config_path = Path(config_path)
        self.settings = None  # 設定オブジェクト（util.load_config等でロード）
        self.data_dir = None  # 一時データ保存先
        self.output_dir = None  # 最終出力先
        self.prompts = None  # プロンプトテンプレート群
        self.inf = None  # 推論バックエンドモジュール
        self._initialized = False

    def initialize(self):
        """
        設定・ディレクトリ・プロンプトの読み込みをまとめて実行。
        """
        from test import setup_directories, load_prompts

        self.settings = util.load_config(str(self.config_path))
        
        # --- 推論バックエンドの動的インポート ---
        backend_name = getattr(self.settings, 'inference_backend', 'vllm')
        if backend_name == "ollama":
            from src import ollama_inf as inf_module
            print("推論バックエンドとして Ollama を使用します。")
        elif backend_name == "api_openai":
            from . import api_openai_inf as inf_module
            print("推論バックエンドとして OpenAI API を使用します。")
        elif backend_name == "api_google":
            from . import api_google_inf as inf_module
            print("推論バックエンドとして Google API を使用します。")
        elif backend_name == "api_openai_comp":
            from . import api_openai_comp_inf as inf_module
            print("推論バックエンドとして OpenAI互換API を使用します。")
        elif backend_name == "vllm":
            from src import vllm_inf as inf_module
            print("推論バックエンドとして vLLM を使用します。")
        else:
            raise ImportError(f"無効な推論バックエンドが指定されました: {backend_name}")
        self.inf = inf_module
        # ---

        self.prompts = load_prompts()
        self.data_dir, self.output_dir = setup_directories(self.settings)
        self._initialized = True

    def generate_questions(self) -> List[Dict]:
        """
        フェーズ1: 質問データ生成（モデルロード・プロンプト構築・推論・重複除去・保存）
        """
        self._require_init()
        from tqdm import tqdm

        settings = self.settings
        prompts = self.prompts
        data_dir = self.data_dir

        # 生成方式に応じてモデルをロード
        if settings.Seed_generation_method == 'base':
            model = self.inf.base_model_load(settings)
            inference_func = self.inf.base_model_inference
            prompt_template = prompts["bare"]
        elif settings.Seed_generation_method == 'inst':
            model = self.inf.inst_model_load(settings)
            inference_func = self.inf.inst_model_inference
            prompt_template = prompts["inst_seed"]
        else:
            raise ValueError("settings.yaml の Seed_generation_method は 'base' または 'inst' である必要があります。")

        questions_file = data_dir / "generated_questions.jsonl"
        if questions_file.exists():
            questions_file.unlink()

        generated_count = 0
        pbar = tqdm(total=settings.Number_of_questions_generated, desc="質問生成中")

        def batched(iterable, n):
            it = iter(iterable)
            while batch := list(islice(it, n)):
                yield batch

        from itertools import islice
        import random

        while generated_count < settings.Number_of_questions_generated:
            batch_size = settings.batch_size
            num_nouns_per_prompt = 5
            total_nouns_to_generate = batch_size * num_nouns_per_prompt

            try:
                noise_nouns = util.random_japanese_nouns(total_nouns_to_generate)
            except ValueError:
                continue

            batch_prompts = []
            for i in range(batch_size):
                start_index = i * num_nouns_per_prompt
                end_index = start_index + num_nouns_per_prompt
                nouns_for_prompt = noise_nouns[start_index:end_index]
                noise = ' '.join(nouns_for_prompt)
                prompt = f"ノイズ: {noise}\n\n{prompt_template}"
                batch_prompts.append(prompt)

            results = inference_func(model, batch_prompts, settings)

            batch_data = []
            for res in results:
                if not res:
                    continue
                if settings.Seed_generation_method == 'base':
                    for line in res.strip().split('\n'):
                        try:
                            data = util.json.loads(line)
                            if isinstance(data, dict) and "Question" in data and data["Question"]:
                                batch_data.append(data)
                        except Exception:
                            continue
                else:
                    batch_data.append({"Question": res.strip(), "Answer": ""})

            if batch_data:
                util.save_jsonl(batch_data, str(questions_file), mode='a')
                generated_count += len(batch_data)
                pbar.update(len(batch_data))

        pbar.close()

        model_name_for_log = settings.base_model_name if settings.Seed_generation_method == 'base' else settings.Instruct_model_name
        del model
        self.inf.unload_model(model_name_for_log)

        all_questions = util.load_jsonl(str(questions_file))
        seen_questions = set()
        unique_questions = []
        for item in all_questions:
            question_text = item.get("Question")
            if question_text and question_text.strip():
                normalized_question = question_text.strip()
                if normalized_question not in seen_questions:
                    unique_questions.append(item)
                    seen_questions.add(normalized_question)
        return unique_questions

    def curate_questions(self, data: List[Dict]) -> List[Dict]:
        """
        フェーズ1: ベースモデル生成時の追加キュレーション
        """
        self._require_init()
        if not data:
            return []
        from tqdm import tqdm
        import math

        settings = self.settings
        prompts = self.prompts

        if settings.Seed_generation_method != 'base':
            return data

        model = self.inf.inst_model_load(settings)
        curation_prompt = prompts["curation_base"]

        def batched(iterable, n):
            it = iter(iterable)
            from itertools import islice
            while batch := list(islice(it, n)):
                yield batch

        chunk_size = 4
        data_chunks = list(batched(data, chunk_size))
        chunk_prompt_pairs = [
            (
                chunk,
                f"{curation_prompt}\n\n{ ' + '.join(d.get('Question', '') for d in chunk) }"
            )
            for chunk in data_chunks
        ]

        curated_data = []
        total_steps = math.ceil(len(chunk_prompt_pairs) / settings.batch_size)
        pbar = tqdm(total=total_steps, desc="ベースデータキュレーション中")

        for pair_batch in batched(chunk_prompt_pairs, settings.batch_size):
            batch_chunks, batch_prompts = zip(*pair_batch)
            results = self.inf.inst_model_inference(model, list(batch_prompts), settings)
            for res, chunk in zip(results, batch_chunks):
                if "yes" in res.lower():
                    curated_data.extend(chunk)
            pbar.update(1)

        pbar.close()
        del model
        self.inf.unload_model(settings.Instruct_model_name)
        return curated_data

    def diversity_filter(self, data: List[Dict]) -> List[Dict]:
        """
        フェーズ1: E5による多様性フィルタリング
        """
        self._require_init()
        if not data:
            return []
        from src import e5

        settings = self.settings
        retention_rate = settings.Data_retention_rate_after_diversity_cut
        cut_rate = 100.0 - retention_rate

        filtered_data = e5.reduce_data_by_diversity(
            data=data,
            data_format='dict',
            cut_rate=cut_rate,
            settings=settings,
            data_key='Question'
        )
        return filtered_data

    def evolve_questions(self, data: List[Dict]) -> List[Dict]:
        """
        フェーズ1: 質問進化パイプライン
        """
        self._require_init()
        if not data:
            return []
        from tqdm import tqdm
        import random

        settings = self.settings
        prompts = self.prompts

        enable_flag = settings.Prompt_evolution
        times = settings.Prompt_evolution_times
        staged_flag = settings.Number_of_stages_of_prompt_evolution
        evo_prompt = prompts["evo_question"]
        data_key = "Question"

        if not enable_flag or times == 0:
            return data

        model = self.inf.inst_model_load(settings)
        evolved_data_stages = {0: data}
        data_to_evolve = [d.copy() for d in data]

        def batched(iterable, n):
            it = iter(iterable)
            from itertools import islice
            while batch := list(islice(it, n)):
                yield batch

        for stage_idx in range(1, times + 1):
            current_stage_data = []
            pbar = tqdm(total=len(data_to_evolve), desc=f"質問進化 {stage_idx}回目")
            for batch in batched(data_to_evolve, settings.batch_size):
                batch_prompts = [f"{evo_prompt}\n\n{d[data_key]}" for d in batch]
                evolved_texts = self.inf.inst_model_inference(model, batch_prompts, settings)
                for i, d in enumerate(batch):
                    new_d = d.copy()
                    new_d[data_key] = evolved_texts[i]
                    current_stage_data.append(new_d)
                pbar.update(len(batch))
            pbar.close()
            evolved_data_stages[stage_idx] = current_stage_data
            data_to_evolve = current_stage_data

        if staged_flag and times > 0:
            final_evolved_data = []
            num_stages = times + 1
            base_count_per_stage = len(data) // num_stages
            for stage_idx in range(num_stages):
                stage_data = evolved_data_stages.get(stage_idx, [])
                count = base_count_per_stage if stage_idx < times else len(data) - len(final_evolved_data)
                final_evolved_data.extend(random.sample(stage_data, min(count, len(stage_data))))
            result_data = final_evolved_data
        else:
            result_data = evolved_data_stages[times]

        del model
        self.inf.unload_model(settings.Instruct_model_name)
        return result_data

    def generate_answers(self, data: List[Dict]) -> List[Dict]:
        """
        フェーズ2: 回答生成（CoT対応版）
        """
        self._require_init()
        if not data:
            return []
        from tqdm import tqdm

        settings = self.settings
        prompts = self.prompts

        # CoTモードが有効な場合
        if getattr(settings, 'Enable_CoT', False):
            return self._generate_cot_answers(data)

        if settings.Using_think_models_for_answer:
            model = self.inf.think_model_load(settings)
            inference_func = self.inf.think_model_inference
            model_name_for_log = settings.think_model_name
        else:
            model = self.inf.inst_model_load(settings)
            inference_func = self.inf.inst_model_inference
            model_name_for_log = settings.Instruct_model_name

        instruction = "以下の質問に日本語で答えてください。"
        answered_data = []
        def batched(iterable, n):
            it = iter(iterable)
            from itertools import islice
            while batch := list(islice(it, n)):
                yield batch

        pbar = tqdm(total=len(data), desc="回答生成中")
        for batch in batched(data, settings.batch_size):
            batch_prompts = [f"{instruction}\n\n{d['Question']}" for d in batch]
            results = inference_func(model, batch_prompts, settings)
            for i, res in enumerate(results):
                d = batch[i].copy()
                if settings.Using_think_models_for_answer:
                    think_part, answer_part = util.separate_think_and_answer(res)
                    d["think"] = think_part
                    d["Answer"] = answer_part
                else:
                    d["Answer"] = res
                answered_data.append(d)
            pbar.update(len(batch))
        pbar.close()

        del model
        self.inf.unload_model(model_name_for_log)
        return answered_data

    def _generate_cot_answers(self, data: List[Dict]) -> List[Dict]:
        """
        Chain of Thought推論による回答生成（アンサンブル対応版）
        """
        from tqdm import tqdm
        settings = self.settings
        prompts = self.prompts

        # アンサンブルモードが有効な場合
        if getattr(settings, 'Enable_Ensemble', False):
            return self._generate_ensemble_cot_answers(data)

        model = self.inf.inst_model_load(settings)
        cot_prompt = prompts.get("cot_ultimate", self._get_default_cot_prompt(settings))
        
        answered_data = []
        def batched(iterable, n):
            it = iter(iterable)
            from itertools import islice
            while batch := list(islice(it, n)):
                yield batch

        pbar = tqdm(total=len(data), desc="CoT推論による回答生成中")
        for batch in batched(data, settings.batch_size):
            batch_prompts = [cot_prompt.format(question=d['Question']) for d in batch]
            results = self.inf.cot_model_inference(model, batch_prompts, settings)
            
            for i, res in enumerate(results):
                d = batch[i].copy()
                
                # CoT推論結果を解析
                if getattr(settings, 'CoT_format', 'structured') == 'structured':
                    cot_parsed = util.parse_cot_reasoning(res)
                    d["cot_reasoning"] = cot_parsed
                    d["Answer"] = cot_parsed["final_answer"]
                    if getattr(settings, 'Save_CoT_process', True):
                        d["reasoning_steps"] = cot_parsed["steps"]
                        d["reasoning_type"] = cot_parsed["reasoning_type"]
                else:
                    # 自然言語形式でそのまま保存
                    d["Answer"] = res
                    d["cot_reasoning"] = res
                
                answered_data.append(d)
            pbar.update(len(batch))
        pbar.close()

        del model
        self.inf.unload_model(settings.Instruct_model_name)
        return answered_data

    def _generate_ensemble_cot_answers(self, data: List[Dict]) -> List[Dict]:
        """
        複数モデルによるアンサンブルCoT推論
        """
        from tqdm import tqdm
        import copy
        settings = self.settings
        prompts = self.prompts

        ensemble_models = getattr(settings, 'Ensemble_models', ["Instruct", "base", "think"])
        ensemble_steps = getattr(settings, 'Ensemble_steps', ["decomposition", "analysis", "hypothesis", "verification", "conclusion"])
        strategy = getattr(settings, 'Ensemble_strategy', 'majority_vote')
        
        # 各ステップ用のプロンプトを生成
        step_prompts = self._generate_step_prompts()
        
        answered_data = []
        def batched(iterable, n):
            it = iter(iterable)
            from itertools import islice
            while batch := list(islice(it, n)):
                yield batch

        pbar = tqdm(total=len(data), desc="アンサンブルCoT推論による回答生成中")
        
        for batch in batched(data, settings.batch_size):
            batch_results = []
            
            for d in batch:
                question = d['Question']
                ensemble_reasoning = {"steps": {}, "models_used": ensemble_models}
                
                # 各ステップを異なるモデルで実行
                for i, step in enumerate(ensemble_steps):
                    model_name = ensemble_models[i % len(ensemble_models)]
                    
                    # モデルロード
                    if model_name == "Instruct":
                        model = self.inf.inst_model_load(settings)
                        inference_func = self.inf.inst_model_inference
                        model_path = settings.Instruct_model_name
                    elif model_name == "base":
                        model = self.inf.base_model_load(settings)
                        inference_func = self.inf.base_model_inference
                        model_path = settings.base_model_name
                    elif model_name == "base1":
                        model = self.inf.base1_model_load(settings)
                        inference_func = self.inf.base1_model_inference
                        model_path = settings.base1_model_name
                    elif model_name == "base2":
                        model = self.inf.base2_model_load(settings)
                        inference_func = self.inf.base2_model_inference
                        model_path = settings.base2_model_name
                    elif model_name == "base3":
                        model = self.inf.base3_model_load(settings)
                        inference_func = self.inf.base3_model_inference
                        model_path = settings.base3_model_name
                    elif model_name == "think":
                        model = self.inf.think_model_load(settings)
                        inference_func = self.inf.think_model_inference
                        model_path = settings.think_model_name
                    else:
                        raise ValueError(f"不明なモデル名: {model_name}. 利用可能なモデル: Instruct, base, base1, base2, base3, think")
                    
                    # ステップ別推論実行
                    step_prompt = step_prompts[step].format(question=question)
                    if i > 0:
                        previous_steps = "\n\n".join([f"**{prev_step}:**\n{ensemble_reasoning['steps'][prev_step]['content']}" 
                                                    for prev_step in ensemble_steps[:i]])
                        step_prompt += f"\n\n**これまでの推論:**\n{previous_steps}"
                    
                    result = inference_func(model, [step_prompt], settings)[0]
                    
                    ensemble_reasoning["steps"][step] = {
                        "content": result,
                        "model": model_name
                    }
                    
                    # モデル解放
                    del model
                    self.inf.unload_model(model_path)
                
                # 最終回答を統合
                final_answer = self._integrate_ensemble_results(ensemble_reasoning, strategy)
                
                result_d = d.copy()
                result_d["Answer"] = final_answer
                result_d["ensemble_reasoning"] = ensemble_reasoning
                result_d["reasoning_type"] = "ensemble_cot"
                
                batch_results.append(result_d)
            
            answered_data.extend(batch_results)
            pbar.update(len(batch))
        
        pbar.close()
        return answered_data

    def _generate_step_prompts(self) -> dict:
        """
        アンサンブル用の各ステップ別プロンプトを生成
        """
        return {
            "decomposition": """# ステップ1：課題の分解と定義

以下の質問について、課題の核心と明らかにすべき点を特定してください。

質問: {question}

この課題の核心は何か？何を明らかにすべきか？
具体的で明確な分析を行ってください。""",
            
            "analysis": """# ステップ2：情報収集と分析

以下の質問について、重要な情報を特定し分析してください。

質問: {question}

どの情報を重視し、どう分析するか？
関連する知識や背景情報を整理してください。""",
            
            "hypothesis": """# ステップ3：仮説の構築

以下の質問について、考えられる解決アプローチや選択肢を提示してください。

質問: {question}

考えられる選択肢やアプローチは何か？
複数の可能性を検討してください。""",
            
            "verification": """# ステップ4：仮説の検証

以下の質問について、各選択肢を検証してください。

質問: {question}

各選択肢のメリット・デメリットは何か？
制約条件と照らし合わせて評価してください。""",
            
            "conclusion": """# ステップ5：結論の導出

以下の質問について、最終的な結論を導出してください。

質問: {question}

なぜその結論が最適だと判断したのか？
論理的根拠を示して最終回答を提示してください。"""
        }

    def _integrate_ensemble_results(self, ensemble_reasoning: dict, strategy: str) -> str:
        """
        アンサンブル結果を統合して最終回答を生成
        """
        if strategy == "majority_vote":
            # 最後のステップ（conclusion）の結果を使用
            return ensemble_reasoning["steps"]["conclusion"]["content"]
        
        elif strategy == "weighted_average":
            # 全ステップの内容を重み付け統合（簡易実装）
            all_contents = []
            for step, result in ensemble_reasoning["steps"].items():
                all_contents.append(f"**{step}:** {result['content']}")
            return "\n\n".join(all_contents)
        
        elif strategy == "best_confidence":
            # 最も長い（詳細な）回答を選択（簡易実装）
            best_step = max(ensemble_reasoning["steps"].items(), 
                          key=lambda x: len(x[1]["content"]))
            return best_step[1]["content"]
        
        else:
            return ensemble_reasoning["steps"]["conclusion"]["content"]

    def _get_default_cot_prompt(self, settings) -> str:
        """
        デフォルトのCoTプロンプトを生成
        """
        cot_mode = getattr(settings, 'CoT_mode', 'step_by_step')
        max_steps = getattr(settings, 'CoT_max_steps', 5)
        
        if cot_mode == "ultimate":
            return """# 命令書

あなたは、世界最高の専門家です。以下の制約条件と入力情報に基づき、最高の回答を作成してください。
その際、あなたの思考プロセスを思考のステップとして詳細に記述し、その上で最終的な回答を生成してください。

## 1. 制約条件
- 日本語で回答する
- 論理的で理解しやすい説明を心がける
- 根拠を明確に示す
- 適切な専門知識を活用する

## 2. 入力情報
- 課題：{question}

## 3. 出力形式
### ◆思考のステップ

**ステップ1：課題の分解と定義**
この課題の核心は何か？何を明らかにすべきか？

**ステップ2：情報収集と分析**
どの情報を重視し、どう分析するか？

**ステップ3：仮説の構築**
考えられる選択肢やアプローチは何か？

**ステップ4：仮説の検証**
各選択肢のメリット・デメリットは何か？制約条件と照らし合わせるとどうなるか？

**ステップ5：結論の導出**
なぜその結論が最適だと判断したのか？

### ◆最終的な回答
（思考のステップに基づいた、具体的で実行可能な最終回答をここに記述してください）"""
        
        elif cot_mode == "step_by_step":
            return f"""以下の質問について、段階的に考えて回答してください。

推論の手順：
1. 問題を理解する
2. 必要な情報を整理する
3. 段階的に解決する（最大{max_steps}ステップ）
4. 最終回答を導く

各ステップを明確に示し、最後に「最終回答: 」として答えを示してください。"""
        
        elif cot_mode == "reasoning":
            return """以下の質問について、論理的な推論過程を示しながら回答してください。

• まず問題の本質を把握してください
• 関連する知識や概念を整理してください  
• 論理的な推論を展開してください
• 最終的な結論を明確に示してください"""
        
        elif cot_mode == "comprehensive":
            return f"""以下の質問について、包括的な分析を行って回答してください。

分析の観点：
- 問題の多角的理解
- 関連情報の体系的整理
- 段階的推論の展開（最大{max_steps}段階）
- 結論の論理的導出
- 回答の妥当性検証

思考過程を明確に示し、最終的に明確な答えを提示してください。"""
        
        else:
            return "以下の質問について、段階的に考えて回答してください。"

    def evolve_answers(self, data: List[Dict]) -> List[Dict]:
        """
        フェーズ2: 回答進化パイプライン
        """
        self._require_init()
        if not data:
            return []
        from tqdm import tqdm
        import random

        settings = self.settings
        prompts = self.prompts

        enable_flag = settings.Answer_evolution
        times = settings.Answer_evolution_times
        staged_flag = settings.Number_of_stages_of_answer_evolution
        evo_prompt = prompts["evo_answer"]
        data_key = "Answer"

        if not enable_flag or times == 0:
            return data

        model = self.inf.inst_model_load(settings)
        evolved_data_stages = {0: data}
        data_to_evolve = [d.copy() for d in data]

        def batched(iterable, n):
            it = iter(iterable)
            from itertools import islice
            while batch := list(islice(it, n)):
                yield batch

        for stage_idx in range(1, times + 1):
            current_stage_data = []
            pbar = tqdm(total=len(data_to_evolve), desc=f"回答進化 {stage_idx}回目")
            for batch in batched(data_to_evolve, settings.batch_size):
                batch_prompts = [f"{evo_prompt}\n\n{d[data_key]}" for d in batch]
                evolved_texts = self.inf.inst_model_inference(model, batch_prompts, settings)
                for i, d in enumerate(batch):
                    new_d = d.copy()
                    new_d[data_key] = evolved_texts[i]
                    current_stage_data.append(new_d)
                pbar.update(len(batch))
            pbar.close()
            evolved_data_stages[stage_idx] = current_stage_data
            data_to_evolve = current_stage_data

        if staged_flag and times > 0:
            final_evolved_data = []
            num_stages = times + 1
            base_count_per_stage = len(data) // num_stages
            for stage_idx in range(num_stages):
                stage_data = evolved_data_stages.get(stage_idx, [])
                count = base_count_per_stage if stage_idx < times else len(data) - len(final_evolved_data)
                final_evolved_data.extend(random.sample(stage_data, min(count, len(stage_data))))
            result_data = final_evolved_data
        else:
            result_data = evolved_data_stages[times]

        del model
        self.inf.unload_model(settings.Instruct_model_name)
        return result_data

    def curate_final(self, data: List[Dict]) -> List[Dict]:
        """
        フェーズ2: 最終キュレーション
        """
        self._require_init()
        if not data:
            return []
        settings = self.settings
        prompts = self.prompts
        if not settings.Data_curation:
            return data

        from tqdm import tqdm

        model = self.inf.inst_model_load(settings)
        curation_prompt = prompts["curation"]

        def batched(iterable, n):
            it = iter(iterable)
            from itertools import islice
            while batch := list(islice(it, n)):
                yield batch

        curated_data = []
        pbar = tqdm(total=len(data), desc="最終データキュレーション中")
        for batch in batched(data, settings.batch_size):
            batch_prompts = []
            for d in batch:
                prompt = f'{curation_prompt}\n\n質問: {d["Question"]}\n回答: {d["Answer"]}'
                batch_prompts.append(prompt)
            results = self.inf.inst_model_inference(model, batch_prompts, settings)
            for i, res in enumerate(results):
                if "yes" in res.lower():
                    curated_data.append(batch[i])
            pbar.update(len(batch))
        pbar.close()

        del model
        self.inf.unload_model(settings.Instruct_model_name)
        return curated_data

    def save_dataset(self, data: List[Dict], path: Optional[Union[str, Path]] = None) -> Path:
        """
        最終データセットをJSONLで保存し、保存先Pathを返す（CoT対応版）
        """
        self._require_init()
        if not data:
            return None
        from src import util

        output_dir = self.output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        backend_name = getattr(self.settings, 'inference_backend', 'vllm')
        
        # CoTモードの場合はファイル名に反映
        cot_suffix = "_cot" if getattr(self.settings, 'Enable_CoT', False) else ""
        output_filename = f"final_dataset_{backend_name}{cot_suffix}_{timestamp}.jsonl"
        output_file = Path(path) if path else output_dir / output_filename

        final_dataset = []
        for item in data:
            formatted_item = {
                "instruction": "以下の質問に日本語で答えてください。",
                "input": item.get("Question", ""),
                "output": item.get("Answer", "")
            }
            
            # CoT情報がある場合は追加フィールドとして保存
            if "cot_reasoning" in item:
                formatted_item["cot_reasoning"] = item["cot_reasoning"]
            
            if "reasoning_steps" in item:
                formatted_item["reasoning_steps"] = item["reasoning_steps"]
                
            if "reasoning_type" in item:
                formatted_item["reasoning_type"] = item["reasoning_type"]
            
            # アンサンブル情報を保存
            if "ensemble_reasoning" in item:
                formatted_item["ensemble_reasoning"] = item["ensemble_reasoning"]
            
            # 従来のthink情報も保持
            if "think" in item:
                formatted_item["think"] = item["think"]
            
            final_dataset.append(formatted_item)

        util.save_jsonl(final_dataset, str(output_file), mode='w')
        return output_file

    def run(self, end_stage: str = "all") -> Path:
        """
        パイプライン全体を一括実行。途中までで停止も可。
        """
        self._require_init()
        # 各フェーズを順に呼び出し
        return Path("output/final_dataset.jsonl")

    def _require_init(self):
        if not self._initialized:
            self.initialize()

# --- 利用者向けエントリポイント ---
def setup_pipeline(config_path: Union[str, Path] = "config/config.yaml") -> Pipeline:
    """
    設定ファイルを読み込み、Pipelineオブジェクトを返す。
    """
    return Pipeline(config_path)
