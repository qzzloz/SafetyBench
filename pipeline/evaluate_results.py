"""
Result evaluation stage
"""

import json
from typing import List, Dict, Any
from pathlib import Path

from .base_pipeline import BasePipeline, process_with_strategy
from core.data_formats import ModelResponse, EvaluationResult, PipelineConfig
from core.unified_registry import UNIFIED_REGISTRY
from .resource_policy import policy_for_evaluation


class ResultEvaluator(BasePipeline):
    """Result evaluator"""

    def __init__(self, config: PipelineConfig):
        super().__init__(config, stage_name="evaluation")
        self.evaluation_configs = config.evaluation

    def load_model_responses(
        self,
        attack_names: List[str] = None,
        model_names: List[str] = None,
        defense_names: List[str] = None,
    ) -> List[ModelResponse]:
        """Load model responses

        Args:
            attack_names: List of attack methods to load, if None then read from configuration
            model_names: List of models to load, if None then read from configuration
            defense_names: List of defense methods to load, if None then read from configuration
        """
        # Get parameters (priority: passed parameters, then read from configuration)
        if model_names is None:
            model_names = self.config.response_generation.get("models", [])

        if defense_names is None:
            defense_names = self.config.response_generation.get("defenses", ["None"])

        if attack_names is None:
            attack_names = self.config.test_case_generation.get("attacks", [])

        # Define file finder function
        def find_response_files():
            if not model_names:
                self.logger.error("Models not specified, cannot load model responses")
                return []

            if not attack_names:
                self.logger.error(
                    "Attack methods not specified, cannot load model responses"
                )
                return []

            files = []
            responses_dir = Path(self.config.system["output_dir"]) / "responses"

            for attack_name in attack_names:
                for model_name in model_names:
                    for defense_name in defense_names:
                        defense_dir = responses_dir / defense_name
                        # Prefer JSONL outputs; fall back to legacy JSON list if present
                        response_file_jsonl = (
                            defense_dir
                            / f"attack_{attack_name}_model_{model_name}.jsonl"
                        )
                        response_file_json = (
                            defense_dir
                            / f"attack_{attack_name}_model_{model_name}.json"
                        )
                        if response_file_jsonl.exists():
                            files.append(response_file_jsonl)
                        elif response_file_json.exists():
                            files.append(response_file_json)
            return files

        # Use unified data loading method
        return self.load_data_files(
            data_type="model responses",
            config_key="input_responses",
            file_finder=find_response_files,
            data_parser=lambda item: ModelResponse.from_dict(item),
        )

    def get_model_responses_count(self) -> int:
        """Get model response count"""
        responses = self.load_model_responses()
        return len(responses)

    def evaluate_single_response(
        self,
        model_response: ModelResponse,
        evaluator_name: str,
        evaluator=None,
    ) -> EvaluationResult:
        """Evaluate single model response"""
        try:
            # Create evaluator instance (unless provided for reuse)
            if evaluator is None:
                evaluator_config = self.evaluation_configs.get(
                    "evaluator_params", {}
                ).get(evaluator_name, {})
                evaluator = UNIFIED_REGISTRY.create_evaluator(
                    evaluator_name, evaluator_config
                )

            # Execute evaluation
            evaluation_result = evaluator.evaluate_response(model_response)

            # ---- Enrich result with contextual fields to make it self-contained ----
            # NOTE: test_case_id is NOT globally unique across attacks/models/defenses.
            # We must persist enough identifiers for correct grouping/analysis.
            evaluation_result.attack_method = model_response.metadata.get("attack_method")
            evaluation_result.original_prompt = model_response.metadata.get("original_prompt")
            evaluation_result.jailbreak_prompt = model_response.metadata.get("jailbreak_prompt")

            # Prefer jailbreak image if present; fall back to any image_path in metadata
            evaluation_result.image_path = (
                model_response.metadata.get("jailbreak_image_path")
                or model_response.metadata.get("image_path")
            )

            evaluation_result.model_response = model_response.model_response
            evaluation_result.model_name = model_response.model_name
            evaluation_result.defense_method = model_response.metadata.get(
                "defense_method", "None"
            )

            # Record evaluator name explicitly for downstream grouping
            if evaluation_result.metadata is None:
                evaluation_result.metadata = {}
            evaluation_result.metadata["evaluator_name"] = evaluator_name
            # 실제 judge 모델명도 기록 (파일명뿐 아니라 파일 내용으로도 구분되게)
            evaluation_result.metadata["judge_model"] = (
                self.evaluation_configs.get("evaluator_params", {})
                .get(evaluator_name, {})
                .get("model", evaluator_name)
            )

            self.logger.debug(
                f"Successfully evaluated response {model_response.test_case_id}"
            )
            return evaluation_result

        except Exception as e:
            self.logger.error(
                f"Failed to evaluate response {model_response.test_case_id}: {e}"
            )
            raise

    def run(self, **kwargs) -> List[EvaluationResult]:
        """Run result evaluation, supports checkpoint resume and real-time batch saving"""
        if not self.validate_config():
            return []

        # Get batch size parameter (priority: kwargs parameter, then configuration parameter)
        batch_size = kwargs.get("batch_size", self.config.batch_size)
        self.logger.info(f"Starting result evaluation stage (batch size: {batch_size})")

        # Get attack method, model and defense method lists
        attack_names = self.config.test_case_generation.get("attacks", [])
        model_names = self.config.response_generation.get("models", [])
        defense_names = self.config.response_generation.get("defenses", ["None"])

        # Load model responses
        model_responses = self.load_model_responses(
            attack_names=attack_names,
            model_names=model_names,
            defense_names=defense_names,
        )
        if not model_responses:
            self.logger.error("No available model responses")
            return []

        # Get evaluator configuration
        evaluator_names = self.evaluation_configs.get("evaluators", ["default_judge"])

        self.logger.info(
            f"Will evaluate {len(model_responses)} model responses using {len(evaluator_names)} evaluators"
        )

        # Extract attack method, model and defense method information from model responses
        attack_names = set()
        model_names = set()
        defense_names = set()

        for response in model_responses:
            # Get attack method name from metadata
            attack_name = response.metadata.get("attack_method")
            if attack_name:
                attack_names.add(attack_name)

            model_names.add(response.model_name)
            defense_name = response.metadata.get("defense_method", "None")
            defense_names.add(defense_name)

        self.logger.info(f"Extracted attack methods: {list(attack_names)}")
        self.logger.info(f"Extracted models: {list(model_names)}")
        self.logger.info(f"Extracted defense methods: {list(defense_names)}")

        # Generate all tasks
        pending_tasks = []

        for model_response in model_responses:
            for evaluator_name in evaluator_names:
                # Generate task ID
                task_config = {
                    "test_case_id": model_response.test_case_id,
                    "evaluator_name": evaluator_name,
                    "evaluator_params": self.evaluation_configs.get(
                        "evaluator_params", {}
                    ).get(evaluator_name, {}),
                    "model_name": model_response.model_name,
                    "defense_method": model_response.metadata.get(
                        "defense_method", "None"
                    ),
                }
                task_id = f"{model_response.test_case_id}_{evaluator_name}_{self.get_task_hash(task_config)}"
                pending_tasks.append((model_response, evaluator_name, task_id))

        pending_count = len(pending_tasks)
        self.logger.info(f"Total tasks: {pending_count}")

        # Group tasks by attack method+model+defense method+evaluator
        tasks_by_combo = {}
        for model_response, evaluator_name, task_id in pending_tasks:
            # Get attack method name from model_response metadata
            attack_name = model_response.metadata.get("attack_method")
            model_name = model_response.model_name
            defense_name = model_response.metadata.get("defense_method", "None")

            key = (attack_name, model_name, defense_name, evaluator_name)
            if key not in tasks_by_combo:
                tasks_by_combo[key] = []
            tasks_by_combo[key].append((model_response, evaluator_name, task_id))

        # Check if each combination has generated complete evaluation results
        completed_combos = []
        pending_combos_to_process = []

        for (
            attack_name,
            model_name,
            defense_name,
            evaluator_name,
        ), combo_tasks in tasks_by_combo.items():
            if not attack_name:  # Skip tasks without attack method name
                continue

            # Generate filename for this combination
            # 파일명의 evaluator 자리에 evaluator 이름(default_judge) 대신
            # 실제 judge 모델명(evaluator_params.<evaluator>.model, 예: qwen2.5-vl-32b-judge)
            # 을 넣어, 어떤 judge 로 채점했는지 파일명으로 구분되게 한다.
            # model 이 지정 안 됐으면 기존처럼 evaluator_name 으로 폴백.
            judge_model_for_name = (
                self.evaluation_configs.get("evaluator_params", {})
                .get(evaluator_name, {})
                .get("model", evaluator_name)
            )
            _, combo_filename = self._generate_filename(
                "evaluation",
                attack_name=attack_name,
                model_name=model_name,
                defense_name=defense_name,
                evaluator_name=judge_model_for_name,
            )

            # Calculate expected evaluation result count for this combination
            # Need to count responses for this attack method+model+defense method combination
            expected_count = 0
            for response in model_responses:
                # Get attack method name from response metadata
                resp_attack_name = response.metadata.get("attack_method")
                resp_defense_name = response.metadata.get("defense_method", "None")
                if (
                    resp_attack_name == attack_name
                    and response.model_name == model_name
                    and resp_defense_name == defense_name
                ):
                    expected_count += 1

            # Check existing evaluation result files
            existing_evaluations = self.load_results(combo_filename)

            if len(existing_evaluations) >= expected_count:
                self.logger.info(
                    f"Combination {attack_name}+{model_name}+{defense_name}+{evaluator_name} has complete evaluation results: {len(existing_evaluations)}/{expected_count}"
                )
                completed_combos.append(
                    (
                        attack_name,
                        model_name,
                        defense_name,
                        evaluator_name,
                        combo_filename,
                        combo_tasks,
                    )
                )
            else:
                self.logger.info(
                    f"Combination {attack_name}+{model_name}+{defense_name}+{evaluator_name} needs to generate evaluation results: {len(existing_evaluations)}/{expected_count}"
                )
                pending_combos_to_process.append(
                    (
                        attack_name,
                        model_name,
                        defense_name,
                        evaluator_name,
                        combo_filename,
                        combo_tasks,
                        expected_count,
                    )
                )

        # If all combinations are completed, directly load existing results
        if not pending_combos_to_process:
            self.logger.info("All combinations completed, loading existing results")
            all_evaluations = self._load_all_evaluations(model_responses)
            self.logger.info(f"Total loaded {len(all_evaluations)} evaluation results")
            return all_evaluations

        self.logger.info(
            f"Need to process {len(pending_combos_to_process)} combinations"
        )

        all_evaluations = []

        # First load evaluation results from completed combinations
        for (
            attack_name,
            model_name,
            defense_name,
            evaluator_name,
            combo_filename,
            combo_tasks,
        ) in completed_combos:
            existing_results = self.load_results(combo_filename)
            for item in existing_results:
                try:
                    evaluation = EvaluationResult.from_dict(item)
                    all_evaluations.append(evaluation)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to parse evaluation result ({attack_name}, {model_name}, {defense_name}, {evaluator_name}): {e}"
                    )
            self.logger.info(
                f"Loaded {len(existing_results)} evaluation results from {combo_filename}"
            )

        # Generate evaluation results for each combination that needs processing
        for (
            attack_name,
            model_name,
            defense_name,
            evaluator_name,
            combo_filename,
            combo_tasks,
            expected_count,
        ) in pending_combos_to_process:
            self.logger.info(
                f"Processing combination: attack={attack_name}, model={model_name}, defense={defense_name}, evaluator={evaluator_name}, tasks={len(combo_tasks)}"
            )

            evaluator_config = self.evaluation_configs.get("evaluator_params", {}).get(
                evaluator_name, {}
            )
            policy = policy_for_evaluation(
                evaluator_config, default_max_workers=self.config.max_workers
            )
            self.logger.info(
                f"Resource policy for evaluator={evaluator_name}: strategy={policy.strategy}, max_workers={policy.max_workers} ({policy.reason})"
            )

            if policy.strategy == "batched" and policy.batched_impl == "evaluator_local":
                # Local evaluator: single worker + batched processing, reuse evaluator instance
                from .base_pipeline import batch_save_context

                evaluator_instance = UNIFIED_REGISTRY.create_evaluator(
                    evaluator_name, evaluator_config
                )
                with batch_save_context(
                    pipeline=self,
                    output_filename=combo_filename,
                    batch_size=batch_size,
                    total_items=len(combo_tasks),
                    desc=f"Evaluate results (local evaluator, {attack_name}, {model_name}, {defense_name}, {evaluator_name})",
                ) as save_manager:
                    for task_item in combo_tasks:
                        model_response, evaluator_name, task_id = task_item
                        try:
                            evaluation = self.evaluate_single_response(
                                model_response,
                                evaluator_name,
                                evaluator=evaluator_instance,
                            )
                            save_manager.add_result(evaluation.to_dict())
                        except Exception as e:
                            self.logger.error(
                                f"Evaluation task failed ({model_response.test_case_id}, {evaluator_name}): {e}"
                            )
            else:
                # Parallel evaluator (API): keep parallel strategy, but follow policy max_workers
                def process_task(task_item):
                    model_response, evaluator_name, task_id = task_item
                    try:
                        evaluation = self.evaluate_single_response(
                            model_response, evaluator_name
                        )
                        return evaluation.to_dict()
                    except Exception as e:
                        self.logger.error(
                            f"Evaluation task failed ({model_response.test_case_id}, {evaluator_name}): {e}"
                        )
                        return None

                process_with_strategy(
                    items=combo_tasks,
                    process_func=process_task,
                    pipeline=self,
                    output_filename=combo_filename,
                    batch_size=batch_size,
                    max_workers=policy.max_workers,
                    strategy_name="parallel",
                    desc=f"Evaluate results ({attack_name}, {model_name}, {defense_name})",
                )

            # Load results for this combination
            combo_results = self.load_results(combo_filename)
            combo_evaluations = []
            for item in combo_results:
                try:
                    evaluation = EvaluationResult.from_dict(item)
                    combo_evaluations.append(evaluation)
                except Exception as e:
                    self.logger.warning(f"Failed to parse evaluation result: {e}")

            all_evaluations.extend(combo_evaluations)
            self.logger.info(
                f"Combination completed: attack={attack_name}, model={model_name}, defense={defense_name}, generated {len(combo_evaluations)} evaluation results"
            )

        if all_evaluations:
            self.logger.info(
                f"Result evaluation completed, evaluated {len(all_evaluations)} results in total"
            )

            # Generate statistical report
            self._generate_report(all_evaluations, model_responses)
        else:
            self.logger.warning("No evaluation results generated")

        return all_evaluations

    def _load_all_evaluations(
        self, model_responses: List[ModelResponse]
    ) -> List[EvaluationResult]:
        """Load evaluation results for all attack method+model+defense method combinations"""
        all_evaluations = []

        # Extract all unique combinations from model responses
        combos = set()
        for response in model_responses:
            # Get attack method name from response metadata
            attack_name = response.metadata.get("attack_method")
            model_name = response.model_name
            defense_name = response.metadata.get("defense_method", "None")
            # We also need evaluator dimension; try to infer from existing evaluation files later.

            if attack_name:  # Ensure attack method name is not empty
                combos.add((attack_name, model_name, defense_name))

        # Since evaluator_name is now part of filename, we load by scanning evaluation directory
        # for matching patterns instead of guessing evaluator names here.
        evaluations_dir = Path(self.config.system["output_dir"]) / "evaluations"
        if not evaluations_dir.exists():
            self.logger.info("Evaluations directory does not exist, nothing to load")
            return []

        for attack_name, model_name, defense_name in combos:
            # Match all evaluator-specific files for this combo
            pattern_jsonl = f"attack_{attack_name}_model_{model_name}_defense_{defense_name}_evaluator_*.jsonl"
            pattern_json = f"attack_{attack_name}_model_{model_name}_defense_{defense_name}_evaluator_*.json"
            matched_files = list(evaluations_dir.glob(pattern_jsonl))
            if not matched_files:
                matched_files = list(evaluations_dir.glob(pattern_json))
            for combo_filename in matched_files:
                combo_results = self.load_results(combo_filename)
                for item in combo_results:
                    try:
                        evaluation = EvaluationResult.from_dict(item)
                        all_evaluations.append(evaluation)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to parse evaluation result ({attack_name}, {model_name}, {defense_name}, file={combo_filename.name}): {e}"
                        )
                if combo_results:
                    self.logger.debug(
                        f"Loaded {len(combo_results)} evaluation results from {combo_filename}"
                    )

        self.logger.info(f"Total loaded {len(all_evaluations)} evaluation results")
        return all_evaluations

    def _generate_report(
        self, evaluations: List[EvaluationResult], responses: List[ModelResponse]
    ):
        """Generate evaluation report - calculate separately by evaluation file (attack method+model+defense method)"""
        # Group by evaluation "file key" (attack + model + defense + evaluator)
        # IMPORTANT: test_case_id is not unique across attack methods; never use it as a global join key.
        eval_by_file: Dict[str, Dict[str, Any]] = {}

        for eval_ in evaluations:
            attack_name = eval_.attack_method or eval_.metadata.get("attack_method", "unknown")
            model_name = eval_.model_name or eval_.metadata.get("model_name", "unknown")
            defense_name = eval_.defense_method or eval_.metadata.get("defense_method", "None")
            evaluator_name = eval_.metadata.get("evaluator_name", "unknown")

            file_key = f"{attack_name}_{model_name}_{defense_name}_{evaluator_name}"

            if file_key not in eval_by_file:
                eval_by_file[file_key] = {
                    "attack_method": attack_name,
                    "model_name": model_name,
                    "defense_method": defense_name,
                    "evaluator_name": evaluator_name,
                    "total": 0,
                    "success": 0,
                    "scores": [],
                }

            eval_by_file[file_key]["total"] += 1
            if eval_.success:
                eval_by_file[file_key]["success"] += 1
            eval_by_file[file_key]["scores"].append(eval_.judge_score)

        # Calculate statistics for each file
        report = {
            "overview": {
                "total_evaluations": len(evaluations),
                "total_responses": len(responses),
                "total_files": len(eval_by_file),
                "success_rate_overall": 0,
            },
            "by_file": {},  # Statistics by file
            "summary_by_attack": {},  # Summary by attack method
            "summary_by_model": {},  # Summary by model
            "summary_by_defense": {},  # Summary by defense method
            "summary_by_evaluator": {},  # Summary by evaluator
        }

        total_success = 0
        total_evaluations = 0

        # Calculate statistics by file
        for file_key, stats in eval_by_file.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                avg_score = (
                    sum(stats["scores"]) / len(stats["scores"])
                    if stats["scores"]
                    else 0
                )

                # Calculate score distribution
                score_distribution = {}
                for score in stats["scores"]:
                    score_key = str(score)
                    if score_key not in score_distribution:
                        score_distribution[score_key] = 0
                    score_distribution[score_key] += 1

                report["by_file"][file_key] = {
                    "attack_method": stats["attack_method"],
                    "model_name": stats["model_name"],
                    "defense_method": stats["defense_method"],
                    "total": stats["total"],
                    "success": stats["success"],
                    "success_rate": round(success_rate, 4),
                    "average_score": round(avg_score, 2),
                    "score_distribution": score_distribution,
                    "min_score": min(stats["scores"]) if stats["scores"] else 0,
                    "max_score": max(stats["scores"]) if stats["scores"] else 0,
                    "median_score": (
                        self._calculate_median(stats["scores"])
                        if stats["scores"]
                        else 0
                    ),
                }

                total_success += stats["success"]
                total_evaluations += stats["total"]

        # Calculate overall success rate
        if total_evaluations > 0:
            report["overview"]["success_rate_overall"] = round(
                total_success / total_evaluations, 4
            )

        # Summary by attack method
        attack_stats = {}
        for file_key, stats in eval_by_file.items():
            attack_name = stats["attack_method"]
            if attack_name not in attack_stats:
                attack_stats[attack_name] = {"total": 0, "success": 0, "scores": []}

            attack_stats[attack_name]["total"] += stats["total"]
            attack_stats[attack_name]["success"] += stats["success"]
            attack_stats[attack_name]["scores"].extend(stats["scores"])

        for attack_name, stats in attack_stats.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                avg_score = (
                    sum(stats["scores"]) / len(stats["scores"])
                    if stats["scores"]
                    else 0
                )

                report["summary_by_attack"][attack_name] = {
                    "total": stats["total"],
                    "success": stats["success"],
                    "success_rate": round(success_rate, 4),
                    "average_score": round(avg_score, 2),
                }

        # Summary by model
        model_stats = {}
        for file_key, stats in eval_by_file.items():
            model_name = stats["model_name"]
            if model_name not in model_stats:
                model_stats[model_name] = {"total": 0, "success": 0, "scores": []}

            model_stats[model_name]["total"] += stats["total"]
            model_stats[model_name]["success"] += stats["success"]
            model_stats[model_name]["scores"].extend(stats["scores"])

        for model_name, stats in model_stats.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                avg_score = (
                    sum(stats["scores"]) / len(stats["scores"])
                    if stats["scores"]
                    else 0
                )

                report["summary_by_model"][model_name] = {
                    "total": stats["total"],
                    "success": stats["success"],
                    "success_rate": round(success_rate, 4),
                    "average_score": round(avg_score, 2),
                }

        # Summary by defense method
        defense_stats = {}
        for file_key, stats in eval_by_file.items():
            defense_name = stats["defense_method"]
            if defense_name not in defense_stats:
                defense_stats[defense_name] = {"total": 0, "success": 0, "scores": []}

            defense_stats[defense_name]["total"] += stats["total"]
            defense_stats[defense_name]["success"] += stats["success"]
            defense_stats[defense_name]["scores"].extend(stats["scores"])

        for defense_name, stats in defense_stats.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                avg_score = (
                    sum(stats["scores"]) / len(stats["scores"])
                    if stats["scores"]
                    else 0
                )

                report["summary_by_defense"][defense_name] = {
                    "total": stats["total"],
                    "success": stats["success"],
                    "success_rate": round(success_rate, 4),
                    "average_score": round(avg_score, 2),
                }

        # Summary by evaluator
        evaluator_stats = {}
        for file_key, stats in eval_by_file.items():
            evaluator_name = stats.get("evaluator_name", "unknown")
            if evaluator_name not in evaluator_stats:
                evaluator_stats[evaluator_name] = {"total": 0, "success": 0, "scores": []}
            evaluator_stats[evaluator_name]["total"] += stats["total"]
            evaluator_stats[evaluator_name]["success"] += stats["success"]
            evaluator_stats[evaluator_name]["scores"].extend(stats["scores"])

        for evaluator_name, stats in evaluator_stats.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                avg_score = (
                    sum(stats["scores"]) / len(stats["scores"])
                    if stats["scores"]
                    else 0
                )
                report["summary_by_evaluator"][evaluator_name] = {
                    "total": stats["total"],
                    "success": stats["success"],
                    "success_rate": round(success_rate, 4),
                    "average_score": round(avg_score, 2),
                }

        # Save report
        report_file = self.output_dir / "evaluation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Generate brief statistics
        self.logger.info("=== Evaluation Report Summary ===")
        self.logger.info(
            f"Overall success rate: {report['overview']['success_rate_overall']:.2%} ({total_success}/{total_evaluations})"
        )
        self.logger.info(f"Total evaluation files: {len(eval_by_file)}")

        # Display statistics by file
        self.logger.info("\n=== Statistics by Evaluation File ===")
        for file_key, stats in report["by_file"].items():
            self.logger.info(
                f"{file_key}: Success rate={stats['success_rate']:.2%} ({stats['success']}/{stats['total']}), "
                f"Average score={stats['average_score']:.2f}, Score range={stats['min_score']}-{stats['max_score']}"
            )

        # Display summary by attack method
        self.logger.info("\n=== Summary by Attack Method ===")
        for attack_name, stats in report["summary_by_attack"].items():
            self.logger.info(
                f"{attack_name}: Success rate={stats['success_rate']:.2%} ({stats['success']}/{stats['total']}), "
                f"Average score={stats['average_score']:.2f}"
            )

        # Display summary by model
        self.logger.info("\n=== Summary by Model ===")
        for model_name, stats in report["summary_by_model"].items():
            self.logger.info(
                f"{model_name}: Success rate={stats['success_rate']:.2%} ({stats['success']}/{stats['total']}), "
                f"Average score={stats['average_score']:.2f}"
            )

        # Display summary by defense method
        self.logger.info("\n=== Summary by Defense Method ===")
        for defense_name, stats in report["summary_by_defense"].items():
            self.logger.info(
                f"{defense_name}: Success rate={stats['success_rate']:.2%} ({stats['success']}/{stats['total']}), "
                f"Average score={stats['average_score']:.2f}"
            )

        # Display summary by evaluator
        self.logger.info("\n=== Summary by Evaluator ===")
        for evaluator_name, stats in report["summary_by_evaluator"].items():
            self.logger.info(
                f"{evaluator_name}: Success rate={stats['success_rate']:.2%} ({stats['success']}/{stats['total']}), "
                f"Average score={stats['average_score']:.2f}"
            )

        self.logger.info(f"\nDetailed report saved to: {report_file}")

    def _calculate_median(self, scores: List[float]) -> float:
        """Calculate median"""
        if not scores:
            return 0.0

        sorted_scores = sorted(scores)
        n = len(sorted_scores)

        if n % 2 == 0:
            # Even number of elements, take average of middle two
            mid1 = sorted_scores[n // 2 - 1]
            mid2 = sorted_scores[n // 2]
            return (mid1 + mid2) / 2
        else:
            # Odd number of elements, take middle value
            return sorted_scores[n // 2]

    def validate_config(self) -> bool:
        """Validate configuration"""
        if not super().validate_config():
            return False

        return True