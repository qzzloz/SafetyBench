"""
Model response generation stage
Supports pre-processing and post-processing defense methods
"""

import json
from typing import List, Dict, Any, Tuple
from pathlib import Path

from .base_pipeline import BasePipeline, process_with_strategy
from core.data_formats import TestCase, ModelResponse, PipelineConfig
from core.unified_registry import UNIFIED_REGISTRY
from utils.logging_utils import log_with_context


from core.base_classes import BaseDefense
from .resource_policy import policy_for_response_generation

class ResponseGenerator(BasePipeline):
    """Model response generator"""

    def __init__(self, config: PipelineConfig):
        super().__init__(config, stage_name="response_generation")
        self.response_configs = config.response_generation

    def load_test_cases(self, attack_names: List[str] = None) -> List[TestCase]:
        """Load test cases

        Args:
            attack_names: List of attack methods to load, if None then read from configuration
        """
        # Get attack method list
        if attack_names is None:
            attack_names = self.config.test_case_generation.get("attacks", [])

        # Define file finder function
        def find_test_case_files():
            if not attack_names:
                self.logger.error(
                    "Attack methods not specified, cannot load test cases"
                )
                return []

            files = []
            for attack_name in attack_names:
                attack_config = self.config.test_case_generation.get(
                    "attack_params", {}
                ).get(attack_name, {})
                target_model_name = attack_config.get("target_model_name")

                try:
                    _, test_cases_file = self._generate_filename(
                        "test_case_generation",
                        attack_name=attack_name,
                        target_model_name=target_model_name,
                    )
                    if test_cases_file.exists():
                        files.append(test_cases_file)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to generate test case file path (attack method: {attack_name}): {e}"
                    )
            return files

        # Use unified data loading method
        return self.load_data_files(
            data_type="test cases",
            config_key="input_test_cases",
            file_finder=find_test_case_files,
            data_parser=lambda item: TestCase.from_dict(item),
        )

    def get_test_cases_count(self) -> int:
        """Get test case count"""
        test_cases = self.load_test_cases()
        return len(test_cases)

    def apply_defense(
        self, test_case: TestCase, defense_name: str, model_name: str
    ) -> Tuple[TestCase, Any]:
        """Apply defense method, return defended test case and defense instance"""
        if defense_name == "None" or not defense_name:
            return test_case, None

        try:
            defense_config = self.response_configs.get("defense_params", {}).get(
                defense_name, {}
            )
            # Add model configuration to defense configuration so defense method can access it
            defense_config["output_dir"] = self.output_dir / defense_name
            defense_config["target_model_name"] = model_name
            defense = UNIFIED_REGISTRY.create_defense(defense_name, defense_config)

            if defense is None:
                error_msg = f"Failed to create defense method '{defense_name}', please check if the defense method is correctly registered and configured"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            defended_test_case = defense.apply_defense(test_case)
            self.logger.debug(
                f"Applied defense {defense_name} to test case {test_case.test_case_id}"
            )
            return defended_test_case, defense
        except Exception as e:
            self.logger.error(f"Failed to apply defense {defense_name}: {e}")
            raise

    def _cleanup_defense_instance(self, defense_instance):
        """Clean up defense instance (especially for defense methods that need to clean up temporary files)"""
        if defense_instance is None:
            return

        try:
            # Check if defense instance has cleanup method
            if hasattr(defense_instance, "cleanup"):
                defense_instance.cleanup()
                self.logger.debug(
                    f"Cleaned up defense instance: {defense_instance.__class__.__name__}"
                )
        except Exception as e:
            self.logger.warning(f"Failed to clean up defense instance: {e}")

    @log_with_context("Generate single model response")
    def generate_single_response(
        self, test_case: TestCase, model_name: str, defense_name: str, _done_ids: set = None
    ) -> ModelResponse:
        """Generate response for a single test case, supports post-processing defense"""
        # Skip already completed test cases
        if _done_ids and str(test_case.test_case_id) in _done_ids:
            return None
        defense_instance = None
        try:
            # Apply defense, get defense instance
            defended_test_case, defense_instance = self.apply_defense(
                test_case, defense_name, model_name
            )

            # Check if defense method has already generated response
            if BaseDefense.META_KEY_GENERATED_RESP in defended_test_case.metadata:
                defense_response = defended_test_case.metadata[
                    BaseDefense.META_KEY_GENERATED_RESP
                ]
                self.logger.info(
                    f"Defense method has already generated response, using directly: {test_case.test_case_id}"
                )
                # Create metadata, including all relevant fields
                metadata = {
                    **defended_test_case.metadata,
                    BaseDefense.META_KEY_GENERATED_RESP_USED: True,
                }

                response = ModelResponse(
                    test_case_id=test_case.test_case_id,
                    model_response=defense_response,
                    model_name=model_name,
                    metadata=metadata,
                )

                # Clean up defense instance
                self._cleanup_defense_instance(defense_instance)
                return response

            # Check if should return default response (e.g., Llama-Guard-4 blocking case)
            if defended_test_case.metadata.get(BaseDefense.META_KEY_SHOULD_BLOCK, False):
                default_response = defended_test_case.metadata.get(
                    BaseDefense.META_KEY_DEFAULT_RESP, "I'm sorry, I can't assist with that."
                )

                self.logger.info(
                    f"Defense method blocked input, returning default response: {test_case.test_case_id}"
                )
                # Create metadata, including all relevant fields
                metadata = {
                    **defended_test_case.metadata,
                    BaseDefense.META_KEY_BLOCKED: True,
                }

                response = ModelResponse(
                    test_case_id=test_case.test_case_id,
                    model_response=default_response,
                    model_name=model_name,
                    metadata=metadata,
                )

                # Clean up defense instance
                self._cleanup_defense_instance(defense_instance)
                return response

            # Create model instance
            model_config = self.response_configs.get("model_params", {}).get(
                model_name, {}
            )
            model = UNIFIED_REGISTRY.create_model(model_name, model_config)

            # Generate original response
            model_response = model.generate_response(defended_test_case)

            # Apply post-processing defense (if supported)
            if defense_instance and hasattr(defense_instance, "post_process_response"):
                try:
                    original_response = model_response.model_response
                    processed_response, postprocessing_metadata = (
                        defense_instance.post_process_response(
                            original_response=original_response,
                            test_case=test_case,
                            model=model,
                        )
                    )

                    # Update response and metadata
                    model_response.model_response = processed_response

                    self.logger.debug(
                        f"Applied post-processing defense {defense_name} to test case {test_case.test_case_id}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Post-processing defense {defense_name} failed: {e}"
                    )
                    model_response.metadata["postprocessing_error"] = str(e)

            self.logger.debug(
                f"Successfully generated response for test case {test_case.test_case_id}"
            )

            # Clean up defense instance
            self._cleanup_defense_instance(defense_instance)
            return model_response

        except Exception as e:
            # Use logger.exception to record complete stack trace
            self.logger.exception(
                f"Failed to generate response for test case {test_case.test_case_id}"
            )
            # Also clean up defense instance on exception
            self._cleanup_defense_instance(defense_instance)
            raise e

    @log_with_context("Batch generate model responses")
    def generate_responses_batch(
        self,
        test_cases: List[TestCase],
        model_name: str,
        defense_name: str,
        max_workers_override: int | None = None,
    ) -> List[ModelResponse]:
        """Batch generate responses for multiple test cases, suitable for locally loaded models or defenses that need to load models"""
        if not test_cases:
            return []

        defense_instance = None
        try:
            # Check if defense needs to load model
            defense_config = self.response_configs.get("defense_params", {}).get(
                defense_name, {}
            )
            defense_load_model = defense_config.get("load_model", False)

            # If defense needs to load model, reuse the same defense instance
            if defense_load_model and defense_name != "None" and defense_name:
                # Create defense instance (only once)
                defense_config["output_dir"] = self.output_dir / defense_name
                defense_config["target_model_name"] = model_name
                defense_instance = UNIFIED_REGISTRY.create_defense(
                    defense_name, defense_config
                )
                self.logger.info(
                    f"Created instance for defense {defense_name}, will batch apply to {len(test_cases)} test cases"
                )

            # Apply defense to all test cases
            defended_test_cases = []
            for test_case in test_cases:
                if defense_instance is not None:
                    # Reuse defense instance
                    defended_test_case = defense_instance.apply_defense(test_case)
                else:
                    # Create new defense instance for each test case (when defense doesn't need to load model)
                    defended_test_case, _ = self.apply_defense(
                        test_case, defense_name, model_name
                    )
                defended_test_cases.append(defended_test_case)

            # Check if any defense method has already generated response
            responses = []
            remaining_test_cases = []

            for defended_test_case in defended_test_cases:
                if BaseDefense.META_KEY_GENERATED_RESP in defended_test_case.metadata:
                    # Defense method has already generated response
                    defense_response = defended_test_case.metadata[
                        BaseDefense.META_KEY_GENERATED_RESP
                    ]
                    metadata = {
                        **defended_test_case.metadata,
                        BaseDefense.META_KEY_GENERATED_RESP_USED: True,
                    }
                    response = ModelResponse(
                        test_case_id=defended_test_case.test_case_id,
                        model_response=defense_response,
                        model_name=model_name,
                        metadata=metadata,
                    )
                    responses.append(response)
                elif defended_test_case.metadata.get(BaseDefense.META_KEY_SHOULD_BLOCK, False):
                    # Defense method blocked input
                    default_response = defended_test_case.metadata.get(
                        BaseDefense.META_KEY_DEFAULT_RESP, "I'm sorry, I can't assist with that."
                    )
                    metadata = {
                        **defended_test_case.metadata,
                        BaseDefense.META_KEY_BLOCKED: True,
                    }
                    response = ModelResponse(
                        test_case_id=defended_test_case.test_case_id,
                        model_response=default_response,
                        model_name=model_name,
                        metadata=metadata,
                    )
                    responses.append(response)
                else:
                    # Need model inference
                    remaining_test_cases.append(defended_test_case)

            if not remaining_test_cases:
                # All test cases have been processed by defense method
                self._cleanup_defense_instance(defense_instance)
                return responses

            # Create model instance
            model_config = self.response_configs.get("model_params", {}).get(
                model_name, {}
            )
            model = UNIFIED_REGISTRY.create_model(model_name, model_config)

            # Batch generate responses
            if model.model_type == "local":
                # Local models use batch inference
                batch_responses = model.generate_responses_batch(remaining_test_cases)
                responses.extend(batch_responses)
            else:
                # API models use parallel processing (utilizing multi-threading)
                from concurrent.futures import ThreadPoolExecutor, as_completed

                max_workers = (
                    max_workers_override
                    if max_workers_override is not None
                    else self.config.max_workers
                )
                self.logger.debug(
                    f"API models use parallel processing, worker threads: {max_workers}, test cases: {len(remaining_test_cases)}"
                )
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_case = {
                        executor.submit(model.generate_response, test_case): test_case
                        for test_case in remaining_test_cases
                    }
                    for future in as_completed(future_to_case):
                        test_case = future_to_case[future]
                        try:
                            response = future.result()
                            responses.append(response)
                        except Exception as e:
                            self.logger.error(
                                f"Failed to generate response (test case: {test_case.test_case_id}): {e}"
                            )

            # Apply post-processing defense (if supported)
            if defense_instance and hasattr(defense_instance, "post_process_response"):
                for response in responses:
                    try:
                        original_response = response.model_response
                        processed_response, postprocessing_metadata = (
                            defense_instance.post_process_response(
                                original_response=original_response,
                                test_case=test_case,
                                model=model,
                            )
                        )
                        response.model_response = processed_response
                    except Exception as e:
                        self.logger.warning(
                            f"Post-processing defense {defense_name} failed: {e}"
                        )
                        response.metadata["postprocessing_error"] = str(e)

            self.logger.info(f"Successfully batch generated {len(responses)} responses")

            # Clean up defense instance
            self._cleanup_defense_instance(defense_instance)
            return responses

        except Exception as e:
            self.logger.exception(f"Batch response generation failed: {e}")
            self._cleanup_defense_instance(defense_instance)
            raise e

    def _generate_responses_local_model_batched(
        self,
        combo_tasks: List[Tuple[TestCase, str, str, str]],
        model_name: str,
        defense_name: str,
        combo_filename: Path,
        batch_size: int,
    ) -> List[ModelResponse]:
        """Unified resource strategy for local models:
        - create the model once
        - run single-worker
        - process test cases in batches
        - save incrementally via BatchSaveManager
        """
        from .base_pipeline import batch_save_context

        # Create (and reuse) defense instance (single worker => safe)
        defense_instance = None
        defense_config = self.response_configs.get("defense_params", {}).get(
            defense_name, {}
        )
        if defense_name != "None" and defense_name:
            defense_config = dict(defense_config)
            defense_config["output_dir"] = self.output_dir / defense_name
            defense_config["target_model_name"] = model_name
            defense_instance = UNIFIED_REGISTRY.create_defense(defense_name, defense_config)

        # Create (and reuse) local model instance
        model_config = self.response_configs.get("model_params", {}).get(model_name, {})
        model = UNIFIED_REGISTRY.create_model(model_name, model_config)
        # 통합 모드: 파이프라인이 만든 타깃 모델을 가드가 재사용하도록 주입.
        # (가드가 자기 타깃을 새로 만들면 GPU0 에 Qwen 이 두 개 올라가 OOM)
        if defense_instance is not None and hasattr(type(defense_instance), "_shared_target_model"):
            type(defense_instance)._shared_target_model = model

        all_responses: List[ModelResponse] = []
        test_cases_only = [t[0] for t in combo_tasks]

        with batch_save_context(
            pipeline=self,
            output_filename=combo_filename,
            batch_size=batch_size,
            total_items=len(test_cases_only),
            desc=f"Generate responses (local model, {model_name}, {defense_name})",
        ) as save_manager:
            for i in range(0, len(test_cases_only), batch_size):
                batch_cases = test_cases_only[i : i + batch_size]

                defended_cases: List[TestCase] = []
                for tc in batch_cases:
                    if defense_instance is None:
                        defended_cases.append(tc)
                    else:
                        defended_cases.append(defense_instance.apply_defense(tc))

                # Handle defense-direct responses / blocked inputs
                ready_responses: List[ModelResponse] = []
                remaining_cases: List[TestCase] = []

                for defended_tc in defended_cases:
                    if BaseDefense.META_KEY_GENERATED_RESP in (defended_tc.metadata or {}):
                        text = defended_tc.metadata[BaseDefense.META_KEY_GENERATED_RESP]
                        meta = {**(defended_tc.metadata or {}), BaseDefense.META_KEY_GENERATED_RESP_USED: True}
                        ready_responses.append(
                            ModelResponse(
                                test_case_id=defended_tc.test_case_id,
                                model_response=text,
                                model_name=model_name,
                                metadata=meta,
                            )
                        )
                    elif (defended_tc.metadata or {}).get(BaseDefense.META_KEY_SHOULD_BLOCK, False):
                        default_resp = (defended_tc.metadata or {}).get(
                            BaseDefense.META_KEY_DEFAULT_RESP, "I'm sorry, I can't assist with that."
                        )
                        meta = {**(defended_tc.metadata or {}), BaseDefense.META_KEY_BLOCKED: True}
                        ready_responses.append(
                            ModelResponse(
                                test_case_id=defended_tc.test_case_id,
                                model_response=default_resp,
                                model_name=model_name,
                                metadata=meta,
                            )
                        )
                    else:
                        remaining_cases.append(defended_tc)

                # Local model batch inference (single worker)
                if remaining_cases:
                    inferred = model.generate_responses_batch(remaining_cases)
                    ready_responses.extend(inferred)

                all_responses.extend(ready_responses)
                save_manager.add_results([r.to_dict() for r in ready_responses])

        # Cleanup
        self._cleanup_defense_instance(defense_instance)
        return all_responses

    def run(self, **kwargs) -> List[ModelResponse]:
        """Run response generation, supports checkpoint resume and real-time batch saving"""
        if not self.validate_config():
            return []

        # Get batch size parameter (priority: kwargs parameter, then configuration parameter)
        batch_size = kwargs.get("batch_size", self.config.batch_size)
        self.logger.info(
            f"Starting model response generation stage (batch size: {batch_size})"
        )

        # Get attack method list
        attack_names = self.config.test_case_generation.get("attacks", [])

        # Load test cases
        test_cases = self.load_test_cases(attack_names=attack_names)
        if not test_cases:
            self.logger.error("No available test cases")
            return []

        # Get model and defense configuration
        model_names = self.response_configs.get("models", [])
        defense_names = self.response_configs.get("defenses", ["None"])

        if not model_names:
            self.logger.error("Models not specified")
            return []

        self.logger.info(
            f"Will generate responses for {len(test_cases)} test cases using {len(model_names)} models and {len(defense_names)} defense methods"
        )

        # Generate all tasks
        pending_tasks = []

        for test_case in test_cases:
            for model_name in model_names:
                for defense_name in defense_names:
                    # Generate task ID
                    task_config = {
                        "test_case_id": test_case.test_case_id,
                        "model_name": model_name,
                        "defense_name": defense_name,
                        "model_params": self.response_configs.get(
                            "model_params", {}
                        ).get(model_name, {}),
                        "defense_params": self.response_configs.get(
                            "defense_params", {}
                        ).get(defense_name, {}),
                    }
                    task_id = f"{test_case.test_case_id}_{model_name}_{defense_name}_{self.get_task_hash(task_config)}"
                    pending_tasks.append((test_case, model_name, defense_name, task_id))

        pending_count = len(pending_tasks)
        self.logger.info(f"Total tasks: {pending_count}")

        # Group tasks by attack method, model and defense method
        tasks_by_combo = {}
        for test_case, model_name, defense_name, task_id in pending_tasks:
            attack_name = test_case.metadata.get("attack_method", "")
            key = (attack_name, model_name, defense_name)
            if key not in tasks_by_combo:
                tasks_by_combo[key] = []
            tasks_by_combo[key].append((test_case, model_name, defense_name, task_id))

        # Check if each combination has generated complete responses
        completed_combos = []
        pending_combos_to_process = []

        for (
            attack_name,
            model_name,
            defense_name,
        ), combo_tasks in tasks_by_combo.items():
            # Generate filename for this combination
            _, combo_filename = self._generate_filename(
                "response_generation",
                attack_name=attack_name,
                model_name=model_name,
                defense_name=defense_name,
            )

            # Calculate expected response count for this combination (count test cases for this attack method)
            expected_count = 0
            for test_case in test_cases:
                if test_case.metadata.get("attack_method", "") == attack_name:
                    expected_count += 1

            # Check existing response files
            existing_responses = self.load_results(combo_filename)

            if len(existing_responses) >= expected_count:
                self.logger.info(
                    f"Combination {attack_name}+{model_name}+{defense_name} has complete responses: {len(existing_responses)}/{expected_count}"
                )
                completed_combos.append(
                    (attack_name, model_name, defense_name, combo_filename, combo_tasks)
                )
            else:
                self.logger.info(
                    f"Combination {attack_name}+{model_name}+{defense_name} needs to generate responses: {len(existing_responses)}/{expected_count}"
                )
                pending_combos_to_process.append(
                    (
                        attack_name,
                        model_name,
                        defense_name,
                        combo_filename,
                        combo_tasks,
                        expected_count,
                    )
                )

        # If all combinations are completed, directly load existing results
        if not pending_combos_to_process:
            self.logger.info("All combinations completed, loading existing results")
            all_responses = self._load_all_responses(model_names, defense_names)
            self.logger.info(f"Total loaded {len(all_responses)} responses")
            return all_responses

        self.logger.info(
            f"Need to process {len(pending_combos_to_process)} combinations"
        )

        all_responses = []

        # First load responses from completed combinations
        for (
            attack_name,
            model_name,
            defense_name,
            combo_filename,
            combo_tasks,
        ) in completed_combos:
            existing_results = self.load_results(combo_filename)
            for item in existing_results:
                try:
                    response = ModelResponse.from_dict(item)
                    all_responses.append(response)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to parse response ({attack_name}, {model_name}, {defense_name}): {e}"
                    )
            self.logger.info(
                f"Loaded {len(existing_results)} responses from {combo_filename}"
            )

        # Generate responses for each combination that needs processing
        for (
            attack_name,
            model_name,
            defense_name,
            combo_filename,
            combo_tasks,
            expected_count,
        ) in pending_combos_to_process:
            self.logger.info(
                f"Processing combination: attack={attack_name}, model={model_name}, defense={defense_name}, tasks={len(combo_tasks)}"
            )

            # Determine unified resource policy (local models => single worker + batched)
            defense_config = self.response_configs.get("defense_params", {}).get(
                defense_name, {}
            )
            model_config = self.response_configs.get("model_params", {}).get(model_name, {})
            policy = policy_for_response_generation(
                model_config, defense_config, default_max_workers=self.config.max_workers
            )

            # Single source of truth: follow policy only
            if policy.strategy == "batched" and policy.batched_impl == "local_model":
                local_responses = self._generate_responses_local_model_batched(
                    combo_tasks=combo_tasks,
                    model_name=model_name,
                    defense_name=defense_name,
                    combo_filename=combo_filename,
                    batch_size=batch_size,
                )
                all_responses.extend(local_responses)
                self.logger.info(
                    f"Combination completed (local policy): attack={attack_name}, model={model_name}, defense={defense_name}, generated {len(local_responses)} responses"
                )
            elif policy.strategy == "batched":
                # Batched policy triggered by defense.load_model (or other future flags)
                test_cases = [task[0] for task in combo_tasks]
                batch_responses = self.generate_responses_batch(
                    test_cases,
                    model_name,
                    defense_name,
                    max_workers_override=policy.max_workers,
                )
                self.save_results_incrementally([r.to_dict() for r in batch_responses], combo_filename)
                all_responses.extend(batch_responses)
                self.logger.info(
                    f"Combination completed (batched defense): attack={attack_name}, model={model_name}, defense={defense_name}, generated {len(batch_responses)} responses"
                )
            else:
                # API model + stateless defense => use multi-threaded parallel processing
                self.logger.info(
                    f"Defense {defense_name} doesn't need to load model and model {model_name} is API model, using multi-threaded parallel processing"
                )

                # Load already completed IDs to skip
                existing_responses = self.load_results(combo_filename)
                done_ids = {str(r.get("test_case_id")) for r in existing_responses}
                self.logger.info(f"Skipping {len(done_ids)} already completed responses")
                # Prepare processing function
                def process_task(task_item):
                    test_case, model_name, defense_name, task_id = task_item
                    try:
                        response = self.generate_single_response(
                            test_case, model_name, defense_name, _done_ids=done_ids
                        )
                        if response is None:
                            return None
                        response_dict = response.to_dict()
                        return response_dict
                    except Exception as e:
                        self.logger.error(
                            f"Task failed ({test_case.test_case_id}, {model_name}, {defense_name}): {e}"
                        )
                        return None

                # For API models and defenses that don't need to load models, use parallel strategy
                results_dicts = process_with_strategy(
                    items=combo_tasks,
                    process_func=process_task,
                    pipeline=self,
                    output_filename=combo_filename,
                    batch_size=batch_size,
                    max_workers=policy.max_workers,
                    strategy_name="parallel",  # API models use parallel strategy
                    desc=f"Generate responses ({attack_name}, {model_name}, {defense_name})",
                )

                # Load results for this combination
                combo_results = self.load_results(combo_filename)
                combo_responses = []
                for item in combo_results:
                    try:
                        response = ModelResponse.from_dict(item)
                        combo_responses.append(response)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse model response: {e}")

                all_responses.extend(combo_responses)
                self.logger.info(
                    f"Combination completed: attack={attack_name}, model={model_name}, defense={defense_name}, generated {len(combo_responses)} responses"
                )

        if all_responses:
            self.logger.info(
                f"Response generation completed, generated {len(all_responses)} responses in total"
            )

        else:
            self.logger.warning("No responses generated")

        return all_responses

    def _load_all_responses(
        self, model_names: List[str], defense_names: List[str]
    ) -> List[ModelResponse]:
        """Load results for all model+defense method combinations"""
        all_responses = []

        # Get attack method list
        attack_names = self.config.test_case_generation.get("attacks", [])

        for attack_name in attack_names:
            for model_name in model_names:
                for defense_name in defense_names:
                    # Generate filename for this combination
                    try:
                        _, combo_filename = self._generate_filename(
                            "response_generation",
                            attack_name=attack_name,
                            model_name=model_name,
                            defense_name=defense_name,
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to generate filename (attack={attack_name}, model={model_name}, defense={defense_name}): {e}"
                        )
                        continue

                    # Load results for this combination
                    combo_results = self.load_results(combo_filename)
                    for item in combo_results:
                        try:
                            response = ModelResponse.from_dict(item)
                            all_responses.append(response)
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to parse model response (attack={attack_name}, model={model_name}, defense={defense_name}): {e}"
                            )

                    if combo_results:
                        self.logger.debug(
                            f"Loaded {len(combo_results)} responses from {combo_filename}"
                        )

        self.logger.info(f"Total loaded {len(all_responses)} responses")
        return all_responses

    def validate_config(self) -> bool:
        """Validate configuration"""
        if not super().validate_config():
            return False

        if not self.response_configs.get("models"):
            self.logger.error("Models not specified")
            return False

        return True
