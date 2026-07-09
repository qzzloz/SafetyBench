#!/usr/bin/env python3
"""
Jailbreak VLM Pipeline unified runner

Supports multiple running modes:
1. Full Pipeline: Run all three stages
2. Single stage: Run only specified stage
3. Combined stages: Run multiple specified stages
4. Create example configuration

Usage examples:
    # Run full Pipeline
    python run_pipeline.py --config config/general_config.yaml --full
    uv run python run_pipeline.py --config config/general_config.yaml --full

    # Run only test case generation
    python run_pipeline.py --config config/general_config.yaml --stage test_case_generation
    python run_pipeline.py --config config/guard_eval_config.yaml --stage test_case_generation

    # Generate response from specified test case file
    python run_pipeline.py --config config/general_config.yaml --stage response_generation --test-cases-file output/test_cases/jood/test_cases.json
    python run_pipeline.py --config config/guard_eval_config.yaml --stage response_generation --test-cases-file output/test_cases/jood/test_cases.json

    # Run multiple stages in combination
    python run_pipeline.py --config config/general_config.yaml --stages test_case_generation,response_generation, evaluation

    # Continue running from specified file
    python run_pipeline.py --config config/general_config.yaml --stage evaluation --input-file output/model_responses.json
"""

# python run_pipeline.py --config config/guard_eval_config.yaml --stage response_generation --test-cases-file output_qwen/test_cases/figstep/test_cases.jsonl 
# 이거 할 차례
import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    load_config,
    validate_config,
)
from pipeline.generate_test_cases import TestCaseGenerator
from pipeline.generate_responses import ResponseGenerator
from pipeline.evaluate_results import ResultEvaluator
from pipeline.run_full_pipeline import FullPipeline
from utils.logging_utils import setup_logging


class PipelineRunner:
    """Pipeline runner"""

    def __init__(self, config):
        self.config = config
        self.stage_outputs = {}

    def run_test_case_generation(self) -> str:
        """Run test case generation stage"""
        logging.info("🚀 Starting test case generation stage...")

        generator = TestCaseGenerator(self.config)
        test_cases = generator.run()

        return test_cases

    def run_response_generation(self, test_cases_file: Optional[str] = None) -> str:
        """Run model response generation stage"""
        logging.info("🚀 Starting model response generation stage...")

        generator = ResponseGenerator(self.config)

        # If test case file is provided, update configuration
        if test_cases_file:
            # Update test case file path in configuration
            generator.response_configs["input_test_cases"] = test_cases_file

        model_responses = generator.run()

        return model_responses

    def run_evaluation(self, model_responses_file: Optional[str] = None) -> str:
        """Run result evaluation stage"""
        logging.info("🚀 Starting result evaluation stage...")

        # If model response file is provided, update configuration
        if model_responses_file:
            # Ensure evaluation configuration exists
            if not hasattr(self.config, "evaluation"):
                self.config.evaluation = {}
            elif self.config.evaluation is None:
                self.config.evaluation = {}

            # Update input_responses configuration
            if isinstance(self.config.evaluation, dict):
                self.config.evaluation["input_responses"] = model_responses_file
            else:
                # If evaluation is other type, convert to dictionary
                eval_dict = (
                    self.config.evaluation.__dict__
                    if hasattr(self.config.evaluation, "__dict__")
                    else {}
                )
                eval_dict["input_responses"] = model_responses_file
                self.config.evaluation = eval_dict

        evaluator = ResultEvaluator(self.config)

        evaluation_results = evaluator.run()

        return evaluation_results

    def run_full_pipeline(self) -> dict:
        """Run full Pipeline"""
        logging.info("🚀 Starting full Pipeline run...")

        pipeline = FullPipeline(self.config)
        results = pipeline.run()

        logging.info("✅ Full Pipeline run completed")
        return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Jailbreak VLM Pipeline unified runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Running mode options
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--stage",
        type=str,
        help="Run single stage: test_case_generation, response_generation, evaluation",
    )
    mode_group.add_argument(
        "--stages",
        type=str,
        help="Run multiple stages in combination, separated by commas",
    )
    mode_group.add_argument("--full", action="store_true", help="Run full Pipeline")

    # General options
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument(
        "--input-file", type=str, help="Input file path (for continuing run)"
    )
    parser.add_argument(
        "--test-cases-file",
        type=str,
        help="Test case JSON file path (for response_generation stage)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides configuration setting)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose log output"
    )
    parser.add_argument("--log-file", type=str, help="Log file path")

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, log_file=args.log_file)

    logger = logging.getLogger(__name__)

    # Check configuration parameter
    if not args.config:
        logger.error("❌ Configuration file path must be provided (--config)")
        return 1

    try:
        # Load configuration
        logger.info(f"Loading configuration file: {args.config}")

        # Load configuration (load_config automatically extracts config directory from config file path)
        config = load_config(args.config)

        # Override output directory
        if args.output_dir:
            config.output_dir = args.output_dir

        # Validate configuration
        if not validate_config(config):
            logger.error("❌ Configuration validation failed")
            return 1

    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return 1

    # Create runner
    runner = PipelineRunner(config)

    try:
        # Execute based on running mode
        if args.stage:
            # Run single stage
            stage = args.stage
            if stage == "test_case_generation":
                runner.run_test_case_generation()
            elif stage == "response_generation":
                # Priority: --test-cases-file parameter, then --input-file parameter
                test_cases_file = args.test_cases_file or args.input_file
                runner.run_response_generation(test_cases_file)
            elif stage == "evaluation":
                runner.run_evaluation(args.input_file)
            else:
                logger.error(f"❌ Unknown stage: {stage}")
                return 1

        elif args.stages:
            # Run multiple stages in combination
            stages = [s.strip() for s in args.stages.split(",")]

            for stage in stages:
                if stage == "test_case_generation":
                    runner.run_test_case_generation()
                elif stage == "response_generation":
                    # Try to use previous stage output, or use --test-cases-file parameter
                    test_cases_file = args.test_cases_file or args.input_file
                    runner.run_response_generation(test_cases_file)
                elif stage == "evaluation":
                    # Try to use previous stage output
                    runner.run_evaluation(args.input_file)
                else:
                    logger.error(f"❌ Unknown stage: {stage}")
                    return 1

        elif args.full:
            # Run full Pipeline
            runner.run_full_pipeline()
        else:
            logger.error("❌ Please specify running mode: --stage, --stages or --full")
            return 1

        logger.info("\n🎉 Pipeline run completed!")
        return 0

    except Exception as e:
        logger.error(f"❌ Pipeline run failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
