#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from core.data_formats import TestCase
from core.base_classes import BaseAttack


@dataclass
class SiuoConfig:
    """New attack configuration - no transformation parameters needed"""

    pass


class SiuoAttack(BaseAttack):
    """
    Passthrough attack for the ELITE benchmark.

    The ELITE dataset already ships pre-built jailbreak image/prompt pairs
    (mixing SPA-VL, FigStep, MM-SafetyBench, JailbreakV-28k, VLGuard, ...),
    so this attack performs no transformation: the existing image and prompt
    are used directly as the jailbreak inputs.
    """

    CONFIG_CLASS = SiuoConfig

    def generate_test_case(
        self,
        original_prompt: str,
        image_path: str,
        case_id: str,
        **kwargs,
    ) -> TestCase:
        return self.create_test_case(
            case_id=case_id,
            jailbreak_prompt=original_prompt,
            jailbreak_image_path=str(image_path),
            original_prompt=original_prompt,
            original_image_path=str(image_path),
        )
