#!/usr/bin/env python3
"""Stable V21-named entry point for the PV6 packet simulation utility."""

import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).with_name("v20_single_score_2phase_sim.py")),
               run_name="__main__")
