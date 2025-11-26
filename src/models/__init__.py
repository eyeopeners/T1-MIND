# --------------------------------------------------------------------------------
# Copyright (c) 2024 Gabriele Lozupone (University of Cassino and Southern Lazio).
# All rights reserved.
# --------------------------------------------------------------------------------
# 
# LICENSE NOTICE
# *************************************************************************************************************
# By downloading/using/running/editing/changing any portion of codes in this package you agree to the license.
# If you do not agree to this license, do not download/use/run/edit/change this code.
# Refer to the LICENSE file in the root directory of this repository for full details.
# *************************************************************************************************************
# 
# Contact: Gabriele Lozupone at gabriele.lozupone@unicas.it
# -----------------------------------------------------------------------------
# Lightweight package init to avoid importing optional backbones.
# Only expose what we actually need for the PPAL_MIND training path.

from importlib import import_module

# Try to expose PPAL_MIND; do not import any other heavy/optional modules here.
try:
    # Note: importing a submodule requires this package __init__ to be import-safe.
    ppal_mind = import_module(".ppal_mind", __name__)
    PPAL_MIND = ppal_mind.PPAL_MIND
except Exception as e:
    PPAL_MIND = None  # will raise at use-time if someone needs it

__all__ = ["PPAL_MIND"]

