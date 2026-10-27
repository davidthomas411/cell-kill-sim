"""Physical and optical constants for light-sheet microscopy datasets."""

from __future__ import annotations

# Lateral voxel pitch of the deposited 256x256 ROI grid,
# from feilab summary.m sqrt(0.0385 um^2)
VOXEL_UM: float = 0.1962

# Optical lateral resolution FWHM (um), Wang et al. Nat Commun 2024 / Sci Data 2026
OPTICAL_FWHM_UM: float = 0.32
