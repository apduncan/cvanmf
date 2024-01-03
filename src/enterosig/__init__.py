"""Enterosignatures blurb

etc etc
"""
# read version from installed package
from importlib.metadata import version
__version__ = version("enterosig")

from enterosig.transform import transform