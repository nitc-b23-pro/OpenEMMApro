from distutils.core import setup
from setuptools import find_packages

# This setup.py lives INSIDE the package it describes (policy_heads/setup.py,
# with policy_heads/__init__.py as its own sibling, models/ and util/ as its
# subpackages) rather than one directory above it. A plain find_packages()
# (relative to this file's own directory, i.e. policy_heads/ itself) can
# never discover a package literally named "policy_heads" -- only its
# subdirectories ("models", "util"), which `pip install -e ./policy_heads`
# would then install as bare, wrongly-unprefixed top-level packages, leaving
# nothing actually importable as `policy_heads` (or `policy_heads.models`,
# which is what build_model.py imports). package_dir maps the "policy_heads"
# package onto this directory itself so every subpackage below it still gets
# the correct "policy_heads." prefix.
setup(
    name='policy_heads',
    version='0.0.0',
    package_dir={'policy_heads': '.'},
    packages=['policy_heads'] + [f'policy_heads.{p}' for p in find_packages('.')],
    license='MIT License',
    long_description=open('README.md').read(),
)
