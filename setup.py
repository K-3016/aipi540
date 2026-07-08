# Portions of this file were developed with assistance from OpenAI ChatGPT/Codex and reviewed/modified by the author.
"""Package setup for Campus Support Message Triage Assistant."""

from setuptools import find_packages, setup


setup(
    name="campus-support-message-triage-assistant",
    version="0.1.0",
    description="NLP assistant for routing synthetic campus support messages.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
