from setuptools import setup, find_packages

setup(
    name="costwindow",
    version="0.1.1",
    description="AI API cost monitoring SDK — CostWindow",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["requests>=2.28"],
    extras_require={
        "anthropic": ["anthropic>=0.39"],
        "openai":    ["openai>=1.0"],
        "all":       ["anthropic>=0.39", "openai>=1.0"],
    },
)
