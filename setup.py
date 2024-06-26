from setuptools import find_namespace_packages, setup

setup(
    name="encodings",
    version="1.0.0",
    python_requires=">3.11",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "pandas", "matplotlib"],
    extra_requires={
        "dev": ["mypy", "pandas-stubs", "black", "isort"],
        "test": ["pytest", "coverage"],
    },
)
