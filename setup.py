from setuptools import find_packages, setup

setup(
    name="encodings_hnns",
    version="1.0.0",
    python_requires=">3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "pandas", "matplotlib", "scipy==1.10.0"],
    extra_requires={
        "dev": [
            "mypy",
            "pandas-stubs",
            "type-python-dateutil",
            "pydocstyle",
            "black",
            "isort",
            "black",
            "isort",
            "docformatter",
        ],
        "test": ["pytest", "coverage", "pytest-cov", "pytest-random-order"],
    },
)
