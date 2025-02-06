from setuptools import find_packages, setup

setup(
    name="encodings_hnns",
    version="1.0.0",
    python_requires=">3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "shapely",
        "tabulate",
        "path",
        "brec",
        "imageio",
        "imageio-ffmpeg",
        "POT",  # Python Optimal Transport library
        # PyTorch ecosystem
        "torch",  # pytorch
        "torch-geometric",  # pyg
        "torch-scatter",
        "torch-sparse",
        "pytorch-lightning",  # Optional but often used with PyG
    ],
    extras_require={
        "dev": [
            "mypy",
            "pandas-stubs",
            "type-python-dateutil",
            "pydocstyle",
            "black",
            "isort",
            "docformatter",
            "ruff",
            "pylint",
        ],
        "test": [
            "pytest",
            "coverage",
            "pytest-cov",
            "pytest-random-order",
        ],
    },
)
