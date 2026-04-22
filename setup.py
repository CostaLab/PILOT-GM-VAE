from setuptools import setup, find_packages

setup(
    name="pilotgm",
    version="0.1.1", 
    author="Mehdi Joodaki",
    author_email="judakimehdi@gmail.com",
    url="https://github.com/CostaLab/PILOT-GM-VAE",
    description="Patient-Level Analysis of Single Cell Disease Atlas with Optimal Transport of Gaussian Mixtures Variational Autoencoders",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",

    python_requires=">=3.9",

    package_dir={"": "src"},
    packages=find_packages(where="src"),

    install_requires=[
        "torch",
        "pilotpy",
        "scanpy",
        "joblib",
        "tqdm",
        "numba",
    ],
)
