from setuptools import setup, find_packages

setup(
    name="PILOT-GM-VAE",
    version="0.1",
    packages=find_packages(),
    author="Mehdi Joodaki, Mina Shaigan",
    author_email="judakimehdi@gmail.com, mina.shaigan@gmail.com",
    description="Patient-Level Analysis of Single Cell Disease Atlas with Optimal Transport of Gaussian Mixtures (PILOT_VAE)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/minashaigan/PILOT_VAE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="==3.14.3",
    install_requires=[
        "torch",
        "pilotpy",
        "scanpy",
        "joblib",
        "tqdm",
        "numba",
    ],
)
