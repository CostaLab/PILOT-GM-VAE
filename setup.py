from setuptools import setup, find_packages

setup(
    name="pilotgm",
    version="0.1.0",
    author="Mehdi Joodaki",
    author_email="judakimehdi@gmail.com",
    url='https://github.com/CostaLab/PILOT-GM-VAE',
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
