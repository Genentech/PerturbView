from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="perturbview",
    packages=find_packages(),
    install_requires=required,
    scripts=[
        "scripts/call_bases.py",
        "scripts/register_xenium.py",
        "scripts/stitch_images.py",
        "scripts/track_particles.py",
    ],
    package_data={
        "": ["*.sh"],
    },
    # set python versoion to 3.10 versions
    python_requires="~=3.10.0",
)
