import io
import os
import sys
from shutil import rmtree

from setuptools import Command, find_packages, setup

# Package meta-data.
NAME = "tabebm"
DESCRIPTION = 'Official implementation of "TabEBM: A Tabular Data Augmentation Method with Class-Specific Energy-Based Models", NeurIPS 2024.'
URL = "https://github.com/andreimargeloiu/TabEBM"
EMAIL = "silencejiang12138@gmail.com"
AUTHOR = "Andrei Margeloiu, Xiangjian Jiang, Nikola Simidjievski, Mateja Jamnik"
REQUIRES_PYTHON = ">=3.10.0"
VERSION = "2025.08.19"

REQUIRED = [
    "matplotlib",
    "matplotlib-inline",
    "numpy",
    "scipy",
    "tabpfn",
]

EXTRAS = {}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=("tests",)),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=["tabebm"],
    # entry_points={
    #     "console_scripts": ["mycli=mymodule:cli"],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="Apache Software License",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
)
