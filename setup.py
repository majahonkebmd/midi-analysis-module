from pathlib import Path
from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def _read_requirements(path: Path) -> list[str]:
    reqs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        reqs.append(raw)
    return reqs


setup(
    name="midi-analysis-module",
    version="0.1.0",
    description="Educational MIDI analysis tool for piano pedagogy",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=_read_requirements(ROOT / "requirements.txt"),
    python_requires=">=3.10",
)
