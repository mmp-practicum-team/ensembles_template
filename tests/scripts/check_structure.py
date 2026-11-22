"""Check the files required by the assignment and the starter template."""

from pathlib import Path
import sys


REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "report.pdf",
    "ui.py",
    "ensembles/__init__.py",
    "ensembles/boosting.py",
    "ensembles/random_forest.py",
    "ensembles/utils.py",
    "ensembles/backend/__init__.py",
    "ensembles/backend/app.py",
    "ensembles/backend/schemas.py",
    "ensembles/frontend/__init__.py",
    "ensembles/frontend/client.py",
    "ensembles/frontend/utils.py",
)


def main() -> int:
    missing = [name for name in REQUIRED_FILES if not Path(name).is_file()]
    if missing:
        print("Не найдены обязательные файлы:")
        for name in missing:
            print(f"  - {name}")
        return 1
    print("OK: все обязательные файлы на месте.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
