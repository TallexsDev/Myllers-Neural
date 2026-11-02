import os
import shutil
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    app_dir = Path(sys.executable).resolve().parent
    resource_dir = Path(sys._MEIPASS)
    data_dir = app_dir / "data"

    data_dir.mkdir(parents=True, exist_ok=True)

    source = resource_dir / "data" / "knowledge.json"
    destination = data_dir / "knowledge.json"

    if source.exists() and not destination.exists():
        shutil.copy2(source, destination)

    os.chdir(app_dir)
