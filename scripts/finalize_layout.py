from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
source_root = root / "src" / "crowdsense"

packages = [
    ("constants", root / "constants"),
    ("modelling", root / "modelling"),
    ("preprocessing", root / "preprocessing"),
    ("views", root / "views"),
]

for package_name, legacy_dir in packages:
    dest_dir = source_root / package_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    for child in legacy_dir.iterdir():
        if child.is_file():
            shutil.copy2(child, dest_dir / child.name)

print("finalized src/crowdsense package files")
