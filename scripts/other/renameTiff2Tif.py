from pathlib import Path
import shutil

in_dir = Path("./data/images/PSR_2ndbatch")
out_dir = Path("./data/images/PSR_2ndbatch2")

in_files = [f for f in in_dir.iterdir() if f.is_file()]
for in_file in in_files:
    out_file = out_dir / (in_file.stem + '.tif')
    in_file = str(in_file)
    if not out_file.is_file():
        shutil.move(in_file, str(out_file))
        print(out_file)