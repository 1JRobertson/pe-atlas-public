"""Install a verified example archive in PE Atlas's study folder."""

import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile


CASE_FILES = {
    "artery-ras.vtp",
    "automatic.nii.gz",
    "candidates.json",
    "ct.nii.gz",
    "source.json",
    "vessels/lung_arteries.nii.gz",
}
DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
DEMOS = {
    "PAT003": ("fumpe-pat003.sha256", {"ATTRIBUTION.md", "reference.nii.gz"}),
    "SupplementalCTPA": ("ctpa-example.sha256", {"ABOUT.md"}),
}


def install(archive_path, home):
    cases = home / "Cases"
    with archive_path.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        checksum = digest.hexdigest()
        for name, (checksum_file, extra_files) in DEMOS.items():
            if checksum == (DEMO_DIR / checksum_file).read_text().strip():
                break
        else:
            raise ValueError("Demo archive checksum mismatch")

        dest = cases / name
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(f"{dest} already exists. Existing studies are kept in place.")

        expected = CASE_FILES | extra_files
        source.seek(0)
        with ZipFile(source) as archive:
            names = archive.namelist()
            evidence = set(names) - expected
            if (
                len(names) != len(expected) + 1
                or not expected.issubset(names)
                or len(evidence) != 1
                or not re.fullmatch(r"evidence-[a-f0-9]{20}\.nii\.gz", next(iter(evidence)))
            ):
                raise ValueError("Unexpected files in the demo archive")

            cases.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".example-", dir=cases) as staging:
                study = Path(staging) / dest.name
                for name in names:
                    target = study / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as member, target.open("xb") as output:
                        shutil.copyfileobj(member, output)
                study.rename(dest)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    home = Path(
        os.environ.get("PE_ATLAS_HOME", Path.home() / "Library/Application Support/PE Atlas Public")
    )
    try:
        print(install(args.archive, home))
    except (OSError, ValueError, BadZipFile) as error:
        parser.exit(1, f"Could not install the demo. {error}\n")
