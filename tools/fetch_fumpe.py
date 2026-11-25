"""Download selected CC BY 4.0 FUMPE cases and their shared reference archive."""

import argparse
import hashlib
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def fetch(root, name, article, sevenzip):
    f = requests.get(
        f"https://api.figshare.com/v2/articles/{article}", timeout=45
    ).json()["files"][0]
    root.mkdir(parents=True, exist_ok=True)
    rar = root / (name + ".rar")
    dest = root / name
    if not rar.exists():
        tmp = rar.with_suffix(".part")
        with requests.get(f["download_url"], stream=True, timeout=120) as reply:
            reply.raise_for_status()
            with tmp.open("wb") as out:
                shutil.copyfileobj(reply.raw, out)
        os.replace(tmp, rar)
    with rar.open("rb") as infile:
        digest = hashlib.file_digest(infile, "md5").hexdigest()
    if digest != f["computed_md5"]:
        raise ValueError("Archive checksum mismatch: " + name)
    dest.mkdir(exist_ok=True)
    subprocess.run(
        [sevenzip, "x", "-y", "-o" + str(dest), str(rar)],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    print(name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("numbers", type=int, nargs="+")
    ap.add_argument("--sevenzip", default="7zz")
    a = ap.parse_args()
    articles = requests.get(
        "https://api.figshare.com/v2/collections/4107803/articles?page_size=100",
        timeout=45,
    ).json()
    by_name = {v["title"]: v["id"] for v in articles}
    jobs = [("Reference", by_name["Ground Truth"])] + [
        (f"PAT{i:03}", by_name[f"Patient{i:02}"]) for i in a.numbers
    ]
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(lambda job: fetch(a.root, *job, a.sevenzip), jobs))
