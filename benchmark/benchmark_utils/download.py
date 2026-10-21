"""Fetch release files from a Zenodo record.

A restricted record (or an unpublished draft) is read with the access token
held in the ``token_env`` environment variable, when it is set.
"""

import os
import shutil
import urllib.request


def download_zenodo(record, names, dest, token_env=None):
    """Download the files ``names`` of ``record`` missing from ``dest``."""
    token = os.environ.get(token_env) if token_env else None
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = dest / name
        if path.exists():
            continue
        url = (
            f"https://zenodo.org/records/{record}/files/{name}"
            "?download=1&preview=1"
        )
        if token:
            url += f"&token={token}"
        print(f"[download] {name} from Zenodo record {record}")
        part = path.with_name(path.name + ".part")
        with urllib.request.urlopen(url) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f)
        part.rename(path)
