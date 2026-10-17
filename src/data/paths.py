"""Raw dataset locations, shared by the manifest builders and the data loaders.

Manifests and split files store image paths relative to these dataset roots,
so the same files work on Windows and Linux; only the roots below have to
point at the local copies.
"""
from pathlib import Path

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_APTOS = Path("F:/Datasets/DR/APTOS")
_LINUX_DEEPDRID = Path("/home/user/Documents/Mohammadi/DeepDRiD/DeepDRiD-master")
DEFAULT_DEEPDRID = _LINUX_DEEPDRID if _LINUX_DEEPDRID.exists() else Path("F:/Datasets/DR/DeepDr/DeepDRiD-master")
DEFAULT_MMRDR = Path("F:/Datasets/DR/MMRDR")


def resolve_image_root(manifest_path: str, image_root: str | None = None) -> Path | None:
    """Root that relative manifest paths are joined to.

    An explicit `image_root` (config `data.image_root`) wins; otherwise the
    dataset is recognised from the manifest path (data/<dataset>/...).
    """
    if image_root:
        return Path(image_root)
    manifest = str(manifest_path).lower()
    for name, root in (("deepdrid", DEFAULT_DEEPDRID), ("aptos", DEFAULT_APTOS), ("mmrdr", DEFAULT_MMRDR)):
        if name in manifest:
            return root
    return None
