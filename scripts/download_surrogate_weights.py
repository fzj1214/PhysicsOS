from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_AUTHOR = "polymathic-ai"
DEFAULT_FAMILIES = ("FNO", "TFNO", "UNetClassic", "UNetConvNext")
DEFAULT_REPOS = (
    "polymathic-ai/FNO-rayleigh_benard",
    "polymathic-ai/TFNO-helmholtz_staircase",
    "polymathic-ai/UNetConvNext-gray_scott_reaction_diffusion",
    "polymathic-ai/UNetClassic-acoustic_scattering_maze",
)
WALRUS_REPOS = (
    "polymathic-ai/walrus",
    "polymathic-ai/walrus_ft_bubbleML_poolBoil",
    "polymathic-ai/walrus_ft_CE-RM",
    "polymathic-ai/walrus_ft_pdearena_ins",
    "polymathic-ai/walrus_ft_post_neutron_star_merger",
    "polymathic-ai/walrus_ft_convective_envelope_rsg",
    "polymathic-ai/walrus_ft_CNS3D_128_Rand",
    "polymathic-ai/walrus_ft_CNS3D_64_Turb",
    "polymathic-ai/walrus_ft_flowbench_skelenton",
)


def family_from_repo(repo_id: str) -> str:
    name = repo_id.split("/")[-1]
    if name.startswith("FNO-"):
        return "grid_neural_operator"
    if name.startswith("TFNO-"):
        return "tensorized_neural_operator"
    if name.startswith(("UNetClassic-", "UNetConvNext-")):
        return "unet_surrogate"
    if name.startswith(("walrus", "aion")):
        return "foundation_surrogate"
    return "custom"


def model_id_from_repo(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace("-", "_").lower()


def domains_from_dataset(dataset: str) -> list[str]:
    name = dataset.lower()
    if any(token in name for token in ["rayleigh", "shear", "turbulence", "mhd", "radiative", "convective"]):
        return ["fluid", "thermal"]
    if any(token in name for token in ["reaction", "gray_scott"]):
        return ["custom", "thermal"]
    if any(token in name for token in ["acoustic", "helmholtz"]):
        return ["acoustic", "custom"]
    if any(token in name for token in ["supernova", "neutron", "planetswe"]):
        return ["fluid", "custom"]
    return ["custom"]


def operator_families_from_dataset(dataset: str) -> list[str]:
    name = dataset.lower()
    families = []
    if "navier" in name or any(token in name for token in ["rayleigh", "shear", "turbulence", "mhd"]):
        families.append("navier_stokes")
    if any(token in name for token in ["thermal", "convective", "radiative", "benard"]):
        families.append("heat")
    if any(token in name for token in ["acoustic", "helmholtz"]):
        families.append("helmholtz")
    if any(token in name for token in ["reaction", "gray_scott"]):
        families.append("reaction_diffusion")
    return families or ["parameterized_pde"]


def list_repos(families: tuple[str, ...], include_walrus: bool) -> list[str]:
    api = HfApi()
    repos = []
    models = list(api.list_models(author=REPO_AUTHOR, limit=300))
    for model in models:
        name = model.modelId.split("/")[-1]
        if any(name.startswith(f"{family}-") for family in families):
            repos.append(model.modelId)
    if include_walrus:
        repos.extend(WALRUS_REPOS)
    return sorted(dict.fromkeys(repos))


def download_repo(repo_id: str, output_root: Path) -> Path:
    local_dir = output_root / repo_id.replace("/", "__")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "README.md",
            "config.json",
            "model.safetensors",
            "*.safetensors",
            "*.pt",
            "*.pth",
        ],
    )
    return local_dir


def entry_for_repo(repo_id: str, local_dir: Path) -> dict:
    name = repo_id.split("/")[-1]
    dataset = name.split("-", 1)[1] if "-" in name else name
    files = list(local_dir.glob("*.safetensors")) + list(local_dir.glob("*.pt")) + list(local_dir.glob("*.pth"))
    checkpoint = files[0] if files else local_dir
    checkpoint_format = checkpoint.suffix.lstrip(".") if checkpoint.is_file() else "directory"
    return {
        "id": model_id_from_repo(repo_id),
        "name": name,
        "family": family_from_repo(repo_id),
        "domains": domains_from_dataset(dataset),
        "operator_families": operator_families_from_dataset(dataset),
        "geometry_encodings": ["multi_resolution_grid", "occupancy_mask"],
        "mesh_kinds": ["structured", "none"],
        "supports_transient": True,
        "checkpoint": {
            "uri": str(checkpoint.as_posix()),
            "kind": "model_checkpoint",
            "format": checkpoint_format,
            "description": f"Downloaded from https://huggingface.co/{repo_id}",
        },
        "runner": "safetensors" if checkpoint_format == "safetensors" else "torch_state_dict",
        "input_adapter": None,
        "output_adapter": None,
        "training_dataset": f"polymathic-ai/{dataset}",
        "expected_fields": ["input_tensor"],
        "output_fields": ["field"],
        "notes": [
            "Downloaded public checkpoint. Actual inference requires a model-specific input/output adapter.",
            f"Source repo: {repo_id}",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="models/huggingface")
    parser.add_argument("--registry", default="configs/surrogates.local.json")
    parser.add_argument("--families", nargs="*", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--repo", action="append", default=[], help="Explicit Hugging Face repo id to download. Can be repeated.")
    parser.add_argument("--all-polymathic", action="store_true", help="Download all matching Polymathic FNO/TFNO/UNet repos instead of the representative default set.")
    parser.add_argument("--include-walrus", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.repo:
        repos = args.repo
    elif args.all_polymathic:
        repos = list_repos(tuple(args.families), args.include_walrus)
    else:
        repos = list(DEFAULT_REPOS)
        if args.include_walrus:
            repos.extend(WALRUS_REPOS[:1])

    print(f"Downloading {len(repos)} repositories...")
    entries = []
    for index, repo_id in enumerate(repos, 1):
        print(f"[{index}/{len(repos)}] {repo_id}")
        local_dir = download_repo(repo_id, output_root)
        entries.append(entry_for_repo(repo_id, local_dir))

    registry_path = Path(args.registry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps({"models": entries}, indent=2), encoding="utf-8")
    print(f"Wrote {registry_path} with {len(entries)} models.")


if __name__ == "__main__":
    main()
