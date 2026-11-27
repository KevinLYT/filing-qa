"""Fetch the two pinned ONNX models and verify their upstream file digests."""
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
MODELS = {
    "dense": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
              "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21",
              "cb202bfe2e3c98645018a6d12f182a434c9d3e02"),
    "reranker": ("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a",
                 "c80a8b34256ea453093d612e3ac48d3d965a0c0a48c7906709af8b8e28461bf9",
                 "688882a79f44442ddc1f60d70334a7ff5df0fb47"),
}


def checked(data, digest):
    # Hugging Face publishes LFS SHA-256 for weights and Git blob IDs for ordinary files.
    if len(digest) == 64:
        return hashlib.sha256(data).hexdigest() == digest
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest


def prepare():
    directory = ROOT / "models/retrieval"
    manifest = {}
    for role, (repo, revision, model_digest, tokenizer_digest) in MODELS.items():
        destination = directory / role
        destination.mkdir(parents=True, exist_ok=True)
        files = {}
        for local, upstream, digest in (("model.onnx", "onnx/model_quint8_avx2.onnx", model_digest),
                                        ("tokenizer.json", "tokenizer.json", tokenizer_digest)):
            path = destination / local
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{upstream}"
            if path.exists():
                data = path.read_bytes()
                if not checked(data, digest):
                    raise ValueError(f"Unexpected contents in existing file: {path}")
            else:
                with urlopen(url, timeout=120) as response:
                    data = response.read()
                if not checked(data, digest):
                    raise ValueError(f"Download checksum mismatch: {url}")
                with NamedTemporaryFile(dir=destination, delete=False) as temp:
                    temp.write(data)
                Path(temp.name).replace(path)
            files[local] = {"source": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            print(f"Verified {role}/{local} ({len(data):,} bytes)", flush=True)
        manifest[role] = {"repository": repo, "revision": revision, "files": files}
    with NamedTemporaryFile("w", dir=directory, encoding="utf-8", delete=False) as temp:
        json.dump(manifest, temp, indent=2)
        temp.write("\n")
    Path(temp.name).replace(directory / "manifest.json")


if __name__ == "__main__":
    prepare()
