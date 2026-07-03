"""Modal app: GRAIL Stage-2b Set-GFlowNet **M2 headline** (3 seeds, full clean test).

Why Modal: the Stage-2b environment is a CPU-bound RDKit rule-application + tautomer
canonicalization loop (GPU sits ~idle -- confirmed on Colab M1). Modal gives a headless,
persistent container with a Volume for the deterministic caches, so the run finishes
unattended -- no Colab-session babysitting.

Key design -- the 3 seeds run SEQUENTIALLY in ONE container so the persistent
``(state, top_k) -> children`` and ``SMILES -> tautomer-InChIKey`` caches (saved to the
``grail-artifacts`` Volume after every epoch) are built ONCE by seed 0 and reused by
seeds 1/2. Only seed 0 pays the multi-hour cold pool-gen on the 1100 unseen substrates.

RDKit is pinned to 2024.9.x to match the Colab M1 environment (tautomer canonicalization
is RDKit-version-dependent, so the M2 numbers stay comparable to M1 and internally
consistent across seeds).

------------------------------------------------------------------------------------
ONE-TIME SETUP (local; needs a funded Modal account -- storage first, then compute):

    bash scripts/modal_stage.sh          # create volumes + upload data (~1.4G) + ckpts

RUN (fully unattended, detached; survives local disconnect):

    modal run --detach scripts/modal_m2.py

FETCH RESULTS when done (check `modal app list` / the dashboard for completion):

    modal volume get grail-artifacts /reranker_gate_cache/gflownet_m2_test_seed0.json results/
    modal volume get grail-artifacts /reranker_gate_cache/gflownet_m2_test_seed1.json results/
    modal volume get grail-artifacts /reranker_gate_cache/gflownet_m2_test_seed2.json results/
    python scripts/aggregate_seeds.py results/gflownet_m2_test_seed*.json
------------------------------------------------------------------------------------
"""
import modal

REPO = "https://github.com/doctawho42/GRAIL.git"
BRANCH = "metabench-reranker"

# Code comes from git (my pushed commits incl. --logz-lr); data/checkpoints/caches come
# from Volumes. numpy<2 and rdkit 2024.9.x are the load-bearing pins for this stack.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libxrender1", "libxext6", "libsm6")
    .run_commands(f"git clone --branch {BRANCH} --depth 1 {REPO} /root/GRAIL")
    .workdir("/root/GRAIL")
    .run_commands(
        "pip install --no-cache-dir 'numpy<2'",
        "pip install --no-cache-dir 'rdkit==2024.9.6'",
        "pip install --no-cache-dir -r requirements.txt",
        "pip install --no-cache-dir -e .",
    )
)

app = modal.App("grail-m2")
data_vol = modal.Volume.from_name("grail-data", create_if_missing=True)
art_vol = modal.Volume.from_name("grail-artifacts", create_if_missing=True)

# M2 hyperparameters. logz_lr=0.04 is rescaled for 1200 substrates: logZ moves
# ~(steps/epoch)*logz_lr = (1200/16)*0.04 ~ 3/epoch, converging to the beta=6 target
# (~O(12)) in a few epochs -- the same convergence speed that gave M1 its PASS at 100
# substrates with logz_lr=0.3 (100/16*0.3 ~ 2/epoch).
# Budget-fit scale ($15 credits left): CPU-only serial env is the wall, so keep the scale
# modest and guaranteed-cheap. logz_lr=0.16 rescaled for 300 subs (300/16 ~ 19 steps/epoch
# x 0.16 ~ 3/epoch -> same logZ-convergence speed as the validated M1 recipe).
M2_ARGS = [
    "--train-substrates", "300",
    "--max-depth", "2",
    "--max-size", "10",
    "--epochs", "8",
    "--top-k", "50",
    "--logz-lr", "0.16",
    "--n-samples", "4",
    "--eval-split", "test",
    "--test-substrates", "200",    # representative clean-test subsample
    "--workers", "8",
    "--no-bootstrap",
]

# The git clone ships tracked files in grail_metabolism/data/ (PCA pickles, smirks) so we
# CANNOT mount the data Volume over it. Mount the Volume at a separate empty path and
# symlink just our staged SDFs + triples into the repo's data dir, leaving the tracked
# featurization files intact.
DATA_MOUNT = "/vol_data"
DATA_FILES = [
    "train.sdf", "val.sdf", "test.sdf",
    "train_triples.txt", "val_triples.txt", "test_triples.txt",
    "train_triples_clean.txt", "val_triples_clean.txt", "test_triples_clean.txt",
]


def _link_data():
    """Symlink the Volume's SDFs + triples into grail_metabolism/data/ (idempotent)."""
    import os

    dst_dir = "/root/GRAIL/grail_metabolism/data"
    for name in DATA_FILES:
        src, dst = f"{DATA_MOUNT}/{name}", f"{dst_dir}/{name}"
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)


@app.function(
    image=image,
    # CPU-ONLY: the env is CPU-bound (RDKit rule application + featurization + tautomer),
    # GPU util was ~0%, and CPU is ~3-5x cheaper/hr AND immune to the CUDA OOM that killed
    # the GPU runs. 32GB RAM comfortably holds the batch-16 forest-graph peak (~14.5GB).
    gpu=None,
    cpu=8.0,               # 8 physical cores for the 8-worker spawn pool (RDKit pool-gen)
    memory=32768,
    volumes={
        DATA_MOUNT: data_vol,                            # SDFs + triples (symlinked into repo data dir)
        "/root/GRAIL/artifacts": art_vol,                # checkpoints + reranker_gate_cache (caches+results)
    },
    timeout=86400,         # 24h; per-epoch cache saves mean a timeout/kill loses no pool-gen (re-run resumes warm)
)
def run_m2(seeds=(0, 1, 2)):
    import os
    import subprocess
    import sys

    os.chdir("/root/GRAIL")
    # Reduce CUDA fragmentation (the OOM error explicitly suggested this).
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    _link_data()
    for seed in seeds:
        out = f"artifacts/reranker_gate_cache/gflownet_m2_test_seed{seed}.json"
        cmd = [sys.executable, "-u", "scripts/run_gflownet.py", *M2_ARGS,
               "--seed", str(seed), "--out", out]
        print(f"\n===== GRAIL M2 seed {seed} =====\n{' '.join(cmd)}\n", flush=True)
        subprocess.run(cmd, check=True)
        art_vol.commit()   # persist caches + this seed's results before the next seed
        print(f"===== seed {seed} done, artifacts committed =====", flush=True)
    print("\n===== M2 ALL SEEDS DONE =====", flush=True)


@app.function(
    image=image,
    volumes={"/root/GRAIL/artifacts": art_vol},   # checkpoint only; no data needed for pre-flight
    timeout=1800,
)
def warmup():
    """Force the image build + validate the env and staged checkpoint before the long run."""
    import os

    os.chdir("/root/GRAIL")
    import rdkit
    import torch

    print(f"torch={torch.__version__} rdkit={rdkit.__version__} cuda={torch.cuda.is_available()}", flush=True)
    import grail_metabolism  # noqa: F401  -- proves `pip install -e .` worked
    from pathlib import Path

    from grail_metabolism.model.grail import _read_checkpoint

    state = _read_checkpoint(Path("artifacts/full5000_priors/checkpoints/generator.pt"))
    assert state is not None, "generator.pt failed to load from the grail-artifacts Volume"
    print(
        f"generator.pt OK: has_arch={'arch' in state} n_rules={len(state.get('rules', []))}",
        flush=True,
    )
    return "warmup ok"


@app.function(
    image=image,
    volumes={DATA_MOUNT: data_vol},
    timeout=1800,
)
def decompress():
    """Gunzip any ``*.sdf.gz`` staged on the data Volume into ``*.sdf`` and drop the .gz.

    The SDFs are text and compress ~17x, so we upload the tiny .gz over a slow home
    uplink and expand them here on Modal's fast disk.
    """
    import glob
    import gzip
    import os
    import shutil

    d = DATA_MOUNT
    for gz in sorted(glob.glob(f"{d}/*.sdf.gz")):
        out = gz[:-3]
        with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=16 * 1024 * 1024)
        os.remove(gz)
        print(f"decompressed {os.path.basename(out)} ({os.path.getsize(out) // 1_000_000}M)", flush=True)
    data_vol.commit()
    print("decompress: committed", flush=True)


@app.function(
    image=image,
    gpu="T4",
    cpu=8.0,
    memory=32768,
    volumes={DATA_MOUNT: data_vol, "/root/GRAIL/artifacts": art_vol},
    timeout=3600,
)
def smoke():
    """Tiny end-to-end run on REAL data to validate the full pipeline before the 12h M2."""
    import os
    import subprocess
    import sys

    os.chdir("/root/GRAIL")
    _link_data()
    cmd = [sys.executable, "-u", "scripts/run_gflownet.py",
           "--train-substrates", "5", "--max-depth", "2", "--max-size", "6",
           "--epochs", "1", "--top-k", "20", "--logz-lr", "0.1", "--n-samples", "2",
           "--eval-split", "val", "--eval-substrates", "3", "--workers", "4", "--no-bootstrap",
           "--out", "artifacts/reranker_gate_cache/gflownet_smoke.json"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    art_vol.commit()
    print("SMOKE OK", flush=True)


@app.local_entrypoint()
def main():
    # .spawn() (NOT .remote()): fire-and-forget server-side so the 3-seed job survives the
    # local client disconnecting. Modal warns .remote()/.map() in detached apps may be
    # canceled when the caller disconnects. Run with `modal run --detach`.
    fc = run_m2.spawn()
    print(f"SPAWNED run_m2 -> function call id: {fc.object_id}", flush=True)
    print("Poll results:  modal volume ls grail-artifacts /reranker_gate_cache", flush=True)
