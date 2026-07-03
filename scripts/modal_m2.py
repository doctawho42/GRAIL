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
M2_ARGS = [
    "--train-substrates", "1200",
    "--max-depth", "2",
    "--max-size", "10",
    "--epochs", "15",
    "--top-k", "50",
    "--logz-lr", "0.04",
    "--n-samples", "16",
    "--eval-split", "test",
    "--test-substrates", "2000",   # >1246 clean-test subs => full test, touch-once
    "--workers", "8",
    "--no-bootstrap",
]


@app.function(
    image=image,
    gpu="T4",              # matches Colab (14GB used there); env is CPU-bound so this is cheap insurance
    cpu=8.0,               # 8 physical cores for the 8-worker spawn pool (RDKit pool-gen)
    memory=32768,
    volumes={
        "/root/GRAIL/grail_metabolism/data": data_vol,   # SDFs + triples (Volume root == data dir)
        "/root/GRAIL/artifacts": art_vol,                # checkpoints + reranker_gate_cache (caches+results)
    },
    timeout=86400,         # 24h; per-epoch cache saves mean a timeout/kill loses no pool-gen (re-run resumes warm)
)
def run_m2(seeds=(0, 1, 2)):
    import os
    import subprocess
    import sys

    os.chdir("/root/GRAIL")
    for seed in seeds:
        out = f"artifacts/reranker_gate_cache/gflownet_m2_test_seed{seed}.json"
        cmd = [sys.executable, "-u", "scripts/run_gflownet.py", *M2_ARGS,
               "--seed", str(seed), "--out", out]
        print(f"\n===== GRAIL M2 seed {seed} =====\n{' '.join(cmd)}\n", flush=True)
        subprocess.run(cmd, check=True)
        art_vol.commit()   # persist caches + this seed's results before the next seed
        print(f"===== seed {seed} done, artifacts committed =====", flush=True)
    print("\n===== M2 ALL SEEDS DONE =====", flush=True)


@app.local_entrypoint()
def main():
    run_m2.remote()
