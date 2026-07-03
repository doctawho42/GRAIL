#!/usr/bin/env bash
# One-time staging of GRAIL Stage-2b M2 inputs onto Modal Volumes.
#   grail-data      (root == grail_metabolism/data/) : SDFs + triples (~1.4G)
#   grail-artifacts (root == artifacts/)             : trained checkpoints (+ later: caches, results)
#
# Storage ops (volume create/put) work on an authed account; only the compute run
# (`modal run`) needs credits. Safe to run this first, pay, then run.
#
# Run once from anywhere:  bash scripts/modal_stage.sh
set -euo pipefail

# Real data files (the worktree has symlinks -> the main checkout).
DATA="/Users/nikitapolomosnov/PycharmProjects/GRAIL/grail_metabolism/data"
# Trained generator/filter checkpoints (in this worktree's artifacts).
CKPT="/Users/nikitapolomosnov/PycharmProjects/GRAIL/.claude/worktrees/hungry-pasteur-25d746/artifacts/full5000_priors/checkpoints"

echo "==> creating volumes (idempotent)"
modal volume create grail-data      2>/dev/null || echo "   grail-data exists"
modal volume create grail-artifacts 2>/dev/null || echo "   grail-artifacts exists"

echo "==> checkpoints (11M, fast) -> grail-artifacts"
modal volume put -f grail-artifacts "$CKPT/generator.pt" /full5000_priors/checkpoints/generator.pt
modal volume put -f grail-artifacts "$CKPT/filter.pt"    /full5000_priors/checkpoints/filter.pt

echo "==> triples (small) -> grail-data"
for f in train_triples.txt val_triples.txt test_triples.txt \
         train_triples_clean.txt val_triples_clean.txt test_triples_clean.txt; do
  modal volume put -f grail-data "$DATA/$f" "/$f"
done

echo "==> SDFs (~1.4G, the slow upload) -> grail-data"
for f in train.sdf val.sdf test.sdf; do
  echo "    uploading $f ..."
  modal volume put -f grail-data "$DATA/$f" "/$f"
done

echo "==> staging complete. Verify:"
echo "    modal volume ls grail-data"
echo "    modal volume ls grail-artifacts /full5000_priors/checkpoints"
echo "Then run:  modal run --detach scripts/modal_m2.py"
