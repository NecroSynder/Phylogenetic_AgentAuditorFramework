#!/usr/bin/env python3
"""
One-time lossless exporter converting pretrained Jerry-0591/Fusang TensorFlow 2.13 Keras
checkpoints (len_240 & len_1200) into standard NumPy `.npz` archives (plus golden Keras
prediction vectors for strict numerical parity verification <= 1e-5 in PyTorch CUDA 12.8).
Run once inside `env_fusang`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np


def export_model(repo_dir: Path, bucket: int, rel_weight_path: str, out_npz: Path, ref_prefix: Path | None = None) -> None:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    import fusang as fusang_mod

    prev = os.getcwd()
    os.chdir(str(repo_dir))
    try:
        model = fusang_mod.get_dl_model_240() if bucket == 240 else fusang_mod.get_dl_model_1200()
        model.load_weights(filepath=rel_weight_path).expect_partial()
    finally:
        os.chdir(prev)

    weights_dict: dict[str, np.ndarray] = {}
    conv_idx = 0
    bn_idx = 0
    bilstm_idx = 0
    dense_idx = 0

    print(f"\n=== Exporting bucket={bucket} from {rel_weight_path} ===")
    for layer in model.layers:
        cls = layer.__class__.__name__
        ws = layer.get_weights()
        if not ws:
            continue
        if cls == "Conv2D":
            weights_dict[f"conv{conv_idx}_kernel"] = ws[0].astype(np.float32)
            weights_dict[f"conv{conv_idx}_bias"] = ws[1].astype(np.float32)
            print(f"  conv{conv_idx}: kernel={ws[0].shape}, bias={ws[1].shape}")
            conv_idx += 1
        elif cls == "BatchNormalization":
            # Keras BN get_weights() order: [gamma, beta, moving_mean, moving_variance]
            weights_dict[f"bn{bn_idx}_gamma"] = ws[0].astype(np.float32)
            weights_dict[f"bn{bn_idx}_beta"] = ws[1].astype(np.float32)
            weights_dict[f"bn{bn_idx}_mean"] = ws[2].astype(np.float32)
            weights_dict[f"bn{bn_idx}_var"] = ws[3].astype(np.float32)
            weights_dict[f"bn{bn_idx}_eps"] = np.array([layer.epsilon], dtype=np.float32)
            print(f"  bn{bn_idx}: shape={ws[0].shape}, eps={layer.epsilon}")
            bn_idx += 1
        elif cls == "Bidirectional":
            # Keras Bidirectional(LSTM) get_weights():
            # [fw_kernel, fw_rec_kernel, fw_bias, bw_kernel, bw_rec_kernel, bw_bias]
            assert len(ws) == 6, f"Expected 6 tensors in Bidirectional LSTM, got {len(ws)}"
            weights_dict[f"bilstm{bilstm_idx}_fw_ih"] = ws[0].astype(np.float32)
            weights_dict[f"bilstm{bilstm_idx}_fw_hh"] = ws[1].astype(np.float32)
            weights_dict[f"bilstm{bilstm_idx}_fw_b"] = ws[2].astype(np.float32)
            weights_dict[f"bilstm{bilstm_idx}_bw_ih"] = ws[3].astype(np.float32)
            weights_dict[f"bilstm{bilstm_idx}_bw_hh"] = ws[4].astype(np.float32)
            weights_dict[f"bilstm{bilstm_idx}_bw_b"] = ws[5].astype(np.float32)
            print(f"  bilstm{bilstm_idx}: fw_ih={ws[0].shape}, fw_hh={ws[1].shape}, bw_ih={ws[3].shape}")
            bilstm_idx += 1
        elif cls == "Dense":
            weights_dict[f"dense{dense_idx}_kernel"] = ws[0].astype(np.float32)
            weights_dict[f"dense{dense_idx}_bias"] = ws[1].astype(np.float32)
            print(f"  dense{dense_idx}: kernel={ws[0].shape}, bias={ws[1].shape}")
            dense_idx += 1
        else:
            raise ValueError(f"Unexpected weighted layer {cls} ({layer.name})")

    assert conv_idx == 8 and bn_idx == 8 and bilstm_idx == 3 and dense_idx == 2, (
        f"Layer count mismatch: conv={conv_idx}, bn={bn_idx}, bilstm={bilstm_idx}, dense={dense_idx}"
    )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_npz), **weights_dict)
    print(f"  [SAVED] {out_npz} ({out_npz.stat().st_size / 1e6:.2f} MB)")

    if ref_prefix is not None:
        rng = np.random.default_rng(42)
        x_ref = rng.integers(0, 5, size=(16, 4, bucket, 1)).astype(np.float32)
        y_ref = model.predict(x_ref, batch_size=16, verbose=0).astype(np.float32)
        np.save(f"{ref_prefix}_x.npy", x_ref)
        np.save(f"{ref_prefix}_y.npy", y_ref)
        print(f"  [SAVED] Golden Keras reference predictions -> {ref_prefix}_{{x,y}}.npy (sample y[0]={y_ref[0]})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusang-repo", default="external_models/Fusang")
    args = parser.parse_args()
    repo_dir = Path(args.fusang_repo).resolve()
    out_dir = repo_dir / "dl_model" / "exported_npz"

    export_model(
        repo_dir, 240, "./dl_model/len_240/S1G/best_weights_clas",
        out_dir / "len_240_standard_gamma.npz",
        ref_prefix=out_dir / "golden_ref_240",
    )
    export_model(
        repo_dir, 1200, "./dl_model/len_1200/S2G/best_weights_clas",
        out_dir / "len_1200_standard_gamma.npz",
        ref_prefix=out_dir / "golden_ref_1200",
    )
    print("\n[DONE] Exact pretrained Fusang Keras weights exported cleanly.")


if __name__ == "__main__":
    main()
