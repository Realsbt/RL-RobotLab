# ToGo LFS Exported Policies

This directory contains inference-only policy exports. Training checkpoints,
optimizer state, TensorBoard events, W&B data, and run logs are intentionally
excluded.

| Directory | Task | Input contract | Asset compatibility |
| --- | --- | --- | --- |
| `locomotion_latency_50000` | latency-robust MoE-CTS locomotion at iteration 50000 | 45-D frame, internal 10-frame term-major history, 12-D action | `ToGo_LFs_v0p1_new` |
| `quiet_beta_1p4_5000` | fixed-beta MUTE experiment at iteration 4999 | 45-D frame, internal 10-frame term-major history, 12-D action | original MUTE training asset/configuration |
| `backflip_r10j` | R9i iteration 225 actor used by the R10j assisted landing controller | 60-D frame, 12-D action | `ToGo_LFs_v0p1_new` |
| `locomotion_legacy_50000` | pre-new-asset locomotion at iteration 50000 | 45-D frame, internal 10-frame term-major history, 12-D action | `ToGo_LFs_v0p1_prototype` only |

Each directory contains a TorchScript `policy.pt` and an ONNX `policy.onnx`.
The TorchScript CTS exports expose both `forward()` and `reset()`; call
`reset()` before priming a new episode. Do not use the legacy locomotion policy
with the new asset because its hip-pitch convention and default pose belong to
the old model.

## SHA-256

```text
304ee7cefd161e0d63cfa74a971bedc23f8dc6f04f9a44d67d22d8ccb55ba030  locomotion_latency_50000/policy.pt
e206127c4cd50c18876457010dd2666bd954036e9d34e7ce7c9beaf710bf01bc  locomotion_latency_50000/policy.onnx
1371dab56f1eb897f2cf38da5f0ab692d42f590ec43724bb01cfde18eee4a8f7  quiet_beta_1p4_5000/policy.pt
5042c902f1123623cc3ec8fb9ba7d0d3cb32d84c00f47792d40f8cbcece07b3f  quiet_beta_1p4_5000/policy.onnx
5e79e346a9100d833b558efed9c9690c16ca9b4ce309b15b15e05215b2f30ac5  backflip_r10j/policy.pt
c804b55ea4b41732aaf4552b3d1cb09553115c61abde7369d8f99b66cde03615  backflip_r10j/policy.onnx
c487d46762fd120b8530e91258e93a081d176de1f22f0721df67baa6b76fb9c9  locomotion_legacy_50000/policy.pt
b45c580289082dc61222c7353e85518f3c2f35445ecd9062776f0365dbbe3f70  locomotion_legacy_50000/policy.onnx
```

The default new-asset MuJoCo configuration uses
`locomotion_latency_50000/policy.pt`. Use `togo_lfs_legacy.yaml` only when
running the prototype asset-policy pair.
