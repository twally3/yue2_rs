# Native macOS setup

Find the `yue2_rs` checkout containing `Cargo.toml`, `src/` and this skill. All
commands in these references run from that repository root. Do not assume the
current directory is the skill folder or that an installed skill includes models.

Build the CLI and the CPU-only skill helpers with Rust and Apple's Command Line Tools:

```sh
cargo build --release --bins --examples
./target/release/yue2_rs --help
./target/release/examples/music_tools --help
```

The default inference backend is Metal/MPS plus Accelerate. MPSGraph decodes the
FP32 VAE on the GPU. `YUE2_BACKEND=cpu` is the diagnostic CPU fallback. The skill
helpers parse scores and package recordings without loading weights or using a GPU.
No PyTorch, Transformers, Python environment or CUDA is needed.

## Checkpoints

| Model | Use |
| --- | --- |
| `m-a-p/YuE2-3B` | Score planning, semantic generation and acoustic synthesis |
| `m-a-p/YuE2-Vae` | Default listening decoder |
| `m-a-p/YuE2-Vae-legacy` | Separate decoder for reproducing upstream evaluation protocols |

The CLI downloads a missing snapshot using system `curl`, records its revision
under `models/`, and reuses it offline. Pin Hub commits with `--revision` and
`--vae-revision`. To ensure a run needs no network, supply complete existing local
snapshot directories with `--model` and `--vae`; the Rust CLI has no `--offline` flag.

The recordings' `result.json` contains checkpoint file hashes, and `config.json`
records generation settings and backend. Keep these with the audio. Checkpoints
and generated outputs are ignored by Git. Do not copy a model into a listening bundle.

The native port does not implement SheetSage2 or MERT transcription/feature extraction.
Their Python dependencies and NVIDIA setup instructions are not part of this port.
See [covers.md](covers.md) for exchanging an externally prepared score.
