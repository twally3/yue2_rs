# YuE2 in Rust

Native inference for the YuE2-3B AR–NAR Mixture-of-Transformers and YuE2 Oobleck VAE on macOS 11 or later, Intel and Apple Silicon. There are no Cargo dependencies. Large matrix operations use Apple's Metal Performance Shaders when a supported GPU is available, with Accelerate for CPU execution. The frozen tokenizer uses macOS ICU, checkpoint integrity uses CommonCrypto, and FLAC export uses CoreAudio. The small Objective-C bridge to Metal is compiled with the system Clang compiler.

The implementation reads the original BF16/FP32 safetensors checkpoints directly. It preserves the native prompt protocol, Unicode NFC/BPE tokenizer, grouped-query attention and RoPE, separate AR/NAR attention and MLP paths, request-local sampling and CFG, full-song acoustic noise and original chunk boundaries, 32-step midpoint flow matching, and FP32 VAE decoding with exact-core halo cropping. The VAE encoder is also available.

Run from the repository root:

```sh
cargo run --release --example generate -- --output outputs/create
cargo run --release --example generate -- --abc-file examples/melody.abc --cot melody --output outputs/melody
cargo run --release --example generate -- --abc-file examples/score.abc --cot full --output outputs/harmony-original
cargo run --release --example generate -- --abc-file examples/score-jazz.abc --cot full --output outputs/harmony-edited
```

These are the four generation examples from the source repository, with the original input files. They retain `audio.flac` (lossless PCM24), `score.abc`, exact planner and semantic token arrays, `latent.npy`, configuration, timings, and checkpoint identities. Existing output directories are refused and truncation produces a nonzero exit status. WAV float output is available through `audio::save_wav`.

Checkpoints download through macOS's `/usr/bin/curl` on first use, then reuse that commit offline. Supply local directories with `--model` and `--vae`, or select Hub commits with `--revision` and `--vae-revision`. The current local checkpoint cache contains YuE2-3B `c044757a011169583f363168348ae380946efff8` and YuE2-Vae `152733a19ad43aa67e367f9b5503ef8075bb5126`. Checkpoint weights and generated outputs are excluded from Git. Building requires Rust and Apple's Command Line Tools.

Export a composition before rendering, or reuse the exact saved token IDs:

```sh
cargo run --release --example plan -- --output outputs/plan
cargo run --release -- generate --plan outputs/plan --output outputs/from-plan
# Edit a copy of outputs/plan/score.abc, then render it:
cargo run --release --example generate -- --abc-file edited.abc --cot full --output outputs/edited
```

The library exposes `Pipeline::plan`, `generate_semantic`, `synthesize`, `decode`, and `generate`. The staged API example is:

```sh
cargo run --release --example staged -- outputs/staged
```

Decode cached latents with either released VAE:

```sh
cargo run --release -- decode --latent outputs/create/latent.npy --output outputs/redecoded.flac
cargo run --release -- decode --latent outputs/create/latent.npy --vae m-a-p/YuE2-Vae-legacy --output outputs/benchmark-decoder.flac
```

Sampling and ODE overrides use `--config` with the source repository's `yue2_generation_config.json` format. The defaults retain the native generation budgets; no context or token budget is silently shortened. The default VAE tile is 512 latent frames with 16-frame halos; use `--vae-core-frames` to change its memory footprint. Memory-mapped weights and conversion panels bound temporary storage. Metal operates on FP32 matrix tiles; model operations retain BF16 rounding and the VAE remains FP32. Single-token autoregressive decoding uses Accelerate. Set `YUE2_BACKEND=cpu` to force CPU execution. Long songs on a CPU take substantial time. Seeds make runs repeatable within a runtime; floating-point kernels and CPU/GPU sampling can change output across architectures and implementations.

Scope is the YuE2 model and its generation examples. The separate SheetSage2 and MERT transcription models, agent skill scripts, evaluation tooling, and CUDA serving backends are outside this port.

Code retains the source repository's Apache 2.0 license and the VAE component notices in `THIRD_PARTY_NOTICES.md` and `licenses/`. Checkpoint weights retain their upstream model license.
