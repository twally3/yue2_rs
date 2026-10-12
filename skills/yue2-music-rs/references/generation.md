# Requests and staged generation

```json
{
  "id": "city_lights",
  "style": "Warm piano pop, expressive English vocal, bass and light drums, 88 BPM",
  "lyrics": "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain",
  "cot": "full",
  "seed": 831001,
  "abc": null,
  "cfg_scale": null
}
```

`style` and `lyrics` are required strings. `id` defaults to `song` and must be a
filename-safe identifier. `cot` defaults to `full`; use `full`, `melody`, or `off`.
`seed` defaults to **831001** and must be an integer in `[0, 2^63)`.
`abc` contains literal score text; use `--abc-file` for a score on disk.
The Rust loader does not implement the Python helper's `abc_path`, `tags`, `text`,
`duration`, `phonemes` or audio-reference conveniences. Use the actual supported fields.

`cfg_scale` defaults to 1.0 for planned modes and 1.01 for `off`; accepted overrides
are finite values from 0 through 20. CLI `--cot`, `--seed` and `--abc-file` override
the request's corresponding fields. External ABC must be nonempty and requires
`full` or `melody`. Seed alone does not guarantee identical audio across runtimes
or optimized kernels. Retain `audio.flac` to preserve an exact recording.

## Generation and mode comparison

```sh
./target/release/yue2_rs generate --request work/prompt.json --output outputs/song
```

For a requested comparison, run each mode sequentially into its own directory:

```sh
for mode in full melody off; do
  ./target/release/yue2_rs generate --request work/prompt.json --cot "$mode" \
    --output "outputs/modes-$mode" || break
done
```

That command assumes `abc` is absent/null; `off` cannot accept external ABC.
If a mode fails, report the failure and which modes were not run. Do not treat a
partial comparison as all modes passing. No wrapper hides or resumes failed runs.

`--config` accepts the native `yue2_generation_config.json` schema: `abc` and
`semantic` sampling objects, `ode_steps`, fixed `context: 24576`,
`ode_method: "midpoint"` and `version: "yue2-native-v1"`. Defaults are 4,096 ABC
tokens, 9,000 semantic tokens and 32 midpoint steps. Do not shorten these as a
silent fallback. `--vae-core-frames` changes decoder tile size, not requested song length.

## Plans and edits

```sh
./target/release/yue2_rs plan --request work/prompt.json --output outputs/plan
./target/release/yue2_rs generate --plan outputs/plan --output outputs/from-plan
```

The plan retains the request, ABC, score token IDs, exact prefix and a hash
manifest. Loading verifies those files. `--plan` preserves symbolic planning;
it still generates semantic tokens and acoustic audio. It is not interrupted-run
resume and does not lock the original waveform. A `plan` run does not produce audio.

Do not edit files inside a saved plan and pass it to `--plan`. To change the
lyrics/style/score or use a new seed, make a new request and supply the revised ABC:

```sh
./target/release/yue2_rs generate --request edits/prompt.json --cot full \
  --abc-file edits/score.abc --output outputs/edited
```

`--plan` cannot be combined with `--request`, `--cot`, `--seed` or `--abc-file`.
`--config` overrides are not stored in a standalone plan; retain the original
config file and pass it again when reproducing a plan rendered with custom settings.
The library also exposes `Pipeline::plan`, `generate_semantic`, `synthesize`,
`decode`, and `generate`; see `examples/staged.rs` for a Rust API example.

## Saved artifacts and cached decoding

Full generation saves `audio.flac` (PCM24, stereo), `request.json`, `config.json`,
`result.json`, `plan.json`, `plan_manifest.json`, `abc_tokens.npy`, `prefix.npy`,
`semantic.npy`, `latent.npy`, and `score.abc` when planned. Check
`result.json.truncated.abc` and `.semantic`, durations, timings and weight hashes.

```sh
./target/release/yue2_rs decode --latent outputs/song/latent.npy \
  --output outputs/redecoded.flac
```

Use `--vae` and `--vae-revision` to select the original decoder. A new decoder can
change the waveform while keeping the acoustic latents fixed. Decode writes a new
FLAC file rather than a full song directory. It is not score or lyric editing.
