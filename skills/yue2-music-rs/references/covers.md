# ABC and audio covers

Given native ABC, inspect it before rendering. For a melody-only cover, remove
chord symbols explicitly; `cot: "melody"` does not strip supplied chords:

```sh
./target/release/examples/music_tools inspect work/reference.abc --output work/source-check.json
./target/release/examples/music_tools strip-chords work/reference.abc work/melody.abc
./target/release/yue2_rs generate --request work/cover.json --cot melody \
  --abc-file work/melody.abc --output outputs/cover
```

Chord removal keeps both melodies by default. Select `--keep-voice Vocal` or `Ins`
only when the task calls for dropping the other part. To preserve harmony too,
retain chords and use `--cot full`. For an instrumental cover, follow
[instrumental.md](instrumental.md) to transfer the lead melody into Ins.

## Audio references and transcription

The Rust repo ports YuE2 generation and its VAE. It does **not** contain SheetSage2,
MERT, ASR or SongBench inference. An audio path cannot be passed to the native
generator as a reference condition. Do not invent a `--transcribe` flag, feed MERT
features as codec tokens, or describe a generated cover as voice cloning,
source separation or local audio inpainting.

If an authorized external transcription tool/environment is available, use it to
produce ABC, preserve the original audio and raw transcription, and inspect the
exported melody, meter, key and warnings. Otherwise explain the missing
transcription capability and request a native ABC score or an appropriate tool.
Do not install the Python/NVIDIA stack as an implicit part of native generation.

Exchange a score file between the environments. Unsupported generic ABC needs
explicit normalization and review; the helper fails rather than dropping notation.
Retain a complete reference phrase and its repetitions when the cover must preserve
them. A symbolic invariant check does not establish sample-accurate realization
or preserve the source recording's singer/timbre.
