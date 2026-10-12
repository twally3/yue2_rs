---
name: yue2-music-rs
description: Generate songs and instrumentals with the native macOS Rust YuE2 port, retain and edit ABC plans, render ABC covers, and build local listening comparisons. Use for YuE2 music creation, style or lyric changes, reharmonization, melody preservation, and instrumental voice transfer.
---

# YuE2 Music for Rust

Turn the user's musical request into playable audio and a retained composition.
Use the native `yue2_rs` runtime on macOS; generation and all bundled helpers
require neither Python nor additional Cargo dependencies.

Locate the Rust checkout. For this repository's skill folder, it is two directory
levels above this folder. If the skill has been installed elsewhere, locate the
checkout separately. Run the commands below from that checkout, using fresh
output paths. Read [setup.md](references/setup.md) for building and model selection.

## Choose a workflow

| Request | Workflow |
| --- | --- |
| Song with editable melody and harmony | Generate with `cot: "full"` |
| Melody plan with free accompaniment | Use `cot: "melody"`; remove chords from supplied ABC |
| Song without an editable score | Use `cot: "off"` |
| Instrumental music | Follow [instrumental.md](references/instrumental.md): YuE2 plans, then transfer Vocal to Ins |
| ABC melody cover | Inspect native ABC, strip chords, render with `cot: "melody"` |
| Reharmonize or change style, lyrics, tempo or structure | Retain a baseline, edit a copy, submit it as new ABC/text |
| Compare recordings | Build a local listening bundle with `music_tools listen` |
| Cover an audio recording | Obtain ABC externally, then use the ABC workflow; see [covers.md](references/covers.md) |

YuE2 writes the initial score unless the user provides one or explicitly asks the
agent to compose it. Keep generation sequential on this Mac and retain original
outputs before editing. Use default BF16 inference, FP32 VAE decoding and 32 ODE
steps unless the user asks for a change; do not shorten a song to hide resource failures.

## Generate, retain and edit

Adapt [assets/prompt.json](assets/prompt.json). Put genre, instruments, vocal character,
language and intended tempo in `style`; put section tags and singable words in `lyrics`.
Keep implementation instructions out of the lyrics. Save the actual request in the
workspace and pass its path explicitly:

```sh
./target/release/yue2_rs generate --request work/prompt.json --output outputs/song
./target/release/yue2_rs plan --request work/prompt.json --output outputs/plan
./target/release/yue2_rs generate --plan outputs/plan --output outputs/from-plan
```

Use the local model directories or recorded revisions when repeating a run. Inspect
`result.json` and its `truncated` flags, `request.json`, `score.abc`, and the recording.
Keep the token arrays and `latent.npy`. A completed process and a playable file do
not establish musical quality. A nonzero generation exit status signals truncation
or an error; retain available evidence and report it.

Read [generation.md](references/generation.md) for request fields, overrides,
saved-plan semantics, all-mode comparisons and cached decoding. Read
[abc-editing.md](references/abc-editing.md) before changing a score.

Define the requested invariants: exact sounding pitch/onset/duration, pitch alone,
contour, or permitted melodic adaptation. Keep passages and voices explicit.
Use [assets/edit-brief.md](assets/edit-brief.md) when it helps record those choices.
Check the complete edited score and connect it to the generation command:

```sh
./target/release/examples/music_tools inspect edits/jazz.abc --output edits/check.json
./target/release/examples/music_tools compare outputs/song/score.abc edits/jazz.abc \
  --voices Vocal --output edits/melody-check.json
./target/release/yue2_rs generate --request edits/jazz.json --cot full \
  --abc-file edits/jazz.abc --output outputs/jazz
```

Add `--allow-tempo-change` when tempo is intentionally changed. A comparison failure
may be expected for authorized rhythmic edits; explain the actual differences.
Preserve the source, edited score, request and edit manifest. Changed style, lyrics
or ABC require regeneration; redecoding old latents cannot implement those edits.

## Deliver audio and evidence

```sh
./target/release/examples/music_tools listen outputs/song outputs/jazz \
  --output outputs/comparison
```

The helper creates a local `index.html` with audio, the exact prompts, score and
metadata, and a hashed `manifest.json`. Read [listening.md](references/listening.md)
for its integrity checks and delivery limits. Link or embed the actual local audio
using absolute paths in the assistant's response. Include the full prompt/lyrics
and requested edit checks. Distinguish score validation from listening; do not
claim to have heard a recording or verified vocal removal without that evidence.

This skill adapts the upstream YuE2 music skill under [Apache 2.0](LICENSE).
The instrumental conversion workflow retains its [MIT notice](licenses/instrumental-MIT.txt).
Model weights retain their own licenses.
