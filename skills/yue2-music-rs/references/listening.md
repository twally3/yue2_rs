# Local listening and delivery

```sh
./target/release/examples/music_tools listen outputs/original outputs/edited \
  --output outputs/comparison
```

Supply full native song directories, not a standalone `plan` or `decode` output.
The output must be fresh and outside all source directories. The helper creates
a self-contained HTML page with no external resources or scripts. It copies only
`request.json`, `config.json`, `result.json`, available `score.abc`, and
`audio.flac`/`audio.wav`. It excludes model weights, latents, token arrays, logs,
machine configuration and arbitrary sidecars. Keep custom edit manifests/checks
beside the source work and link them separately.

Copied artifacts are hashed and checked against the source's native receipt.
A mismatched/missing receipt withholds the audio player. Missing/invalid metadata,
truncation, source failure records and failed score checks appear as issues.
Exit 0 means the cases passed these checks; 1 means a bundle was created with
cases needing review; 2 means a command error. Do not discard cases needing review
to make a comparison look successful. `manifest.json` hashes copied files and
`index.html`, excluding its own recursive self-hash.

The helper checks an audio container signature, not full codec decoding. These
checks establish file integrity and score structure; they do not rate composition,
singing, exact acoustic note realization or instrument/vocal removal. If listening
is available, compare whole songs and the changed passages. Otherwise state that
the recording has not been audited by ear. Optional ASR or evaluation needs a
separate available tool and must be labeled separately from symbolic evidence.

Deliver playable `audio.flac`, the full actual style/lyrics, the recorded duration,
and requested before/after score and invariant reports. Use absolute local paths
for response links/media. No bundle is published or uploaded by the helper. Its
prompts, lyrics and compositions remain user content; sharing requires the user's
authorization.
