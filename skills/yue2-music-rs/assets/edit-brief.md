# Score edit brief

Goal: describe the requested musical change.

Inputs: source `score.abc`, `request.json`, and available listening notes.

Record the user-approved invariants: voices/passages, sounding pitches, onsets,
durations, tempo, meter, structure and any permitted melodic adaptation.
Specify the desired instruments and lyric changes, if any.

Write into a fresh directory:

- `score.abc`: complete edited native two-voice score, including untouched passages.
- `request.json`: actual style, lyrics, seed and cot; use supported fields only.
- `edit_manifest.json`: source hash, edited bars/voices, harmony changes, allowed
  melody/rhythm/lyric changes, and the preserved invariants.
- A short musical rationale tied to note spans and beat positions.

Inspect the whole score with `music_tools inspect` and check retained voices with
`music_tools compare`. Submit the new score through `--abc-file`; do not modify
a saved plan and reuse its original prefix. A request with null/absent `abc` and
no `--abc-file` generates a new plan rather than applying the score edit.

Distinguish written notation, successful rendering, file integrity and listening
evidence in the result. Do not report audio generation when only the score was edited.
