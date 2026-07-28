# Assets

`samples/` is for short demo clips you want tracked in git (they are exempted
from the audio ignore rules in `.gitignore`).

Reference recordings you actually use belong in the **Voice Library** instead:

    voiceclone voices add "Narrator" -r /path/to/reference.wav

Saved voices live in `voices/`, which is git-ignored - reference audio is
personal data and should not be committed.
