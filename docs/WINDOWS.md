# Working on this project from Windows

The Mac is not needed. Every GPU hour runs on Kaggle; the machine you sit at
only builds notebooks, pushes them, downloads what they wrote, records it and
commits. Any machine with Git, Python and the Kaggle CLI does that.

The code lives in the repository
`https://github.com/Ahmadreza-Merikh/voice-cloning`. `PROJECT_STATUS.md` is
the handover: where the work is and the next concrete step. Read it first,
every time.

## One-time setup

1. Install Git for Windows, Python 3.11 (the project takes 3.10-3.12) and the
   Claude desktop app.
2. Clone **without line-ending conversion**, so files stay byte-identical to
   what was staged and verified from the Mac:

   ```
   git clone -c core.autocrlf=false https://github.com/Ahmadreza-Merikh/voice-cloning.git "Voice Cloning"
   cd "Voice Cloning"
   ```

3. The environment:

   ```
   python -m venv .venv
   .venv\Scripts\pip install -e ".[dev]" kaggle
   ```

   That is enough to drive Kaggle and run the tests; the engines' own extras
   (`f5`, `asr`, `eval`, ...) matter only for rendering on the machine itself.

4. Kaggle, as `marsmanam`: `.venv\Scripts\kaggle auth login` opens the
   browser. Check with `.venv\Scripts\kaggle config view` (it must say
   `auth_method: OAUTH`) and `.venv\Scripts\kaggle quota`.
   On the Mac an old read-only `kaggle.json` shadowed the login, which is why
   the commands in this repository say `KAGGLE_CONFIG_DIR=~/.kaggle-oauth`.
   On Windows drop that part, unless a `kaggle.json` sits in
   `%USERPROFILE%\.kaggle` - then do the same: log in with
   `$env:KAGGLE_CONFIG_DIR="$HOME\.kaggle-oauth"` set, in every shell.
5. Claude: unzip `claude-settings.zip` (made on the Mac) into
   `%USERPROFILE%\.claude\`: your instructions (`CLAUDE.md`, `rules\`) and the
   memory notes. Then in the app's Code tab open the project folder and start
   with: "Read PROJECT_STATUS.md and continue from there." The memory notes
   are in `memory\` inside the zip; ask the session to copy them into its own
   memory folder, whose path it knows.

## What is not in Git

* `outputs/benchmark/{baseline,finetuned,finetuned-bestof4}` - renders that
  `kaggle/build_dataset.py` stages into the Kaggle dataset. Needed only to
  re-stage it. The dataset itself carries them: download
  `marsmanam/persian-voice-eval` and copy `repo/outputs/benchmark/<name>` into
  `outputs/benchmark/`.
* `remote/` - kernel outputs downloaded for reading or listening. Fetch them
  again from Kaggle when needed: `kaggle/fetch_output.py` downloads only the
  files you name, e.g.

  ```
  .venv\Scripts\python kaggle\fetch_output.py marsmanam/persian-voice-omnivoice-test remote\omnivoice-test-v1 test/
  ```

* The Mac app's model caches (`~/.cache/persian-voice-cloning`) - only for
  rendering on the machine itself; nothing in the Kaggle pipeline reads them.

## Rules that hold on every machine

* **One machine drives Kaggle and Git at a time.** Both machines log in as
  the same Kaggle account; one T4 pair session takes both of its GPU slots and
  the weekly quota is shared. Before starting anywhere: `git pull`. When
  stopping: commit and `git push`, so the other machine starts from it.
* A Kaggle push starts a paid run. Every long run's budget is written into its
  record before the push (PROJECT_STATUS.md, "Standing limits").
* Do not rebuild a committed notebook on Windows unless it has to change: the
  builders write text with the platform's line ending, so a rebuild there
  shows every line as changed. Push the committed one as it is.
* The Mac's hooks (`~/.claude/hooks`, bash) do not come along. What they
  enforced still holds: commit after each unit of work, and no heavy training
  on the local machine - it goes to Kaggle.

## Commit hashes

On 2026-09-30 the history was rewritten before its first push, so every hash
cited before that date is an old one. `docs/HISTORY-REWRITE-2026-09-30.md`
maps each cited hash to its commit now.
