# collaborating-with-gemini-cli

[中文](README.md) | English

A skill for **Codex CLI**: via a JSON bridge script, it delegates tasks such as **code review / debugging / alternative implementation comparisons** to **Google Gemini CLI** (default model: `gemini-3-pro-preview`), and returns results as structured JSON for multi-model collaboration.

Compared to [collaborating-with-claude-code](https://github.com/ZhenHuangLab/collaborating-with-claude-code), due to the poor context capability and attention mechanism of `gemini-3-pro-preview`, this skill is intentionally conservative by default:

- Defaults to **read-only** (`--no-full-access`)
- Encourages **one-shot + file-scoped runs** (defaults to less-than-5-file preference; the main agent / Codex can include more via explicit `--file`, adjusting `--max-files`, or disabling guardrails)

Main entry points:

- `SKILL.md` (the Codex skill definition)
- `scripts/gemini_cli_bridge.py` (the bridge script)

## Install to `~/.codex/skills/`

1) Choose the install directory `~/.codex/skills` (create it if it doesn't exist):

```bash
mkdir -p ~/.codex/skills
```

2) Clone this repository into the `skills/` directory:

```bash
cd ~/.codex/skills
git clone https://github.com/ZhenHuangLab/collaborating-with-gemini-cli.git collaborating-with-gemini-cli
```

3) Verify the folder structure; it should contain at least `SKILL.md` and `scripts/`:

```bash
ls -la ~/.codex/skills/collaborating-with-gemini-cli
```

After that, mention `collaborating-with-gemini-cli` (or `$collaborating-with-gemini-cli`, or say something equivalent) in a conversation to trigger it.

## Dependencies

- Python 3 (to run the bridge script).
- Gemini CLI installed and available (`gemini --version`).
  - Install: `npm i -g @google/gemini-cli`
- Gemini CLI authenticated (Google account login or API key auth, depending on your local setup).

## Run manually (without Codex CLI)

One-shot headless run:

```bash
python scripts/gemini_cli_bridge.py --cd "/path/to/repo" --PROMPT "Review src/auth/login.py for bypasses; propose fixes as a unified diff."
```

Recommended (explicit file scope):

```bash
python scripts/gemini_cli_bridge.py --cd "/path/to/repo" --file "src/auth/login.py" --PROMPT "Review this file and propose a unified diff."
```

Multi-turn sessions (same `SESSION_ID`):

```bash
# Turn 1
python scripts/gemini_cli_bridge.py --cd "/repo" --file "src/auth/login.py" --PROMPT "List risks + propose patch."

# Turn 2 (resume)
python scripts/gemini_cli_bridge.py --cd "/repo" --SESSION_ID "uuid-from-response" --PROMPT "Now propose tests for the patch."
```

Allow edits (defaults to `auto_edit`; see `SKILL.md` for YOLO mode):

```bash
python scripts/gemini_cli_bridge.py --full-access --cd "/repo" --file "src/foo.py" --PROMPT "Refactor for clarity; keep behavior; apply edits."
```

For a complete parameter reference, output format, and guardrail behavior, see `SKILL.md`.

## License

MIT License. See `LICENSE`.
