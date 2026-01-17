# collaborating-with-gemini-cli

中文 | [English](README_EN.md)

这是 **Codex CLI** 的一个 skill：通过一个 JSON bridge 脚本，把“代码审查 / 调试 / 方案对比”等任务委托给 **Google Gemini CLI**（默认模型：`gemini-3-pro-preview`），并以结构化 JSON 结果返回，便于在多模型协作中使用。

与 [collaborating-with-claude-code](https://github.com/ZhenHuangLab/collaborating-with-claude-code) 相比，由于 `gemini-3-pro-preview` 的注意力机制以及上下文能力较差，本 skill 的默认策略更保守：

- 默认 **只读**（`--no-full-access`）
- 鼓励 **one-shot + 文件聚焦**（默认建议 5 个文件以内；如确实需要更多文件，应由主 agent / Codex 决定，可显式传多个 `--file` 或调整 `--max-files` / 关闭 guardrails）

核心入口：

- `SKILL.md`（Codex skill 定义）
- `scripts/gemini_cli_bridge.py`（Gemini CLI 桥接脚本）

## 安装到 `~/.codex/skills/`

1) 选择安装目录 `~/.codex/skills` (若不存在请创建)：

```bash
mkdir -p ~/.codex/skills
```
2) Clone 本仓库到 `skills/` 目录下

```bash
cd ~/.codex/skills
git clone https://github.com/ZhenHuangLab/collaborating-with-gemini-cli.git collaborating-with-gemini-cli

```
3) 验证文件结构，保证目录结构至少包含 `SKILL.md` 和 `scripts/`：

```bash
ls -la ~/.codex/skills/collaborating-with-gemini-cli
```

完成后，在对话中提到 `collaborating-with-gemini-cli`（或 `$collaborating-with-gemini-cli` , 或者表达类似的意思）即可触发使用。

## 依赖

- Python 3（用于运行 bridge 脚本）。
- Gemini CLI 已安装并可用（确保 `gemini --version` 可运行）。
  - 安装：`npm i -g @google/gemini-cli`
- Gemini CLI 已完成认证（Google 账号登录或 API Key 方式，取决于你的本机配置）。

## 手动运行（不通过 Codex CLI）

最简单的一次性调用：

```bash
python scripts/gemini_cli_bridge.py --cd "/path/to/repo" --PROMPT "Review src/auth/login.py for bypasses; propose fixes as a unified diff."
```

推荐（明确指定 focus files；默认建议 5 个文件以内，更贴合 effective-context 约束）：

```bash
python scripts/gemini_cli_bridge.py --cd "/path/to/repo" --file "src/auth/login.py" --PROMPT "Review this file and propose a unified diff."
```

多轮会话（同一 `SESSION_ID`）：

```bash
# Turn 1
python scripts/gemini_cli_bridge.py --cd "/repo" --file "src/auth/login.py" --PROMPT "List risks + propose patch."

# Turn 2 (resume)
python scripts/gemini_cli_bridge.py --cd "/repo" --SESSION_ID "uuid-from-response" --PROMPT "Now propose tests for the patch."
```

允许改文件（默认 `auto_edit`；更危险的 YOLO 见 `SKILL.md`）：

```bash
python scripts/gemini_cli_bridge.py --full-access --cd "/repo" --file "src/foo.py" --PROMPT "Refactor for clarity; keep behavior; apply edits."
```

更完整的参数说明、输出格式与 guardrails 见 `SKILL.md`。

## License

MIT License，详见 `LICENSE`。
