# AIHero Skills 使用说明

## 来源

本机已安装 AIHero / Matt Pocock 的 Skills for Real Engineers catalog。

- 目录页：https://www.aihero.dev/skills-catalog
- 源码仓库：https://github.com/mattpocock/skills
- 本次安装位置：`C:\Users\Yangfeifei\.codex\skills`

安装时官方 `npx skills@latest add mattpocock/skills -y -g` 路线在 `git clone` 阶段遇到 TLS 握手失败，因此改为下载 GitHub zip 包后手动安装到 Codex 全局 skills 目录。

## 已安装 Skill

本次安装了活跃目录中的 engineering 和 productivity skills，共 21 个：

| Skill | 用途 |
| --- | --- |
| `ask-matt` | 不确定该用哪个 skill 时，让 agent 帮你选择工作流。 |
| `setup-matt-pocock-skills` | 第一次在项目中使用这些 skills 前，配置 issue tracker、triage 标签和文档布局。 |
| `grill-me` | 对想法、计划或设计进行追问澄清。 |
| `grill-with-docs` | 在追问澄清的同时沉淀项目术语、ADR 和相关文档。 |
| `grilling` | `grill-me` / `grill-with-docs` 背后的可复用追问流程。 |
| `to-prd` | 把当前对话整理为 PRD。 |
| `to-issues` | 把 PRD、计划或规格拆成可独立执行的 issue。 |
| `triage` | 梳理 backlog、issue 或外部 PR，生成可交给 agent 执行的任务说明。 |
| `implement` | 基于 PRD 或 issue 执行开发任务。 |
| `tdd` | 按 red-green-refactor 流程做测试驱动开发。 |
| `prototype` | 用一次性原型验证状态模型、逻辑或 UI 方向。 |
| `diagnosing-bugs` | 按复现、缩小、假设、插桩、修复、回归测试的循环诊断问题。 |
| `research` | 基于高可信来源调研问题，并把结果整理为 Markdown。 |
| `code-review` | 从指定 commit、branch 或 merge-base 开始做代码审查。 |
| `codebase-design` | 设计或改进模块接口，提高模块深度、可测试性和可维护性。 |
| `domain-modeling` | 建立或打磨项目领域模型、术语表和 ADR。 |
| `improve-codebase-architecture` | 扫描代码库架构问题，并输出可视化分析报告。 |
| `resolving-merge-conflicts` | 处理正在进行的 merge 或 rebase 冲突。 |
| `handoff` | 把当前上下文压缩成交接文档，方便另一个 agent 接手。 |
| `teach` | 在当前工作区内讲解一个技能或概念。 |
| `writing-great-skills` | 编写和改进 skills 的参考工作流。 |

## 如何使用

安装后需要重启 Codex，新的 skills 才会被加载到可用 skill 列表里。

重启后，可以直接在请求里点名 skill，例如：

```text
使用 grill-with-docs，帮我梳理这个危险废物路径优化项目的术语和架构决策。
```

```text
使用 tdd，给 JSON 参数读取功能补测试。
```

```text
使用 code-review，review 当前工作区相对 main 的改动。
```

```text
使用 to-prd，把我们关于危险废物调度优化的需求整理成 PRD。
```

也可以先让路由 skill 判断：

```text
使用 ask-matt，判断这个任务该用哪个 skill：我想把 MILP 模型改造成可配置的实验平台。
```

## 推荐使用流程

新项目或第一次使用这些 skills 时：

1. 先运行 `setup-matt-pocock-skills`，确定 issue tracker、标签体系和文档位置。
2. 需求还模糊时，用 `grill-me` 或 `grill-with-docs` 追问清楚。
3. 需要正式沉淀需求时，用 `to-prd`。
4. 需要拆任务时，用 `to-issues`。
5. 进入开发时，用 `implement` 或 `tdd`。
6. 遇到复杂问题时，用 `diagnosing-bugs`、`prototype` 或 `research`。
7. 开发完成后，用 `code-review` 检查 diff。
8. 长会话结束前，用 `handoff` 生成交接文档。

## 针对当前项目的例子

当前项目是危险废物收运和处置优化，可以这样使用：

```text
使用 domain-modeling，帮我建立危险废物收运优化项目的领域术语表。
```

```text
使用 grill-with-docs，追问我 JSON 参数文件后续还需要支持哪些业务字段，并记录 ADR。
```

```text
使用 tdd，为 sample_params.json 的读取、写入和 tuple key 还原补测试。
```

```text
使用 improve-codebase-architecture，扫描当前 Python 项目的模块职责和后续扩展点。
```

```text
使用 research，调研危险废物路径优化中常见的 MILP 建模约束，并生成引用文档。
```

## 维护说明

- 已下载的源码包位于当前项目的 `mattpocock-skills-main.zip` 和 `mattpocock-skills-main/`。
- 真正生效的安装目录是 `C:\Users\Yangfeifei\.codex\skills`。
- 如果未来重新安装同名 skill，先确认是否要覆盖已有目录，避免丢失本地修改。
- Codex 当前会话不会自动加载刚安装的 skills；请重启 Codex 后再使用。
