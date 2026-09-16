# Preview release: v0.1.5-rc.2

Released: September 10, 2026

## Highlights

**Sidebar and file previews:** the right sidebar becomes a docking surface with tab types and a resource model, previewing file and skill references, images, and documents (HTML, PDF, Markdown, code) plus Mermaid, Graphviz, SVG, and HTML code fences — with unified file icons and delivery actions.

**Desktop distribution:** a managed Electron distribution ships alongside the existing web and CLI entry points, macOS notarization and runtime packaging run in parallel, and the workspace can be opened in local apps directly from the web UI.

**Session formats and persistence:** session logs gain a versioned format family with a catalog and v0→v1→v2→v3 migration decoders that stream released migrations, plus a turn-outline projection — while the SQLite persistence backend is removed in favor of the JSONL store.

**Models and routing:** the DeepSeek V41 Flash model joins the catalog alongside V4 (with a V4 Flash Vision experimental entry) and becomes the Chat Completions default, model switches are announced in the transcript, and every outbound request routes through the configured proxy.

**Plugin manifest and release tooling:** package metadata separates from DSH declarations, plugin manifests declare format, tags, and host compatibility, DSH alpha and canary release channels land, and pull requests adopt weighted approvals and LOC-ranked review owners.

## What's Changed

- feat(client): right Sidebar architecture — docking surface, tab types, resource model, workspace files [#3588](https://github.com/deepseek-ai/deepseek-harness/pull/3588)

- feat(web): preview file and skill references in the sidebar [#3917](https://github.com/deepseek-ai/deepseek-harness/pull/3917)

- feat(sidebar): preview image files [#3890](https://github.com/deepseek-ai/deepseek-harness/pull/3890)

- feat(client): document previews — HTML, PDF, Markdown and code [#3798](https://github.com/deepseek-ai/deepseek-harness/pull/3798)

- feat(web): preview Mermaid, Graphviz, SVG, and HTML code fences [#3710](https://github.com/deepseek-ai/deepseek-harness/pull/3710)

- feat(web): unify file icons, delivery actions, and sidebar interactions [#3819](https://github.com/deepseek-ai/deepseek-harness/pull/3819)

- feat(desktop): add managed Electron distribution [#3413](https://github.com/deepseek-ai/deepseek-harness/pull/3413)

- feat(workspace): open the workspace in local apps from the web UI [#3409](https://github.com/deepseek-ai/deepseek-harness/pull/3409)

- feat(session): V3 log migration collaboration baseline and version-upgrade cookbook [#3631](https://github.com/deepseek-ai/deepseek-harness/pull/3631)

- perf(session): stream released format migrations [#3585](https://github.com/deepseek-ai/deepseek-harness/pull/3585)

- refactor(session)!: remove SQLite persistence backend [#3339](https://github.com/deepseek-ai/deepseek-harness/pull/3339)

- feat(llm): add V41 Flash alongside V4 models [#3824](https://github.com/deepseek-ai/deepseek-harness/pull/3824)

- feat(agent): announce model switches [#3507](https://github.com/deepseek-ai/deepseek-harness/pull/3507)

- feat(net): route every outbound request through the configured proxy [#3198](https://github.com/deepseek-ai/deepseek-harness/pull/3198)

- feat(manifest): separate package metadata from DSH declarations [#3899](https://github.com/deepseek-ai/deepseek-harness/pull/3899)

- feat(release): add DSH alpha and canary channels [#2988](https://github.com/deepseek-ai/deepseek-harness/pull/2988)

- feat: enforce weighted pull request approvals [#3783](https://github.com/deepseek-ai/deepseek-harness/pull/3783)

- feat: rank review owners by changed LOC [#3764](https://github.com/deepseek-ai/deepseek-harness/pull/3764)

- feat(plugin-inventory): scope-grouped plugin list carrying agent preset compositions [#3316](https://github.com/deepseek-ai/deepseek-harness/pull/3316)

- feat(subagent): record and observe parent-owned child catalogs [#3674](https://github.com/deepseek-ai/deepseek-harness/pull/3674)

**Full Changelog:** https://github.com/deepseek-ai/deepseek-harness/compare/master@{19day}...master@{5day}
