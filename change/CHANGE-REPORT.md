# CHANGE REPORT (w3)

Window: cd5ef814..c291e79 (2285 commits, 8458 files) — master@{19day}...master@{5day}

Packages changed: 290 (minor: 213, patch: 77)

## New packages in window (25)
- @deepseek-ai/dsh-api-workspace-files (packages/api/workspace-files)
- @deepseek-ai/dsh-client-file-upload (packages/client/file-upload)
- @deepseek-ai/dsh-client-resources (packages/client/resources)
- @deepseek-ai/dsh-client-ui-dockkit (packages/client/ui-dockkit)
- @deepseek-ai/dsh-client-ui-open-in-app (packages/client/ui-open-in-app)
- @deepseek-ai/dsh-client-ui-schedule (packages/client/ui-schedule)
- @deepseek-ai/dsh-client-ui-sidebar-documentpreview (packages/client/ui-sidebar-documentpreview)
- @deepseek-ai/dsh-client-ui-sidebar-files (packages/client/ui-sidebar-files)
- @deepseek-ai/dsh-client-ui-sidebar-right (packages/client/ui-sidebar-right)
- @deepseek-ai/dsh-experimental-code-runtime-python (packages/experimental/code-runtime-python)
- @deepseek-ai/dsh-tool-present (packages/fs/tool-present)
- @deepseek-ai/dsh-host-open-in-app (packages/host/open-in-app)
- @deepseek-ai/dsh-session-format-catalog (packages/session/session-format-catalog)
- @deepseek-ai/dsh-session-format-v0-to-v1 (packages/session/session-format-v0-to-v1)
- @deepseek-ai/dsh-session-format-v1-to-v2 (packages/session/session-format-v1-to-v2)
- @deepseek-ai/dsh-session-format-v2-to-v3 (packages/session/session-format-v2-to-v3)
- @deepseek-ai/dsh-session-format (packages/session/session-format)
- @deepseek-ai/dsh-session-turn-outline (packages/session/session-turn-outline)
- @deepseek-ai/dsh-remote-mock (packages/test-support/remote-mock)
- @deepseek-ai/dsh-chunked-list (packages/util/chunked-list)
- @deepseek-ai/dsh-deque (packages/util/deque)
- @deepseek-ai/dsh-http-proxy (packages/util/http-proxy)
- @deepseek-ai/dsh-package-manifest (packages/util/package-manifest)
- @deepseek-ai/dsh-util-time (packages/util/time)
- @deepseek-ai/dsh-util-values (packages/util/values)

## Sample classified changes
- **minor** @deepseek-ai/dsh: feat(dsh-cli): backport feedback and file refinements to 0.1.5
- **minor** @deepseek-ai/dsh-desktop-host: feat(dsh-desktop-host): backport feedback and file refinements to 0.1.5
- **minor** @deepseek-ai/dsh-desktop: feat(dsh-desktop): backport feedback and file refinements to 0.1.5
- **minor** @deepseek-ai/dsh-web-frontend: feat(dsh-web): backport feedback and file refinements to 0.1.5
- **minor** @deepseek-ai/dsh-benchmarks: feat(dsh-benchmarks): backport feedback and file refinements to 0.1.5
- **minor** @deepseek-ai/node-addon-system-workspace: feat(dsh-system): backport feedback and file refinements to 0.1.5
- **minor** @deepseek-ai/node-addon-system-darwin-arm64: feat(dsh-darwin-arm64): keep model persona prefix and place cwd in suffix
- **minor** @deepseek-ai/node-addon-system-darwin-x64: feat(dsh-darwin-x64): keep model persona prefix and place cwd in suffix
- **minor** @deepseek-ai/node-addon-system: feat(dsh-entry): restore V4 Flash Vision Exp catalog entry
- **minor** @deepseek-ai/node-addon-system-linux-arm64: feat(dsh-linux-arm64): keep model persona prefix and place cwd in suffix
- **minor** @deepseek-ai/node-addon-system-linux-x64: feat(dsh-linux-x64): keep model persona prefix and place cwd in suffix
- **patch** @deepseek-ai/dsh-acp: fix(dsh-acp): isolate npm caches and synchronize ACP snapshot completion
- **minor** @deepseek-ai/dsh-api-gateway: feat(dsh-gateway): serve bounded files through the filesystem provider
- **minor** @deepseek-ai/dsh-api-remotes: feat(dsh-remotes): expose workspace file operations to the Client
- **minor** @deepseek-ai/dsh-api-session-controller: feat(dsh-session-controller): add identity V2-to-V3 migration and writer skeleton
