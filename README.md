# Amplifier Basic Loop Orchestrator Module

Reference implementation of sequential agent loop orchestration.

## Prerequisites

- **Python 3.11+**
- **[UV](https://github.com/astral-sh/uv)** - Fast Python package manager

### Installing UV

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Purpose

Provides a standard, deterministic orchestration loop for AI agents. This is the default orchestrator and serves as the reference implementation.

## Contract

**Module Type:** Orchestrator
**Mount Point:** `orchestrators`
**Entry Point:** `amplifier_module_loop_basic:mount`

## Behavior

- **Parallel tool execution**: Multiple tool calls execute concurrently (assumes LLM intent)
- Deterministic context updates: Results added in original order
- Robust error handling: Failures return error results, never crash
- Event correlation: `parallel_group_id` tracks related executions
- No streaming (see loop-streaming for that)

### Experimental media references

With a compatible host-bound `experimental.media.v1` capability, canonical rich
tool images are staged before `tool:post` observers. Only the final effective
hook output's structured `output["experimental.media.v1"] = {"refs": [...]}` is
published. Explicit null, removal, denial, errors and JSON-looking strings cannot
restore refs from the original result. Ordered tool messages carry admitted refs
in the same metadata namespace, including completed siblings during cancellation.

Media admission also requires the host's `session.media.tool_post` callable:
`async boundary(event, *, hooks=None, coordinator=None)` returns
`{hook_result: processed HookResult, result: final effective value, complete: bool}`.
The callback owns hook emission and coordinator processing exactly once. A
complete explicit result, including null, is authoritative over registry
aggregate data. Missing/incomplete/malformed witnesses drop staged refs and
record explicit unavailable text, without replaying the hook chain.
Fail-closed media admission preserves ordinary text/status from the processed
effective result, with the reserved media member removed. Explicit null/removal
never recovers the original text or refs; a processed denial keeps its reason.

This witness is necessary because installed Rust Core 2.0.2 can lose result data
while aggregating injection chains, and discards unreturned in-place Python
handler edits. A qualified host bridge must witness the full chain, including
those mutations and errors; aggregate-only or synthetic-registry tests do not
establish support. Without a bridge, ordinary text keeps legacy hook semantics,
not a new universal text-redaction guarantee.

The capability is structural (version 1; `ingest`, `publish`, `validate`,
`resolve`, `prepare_request`, `report`); this module imports only Core, not
Foundation or an application. Producers can read their actual concurrent call
identity from the host's task-keyed `_tool_dispatch_contexts` during execution.
Consumer readiness is reported only after a complete boundary witness, never
merely because an unverified callable is mounted. It is an observation, not
execution permission or producer activation.

The host's existing no-argument `session.durable_checkpoint` runs after the
settled ordered append. Ref-bearing results require that callback; `False` or an
exception stops subsequent requests without rerunning tools. The host owns the
exact saved-prefix acknowledgment and must refuse media preparation without an
actual canonical commit. Cancellation saves paired results and the closing
assistant message, and never starts a new provider request. An interrupted
checkpoint has an unknown durable outcome: cleanup runs, in-memory evidence is
preserved, and the host reconciles the save without the loop replaying it.
If a child self-cancels, the loop requests cancellation of unfinished siblings
before draining them. One shielded settlement preserves completed evidence,
ordered synthetic cancellation rows and cleanup under repeated caller
cancellation; it never redispatches tools or repeats an uncertain checkpoint.

Both ordinary and max-iteration final `ChatRequest` objects remain reference-only.
Install the qualified host/provider adapter on the actual selected provider,
including context compaction/counting: the loop never resolves images or hydrates
saved requests. Without compatible handling, canonical rich images become useful
text plus an explicit unavailable warning, not fallback base64. Restored refs on
an unsupported loop request receive the same warning without altering history.
An explicitly disabled capability (`enabled=False`) uses this same optional
text-only fallback, without ingest, publication or readiness calls. Enabled
capability validation failures still refuse; they are not silently downgraded.

This is an experimental integration boundary, not a claim of visual inference,
exact image-budget accounting, store durability or host/provider parity. Those
require qualification against the accepted Foundation and host/provider graph.

Optional candidate pairing checks in `tests/test_media_f1_pairing.py` require
`AMPLIFIER_MEDIA_F1_SOURCE` pointing to an exact exported Foundation tree, with
that root first on `PYTHONPATH`. They use the actual store, source-sealed legacy
checkpoint and private provider wrapper; without that explicit setup they skip.
Their synthetic complete hook witness is not real-registry bridge qualification.
Publication happens once per settled origin: retain the surviving ref set, or
drop the complete staged set. Removed refs are handled by that same publication.
The separate `safe-*` cases require `AMPLIFIER_MEDIA_HOOK_BRIDGE_SOURCE` pointing
to an exact bridge export. They invoke only the helper's public installer and
callback with actual Core HookRegistry. The initial safe bridge deliberately
returns `complete=False/result=None` for every call: these checks prove no
double emission/processing, no media promotion, and retained useful text/status
on that negative path. They do not prove positive image delivery or authorize
activation. No runtime module may use the helper's private partial-witness seam.

## Configuration

```toml
[[orchestrators]]
module = "loop-basic"
name = "basic"
config = {
    max_iterations = -1,  # Maximum iterations (-1 = unlimited, default)
    timeout = 300         # Timeout in seconds
}
```

## Usage

```python
# In amplifier configuration
[session]
orchestrator = "loop-basic"
```

## Dependencies

- `amplifier-core>=1.0.0`

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
