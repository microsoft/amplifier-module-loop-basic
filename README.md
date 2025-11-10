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
