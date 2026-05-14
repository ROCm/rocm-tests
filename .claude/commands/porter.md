Use the `porter` subagent to port an external test — a shell script, raw Python file, non-compliant pytest, C++ gtest, or test from another AMD framework — into a fully framework-compliant rocm-tests pytest file.

Invoke it with: Agent(subagent_type="porter", prompt="Port <source file path>: <any additional context>")

Usage: /porter <source-file>

If no source file is provided, ask:
> "Which file do you want to port? Provide the path or paste the contents."

Pass the source file path and any relevant context (target layer, expected CI tier) to the porter agent.
