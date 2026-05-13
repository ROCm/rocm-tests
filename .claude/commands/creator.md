Use the `creator` subagent to generate a 100% framework-compliant rocm-tests pytest test from a GPU feature requirement or requirements document.

Invoke it with: Agent(subagent_type="creator", prompt="<user's requirement>")

Usage: /creator

If the user hasn't described what they want to test, ask:
> "What GPU operation or feature do you want to test? Include the expected outcome and any performance thresholds."

Then pass their full description to the creator agent.
