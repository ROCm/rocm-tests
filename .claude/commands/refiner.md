Use the `refiner` subagent to review an existing rocm-tests pytest test for regressions, stability issues, and coverage gaps — or to extend it with new edge-case variants.

Invoke it with: Agent(subagent_type="refiner", prompt="<file path and mode or request>")

Usage:
  /refiner <file>                          # full 4-persona review + top-3 improvements
  /refiner review-as <persona> <file>      # single-persona deep review
  /refiner <file> add <description>        # extend with a new test variant

Valid personas: developer, tester, automation, devops

If the file path is missing, ask the user which test file to review or extend.
If the intent (review vs extend) is unclear, ask:
> "Do you want to review this test for improvements, or extend it with new variants?"
