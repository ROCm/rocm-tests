Use the `refiner` subagent to review an existing rocm-tests pytest test for regressions, stability issues, and coverage gaps — extend it with new variants — or compare it against its upstream source for drift.

Invoke it with: Agent(subagent_type="refiner", prompt="<file path and mode or request>")

Usage:
  /refiner <file>                          # full 4-persona review + top-3 improvements
  /refiner review-as <persona> <file>      # single-persona deep review
  /refiner <file> add <description>        # extend with a new test variant
  /refiner <file> drift                    # upstream drift analysis (requires Ported from: in docstring
                                           # or user-provided upstream source)

Valid personas: developer, tester, automation, devops

If the file path is missing, ask the user which test file to review, extend, or drift-check.
If the intent is unclear, ask:
> "Do you want to review this test for improvements, extend it with new variants, or compare it against its upstream source?"

Drift mode checks:
  - Reads Ported from: / Upstream ref: / Ported on: from the module docstring
  - Compares test objective, CLI args, assertion sentinels, thresholds, and test variants
  - Applies the Objective Deviation Protocol when upstream objective has fundamentally changed
  - Does NOT silently update a test whose upstream objective has deviated
