# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
linter.py -- MarkerLinter: validates test functions carry required marker dimensions.

Required dimensions: hw.*, ci.*, layer.* (from REQUIRED_DIMENSIONS in this file).
Run via: python3 -m framework.markers.linter tests/e2e/myfile.py
Integrated into the PostToolUse hook in .claude/settings.json.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import pathlib

from framework.markers.taxonomy import CATEGORY_PROFILES, MARKER_SCHEMA, REQUIRED_DIMENSIONS


@dataclass
class MarkerViolation:
    """A single marker validation failure.

    Attributes:
        file:       Path to the offending file.
        function:   Name of the test function with the violation.
        line:       Line number of the function definition.
        message:    Human-readable description of the violation.
    """

    file: str
    function: str
    line: int
    message: str


class MarkerLinter:
    """Validate pytest markers in a Python source file against MARKER_SCHEMA.

    Instantiate once and call ``lint_file()`` for each file to check.
    The linter is stateless between calls.
    """

    def lint_file(self, path: str) -> list[MarkerViolation]:
        """Parse *path* and return all marker violations found.

        Effective markers for each test function are the union of:
        - Category profile markers (from ``CATEGORY_PROFILES``, keyed by path prefix)
        - Module-level ``pytestmark = [...]`` assignments in the file
        - Function-level ``@pytest.mark.*`` decorators

        Args:
            path: Absolute or relative path to a ``*.py`` test file.

        Returns:
            List of MarkerViolation instances, empty if all markers are valid.
        """
        source = pathlib.Path(path).read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as exc:
            return [MarkerViolation(file=path, function="<module>", line=0, message=f"SyntaxError: {exc}")]

        inherited = self._inherited_markers(path)
        module_marks = self._module_pytestmark(tree)

        violations: list[MarkerViolation] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            func_markers = self._extract_markers(node)
            # Merge all sources; set eliminates duplicates while preserving coverage
            effective = list(set(inherited) | set(module_marks) | set(func_markers))
            violations.extend(self._check_markers(path, node.name, node.lineno, effective))
        return violations

    def _inherited_markers(self, path: str) -> list[str]:
        """Return profile marker strings for the longest matching CATEGORY_PROFILES prefix.

        Args:
            path: File path being linted (absolute or relative).

        Returns:
            List of ``"dim.val"`` strings from the matching profile, or empty list.
        """
        try:
            rel = pathlib.Path(path).resolve().relative_to(pathlib.Path.cwd().resolve())
        except ValueError:
            rel = pathlib.Path(path)
        rel_str = rel.as_posix()
        match = ""
        for prefix in CATEGORY_PROFILES:
            if rel_str.startswith(prefix) and len(prefix) > len(match):
                match = prefix
        return list(CATEGORY_PROFILES.get(match, []))

    def _module_pytestmark(self, tree: ast.Module) -> list[str]:
        """Extract marker strings from module-level ``pytestmark = [...]`` assignments.

        Handles the common form::

            pytestmark = [pytest.mark.ci.nightly, pytest.mark.hw.gpu]

        Args:
            tree: Parsed AST of the test module.

        Returns:
            List of ``"dim.val"`` strings, empty if no ``pytestmark`` assignment found.
        """
        markers: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "pytestmark":
                    elts = []
                    if isinstance(node.value, ast.List):
                        elts = node.value.elts
                    elif isinstance(node.value, ast.Attribute):
                        # Single marker without brackets
                        elts = [node.value]
                    for elt in elts:
                        name = self._dotted_name(elt)
                        if name and name.startswith("pytest.mark."):
                            markers.append(name[len("pytest.mark.") :])
        return markers

    def _extract_markers(self, func_node: ast.FunctionDef) -> list[str]:
        """Return a list of ``dimension.value`` strings from @pytest.mark.* decorators."""
        markers: list[str] = []
        for dec in func_node.decorator_list:
            name = self._dotted_name(dec)
            if name and name.startswith("pytest.mark."):
                markers.append(name[len("pytest.mark.") :])
        return markers

    def _dotted_name(self, node: ast.expr) -> str | None:
        """Recursively build dotted attribute path, e.g. 'pytest.mark.hw.gpu'."""
        if isinstance(node, ast.Attribute):
            parent = self._dotted_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Call):
            return self._dotted_name(node.func)
        return None

    def _check_markers(self, file: str, func: str, line: int, markers: list[str]) -> list[MarkerViolation]:
        """Validate the marker list for one test function."""
        violations: list[MarkerViolation] = []

        # Build dimension → values map from the markers present
        dim_values: dict[str, list[str]] = {}
        for marker in markers:
            parts = marker.split(".", 1)
            if len(parts) != 2:
                continue
            dim, val = parts
            dim_values.setdefault(dim, []).append(val)

        # Check required dimensions
        for dim in REQUIRED_DIMENSIONS:
            if dim not in dim_values:
                violations.append(
                    MarkerViolation(
                        file=file,
                        function=func,
                        line=line,
                        message=f"Missing required marker dimension '{dim}.*'",
                    )
                )

        # Check values against schema
        for dim, vals in dim_values.items():
            if dim not in MARKER_SCHEMA:
                violations.append(
                    MarkerViolation(
                        file=file,
                        function=func,
                        line=line,
                        message=f"Unknown marker dimension '{dim}' — not in MARKER_SCHEMA",
                    )
                )
                continue
            allowed = MARKER_SCHEMA[dim]
            for val in vals:
                if val not in allowed:
                    violations.append(
                        MarkerViolation(
                            file=file,
                            function=func,
                            line=line,
                            message=(f"Invalid marker '{dim}.{val}' — " f"allowed: {sorted(allowed)}"),
                        )
                    )

        return violations

    @staticmethod
    def format_violations(violations: list[MarkerViolation]) -> str:
        """Format a list of violations into a human-readable string.

        Args:
            violations: Output of lint_file().

        Returns:
            Formatted multi-line string, empty if violations is empty.
        """
        lines = []
        for v in violations:
            lines.append(f"  {v.file}:{v.line}  {v.function}()  — {v.message}")
        return "\n".join(lines)
