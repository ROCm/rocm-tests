# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_rock_artifacts_available.py -- public ROCm/TheRock artifact endpoints are reachable.

Verifies the public distribution channels that TheRock/rockrel publish to are up
and populated (rock_deb/rpm_package_availability_check + tarball/pip channels):

- nightly tarball S3 bucket lists ``.tar.gz`` dist artifacts for the GPU family,
- the nightly pip index (PEP 503) responds,
- the prerelease deb/rpm package repository base + GPG key respond.

All checks are HTTP/S3 only (no GPU, no download, no install, no root) so the
suite is ``hw.cpu_only``. Endpoints are the public CloudFront/S3 URLs documented
in ``ROCm/rockrel`` and used by ``ROCm/TheRock/build_tools/install_rocm_from_artifacts.py``.

The GPU family comes from the ``artifact_group`` fixture, derived from the
mandatory ``--gpu-arch``.
"""

import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import pytest

_NIGHTLY_TARBALL_S3 = "https://therock-nightly-tarball.s3.amazonaws.com"
_NIGHTLY_PIP_INDEX = "https://rocm.nightlies.amd.com/v2/"
_PRERELEASE_PKG_BASE = "https://rocm.prereleases.amd.com/packages/"
_PRERELEASE_GPG_KEY = "https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg"
_TIMEOUT = 30


def _http_get(url: str, what: str) -> tuple[int, bytes]:
    """GET *url*; return (status, body).

    An HTTP error status or an unreachable host both mean the channel is not
    available, which is the condition under test, so each is reported as a
    failure naming *what* rather than surfacing as a raw traceback.
    """
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "rocm-tests"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        pytest.fail(f"{what} unavailable: HTTP {exc.code} {exc.reason} for {url}")
    except urllib.error.URLError as exc:
        pytest.fail(f"{what} unreachable: {exc.reason} for {url}")


@pytest.mark.runtime.fast
def test_rock_nightly_tarball_available(artifact_group: str):
    """The public nightly S3 bucket lists dist tarballs for the GPU family."""
    prefix = f"therock-dist-linux-{artifact_group}-"
    url = f"{_NIGHTLY_TARBALL_S3}/?list-type=2&prefix={prefix}&max-keys=100"
    status, body = _http_get(url, "nightly tarball S3 bucket")
    assert status == 200, f"S3 list returned HTTP {status} for {url}"
    keys = [el.text for el in ET.fromstring(body).iter() if el.tag.endswith("Key") and el.text]
    tarballs = [k for k in keys if k.endswith(".tar.gz")]
    assert tarballs, f"no nightly dist tarballs found under prefix {prefix!r} in {_NIGHTLY_TARBALL_S3}"


@pytest.mark.runtime.fast
def test_rock_pip_index_available(artifact_group: str):
    """The public nightly pip index lists the GPU family and its ROCm packages.

    The index is per-family: ``/v2/`` lists GPU families, and ``/v2/<family>/`` is
    the PEP 503 package index for that family (``rocm``, ``rocm-sdk``, ...).
    """
    root_status, root_body = _http_get(_NIGHTLY_PIP_INDEX, "nightly pip index root")
    assert root_status == 200, f"pip index root returned HTTP {root_status} for {_NIGHTLY_PIP_INDEX}"
    assert (
        artifact_group.encode() in root_body
    ), f"GPU family {artifact_group!r} not listed at pip index root {_NIGHTLY_PIP_INDEX}"

    family_index = f"{_NIGHTLY_PIP_INDEX}{artifact_group}/"
    fam_status, fam_body = _http_get(family_index, f"per-family pip index for {artifact_group}")
    assert fam_status == 200, f"per-family pip index returned HTTP {fam_status} for {family_index}"
    lowered = fam_body.lower()
    assert (
        b"rocm-sdk" in lowered or b'href="rocm/"' in lowered
    ), f"no ROCm packages listed at per-family pip index {family_index}"


@pytest.mark.runtime.fast
@pytest.mark.parametrize(
    ("name", "url"),
    [
        ("deb/rpm package repo base", _PRERELEASE_PKG_BASE),
        ("ROCm repo GPG key", _PRERELEASE_GPG_KEY),
    ],
    ids=["package_repo_base", "gpg_key"],
)
def test_rock_package_repo_available(name: str, url: str):
    """The public prerelease deb/rpm package repository base and GPG key are reachable."""
    status, body = _http_get(url, name)
    assert status == 200, f"{name} returned HTTP {status} for {url}"
    assert body, f"{name} returned an empty response for {url}"
