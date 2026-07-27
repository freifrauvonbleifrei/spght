# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from spght.util import depends_on_optional, module_is_available
from spght import pywt_compat


def test_module_is_available():
    assert module_is_available("numpy")
    assert not module_is_available("definitely_not_a_module")


def test_depends_on_optional():
    @depends_on_optional("definitely_not_a_module")
    def needs_missing_module():
        return 1  # pragma: no cover

    with pytest.raises(ImportError, match="definitely_not_a_module"):
        needs_missing_module()

    @depends_on_optional("numpy")
    def needs_available_module():
        return 1

    assert needs_available_module() == 1


def test_pywt_compat_degrades_without_pywt(monkeypatch):
    monkeypatch.setattr(pywt_compat, "PYWT_AVAILABLE", False)
    # the type check needs no import when pywt is (simulated) missing
    assert pywt_compat.is_pywt_wavelet(object()) is False
