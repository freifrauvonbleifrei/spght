# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

import pywt


# we want wavelets where we implicitly assume that higher level
# means smaller intervals (nesting)
# -> half normalization instead of 1/sqrt(2) normalization

half_haar_filters = ([0.5, 0.5], [-0.5, 0.5], [1.0, 1.0], [1.0, -1.0])

half_haar = pywt.Wavelet(name="half_haar", filter_bank=half_haar_filters)
