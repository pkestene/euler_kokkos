# SPDX-FileCopyrightText: 2025 euler_kokkos authors
#
# SPDX-License-Identifier: Unlicense

add_executable(test_segfault test_segfault.cpp ${BACKWARD_ENABLE})
add_backward(test_segfault)
