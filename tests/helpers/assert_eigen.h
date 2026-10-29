#pragma once

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace sio_test {

template <typename Derived>
void ExpectMatrixFinite(const Eigen::MatrixBase<Derived> &m) {
  EXPECT_TRUE(m.allFinite());
}

template <typename DerivedA, typename DerivedB>
void ExpectMatrixNear(const Eigen::MatrixBase<DerivedA> &a,
                      const Eigen::MatrixBase<DerivedB> &b, double tol) {
  ASSERT_EQ(a.rows(), b.rows());
  ASSERT_EQ(a.cols(), b.cols());
  for (int r = 0; r < a.rows(); ++r) {
    for (int c = 0; c < a.cols(); ++c) {
      EXPECT_NEAR(a(r, c), b(r, c), tol);
    }
  }
}

template <typename Derived>
void ExpectSymmetric(const Eigen::MatrixBase<Derived> &m, double tol) {
  ASSERT_EQ(m.rows(), m.cols());
  ExpectMatrixNear(m, m.transpose(), tol);
}

} // namespace sio_test
