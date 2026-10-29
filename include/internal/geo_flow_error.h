#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>

// Geometric-flow gate: Σ_f = J_f P_cam J_fᵀ. The caller supplies the
// camera-frame relative pose covariance (including cross terms). Static
// pixels must keep (f_raft - f_geo) inside that ellipse (Mahalanobis² < 1
// after a ridge).
namespace sio_geo_flow {

constexpr float kMinFerrPx = 1.5f;
constexpr float kMinGeoOrRaftSq = 0.25f;
constexpr float kMinDisp = 0.1f;
constexpr float kFerrEps = 1e-6f;
constexpr float kMinSigmaTransM = 0.02f;

struct DofSigma {
  float wx = 0.0f;
  float wy = 0.0f;
  float wz = 0.0f;
  float tx = kMinSigmaTransM;
  float ty = kMinSigmaTransM;
  float tz = kMinSigmaTransM;
};

struct FlowJac {
  float du_wx = 0.0f;
  float dv_wx = 0.0f;
  float du_wy = 0.0f;
  float dv_wy = 0.0f;
  float du_wz = 0.0f;
  float dv_wz = 0.0f;
  float du_tx = 0.0f;
  float dv_tx = 0.0f;
  float du_ty = 0.0f;
  float dv_ty = 0.0f;
  float du_tz = 0.0f;
  float dv_tz = 0.0f;
};

struct SigmaF {
  float suu = 0.0f;
  float suv = 0.0f;
  float svv = 0.0f;
};

inline DofSigma MakeDofSigma(
    const Eigen::Matrix<double, 6, 1> *pose_sigma_cam) {
  DofSigma s;
  if (pose_sigma_cam == nullptr) {
    return s;
  }
  auto nonneg = [](double v) -> float {
    return std::max(0.0f, static_cast<float>(v));
  };
  s.wx = nonneg((*pose_sigma_cam)(0));
  s.wy = nonneg((*pose_sigma_cam)(1));
  s.wz = nonneg((*pose_sigma_cam)(2));
  s.tx = std::max(kMinSigmaTransM, nonneg((*pose_sigma_cam)(3)));
  s.ty = std::max(kMinSigmaTransM, nonneg((*pose_sigma_cam)(4)));
  s.tz = std::max(kMinSigmaTransM, nonneg((*pose_sigma_cam)(5)));
  return s;
}

// Camera-frame 6×6 pose covariance for Σ_f = J P Jᵀ. Null → rotation 0,
// translation floor only. Translation diagonals are floored at σ_t,min²;
// off-diagonals are kept (inflating diag only cannot increase |ρ|).
inline Eigen::Matrix<float, 6, 6> MakePoseCovCam(
    const Eigen::Matrix<double, 6, 6> *pose_cov_cam) {
  const float t2 = kMinSigmaTransM * kMinSigmaTransM;
  Eigen::Matrix<float, 6, 6> P = Eigen::Matrix<float, 6, 6>::Zero();
  P(3, 3) = t2;
  P(4, 4) = t2;
  P(5, 5) = t2;
  if (pose_cov_cam == nullptr) {
    return P;
  }
  P = pose_cov_cam->cast<float>();
  P = 0.5f * (P + P.transpose().eval());
  for (int i = 0; i < 6; ++i) {
    P(i, i) = std::max(P(i, i), 0.0f);
  }
  P(3, 3) = std::max(P(3, 3), t2);
  P(4, 4) = std::max(P(4, 4), t2);
  P(5, 5) = std::max(P(5, 5), t2);
  return P;
}

inline FlowJac ComputeFlowJac(float xn, float yn, float fx, float fy,
                              float inv_z) {
  FlowJac j;
  j.du_wx = -fx * xn * yn;
  j.dv_wx = -fy * (1.0f + yn * yn);
  j.du_wy = fx * (1.0f + xn * xn);
  j.dv_wy = fy * xn * yn;
  j.du_wz = -fx * yn;
  j.dv_wz = fy * xn;
  j.du_tx = fx * inv_z;
  j.dv_tx = 0.0f;
  j.du_ty = 0.0f;
  j.dv_ty = fy * inv_z;
  j.du_tz = -fx * xn * inv_z;
  j.dv_tz = -fy * yn * inv_z;
  return j;
}

inline Eigen::Matrix<float, 2, 6> FlowJacMatrix(const FlowJac &j,
                                                bool include_trans) {
  Eigen::Matrix<float, 2, 6> J;
  J(0, 0) = j.du_wx;
  J(1, 0) = j.dv_wx;
  J(0, 1) = j.du_wy;
  J(1, 1) = j.dv_wy;
  J(0, 2) = j.du_wz;
  J(1, 2) = j.dv_wz;
  if (include_trans) {
    J(0, 3) = j.du_tx;
    J(1, 3) = j.dv_tx;
    J(0, 4) = j.du_ty;
    J(1, 4) = j.dv_ty;
    J(0, 5) = j.du_tz;
    J(1, 5) = j.dv_tz;
  } else {
    J(0, 3) = 0.0f;
    J(1, 3) = 0.0f;
    J(0, 4) = 0.0f;
    J(1, 4) = 0.0f;
    J(0, 5) = 0.0f;
    J(1, 5) = 0.0f;
  }
  return J;
}

inline SigmaF AccumulateSigmaFFromPoseCov(
    const FlowJac &j, const Eigen::Matrix<float, 6, 6> &P, bool include_trans) {
  const Eigen::Matrix<float, 2, 6> J = FlowJacMatrix(j, include_trans);
  const Eigen::Matrix<float, 2, 2> Sf = J * P * J.transpose();
  SigmaF s;
  s.suu = Sf(0, 0);
  s.suv = 0.5f * (Sf(0, 1) + Sf(1, 0));
  s.svv = Sf(1, 1);
  return s;
}

inline void AddAxis(float ju, float jv, float sig, SigmaF &s) {
  const float vu = ju * sig;
  const float vv = jv * sig;
  s.suu += vu * vu;
  s.suv += vu * vv;
  s.svv += vv * vv;
}

// Diagonal-P special case: Σ_f = J diag(σ²) Jᵀ = Σ_i (J_i σ_i)(J_i σ_i)ᵀ.
inline SigmaF AccumulateSigmaF(const FlowJac &j, const DofSigma &sigma,
                               bool include_trans) {
  Eigen::Matrix<float, 6, 6> P = Eigen::Matrix<float, 6, 6>::Zero();
  P(0, 0) = sigma.wx * sigma.wx;
  P(1, 1) = sigma.wy * sigma.wy;
  P(2, 2) = sigma.wz * sigma.wz;
  P(3, 3) = sigma.tx * sigma.tx;
  P(4, 4) = sigma.ty * sigma.ty;
  P(5, 5) = sigma.tz * sigma.tz;
  return AccumulateSigmaFFromPoseCov(j, P, include_trans);
}

inline float TraceSqrt(const SigmaF &s) {
  return std::sqrt(std::max(s.suu + s.svv, 0.0f));
}

inline SigmaF AddIsotropicRidge(const SigmaF &s, float ridge_px) {
  SigmaF out = s;
  const float r2 = ridge_px * ridge_px;
  out.suu += r2;
  out.svv += r2;
  return out;
}

// rᵀ Σ⁻¹ r. Ridge should already be in s so a zero-Σ pixel still has a floor.
inline float MahalanobisSq(float ru, float rv, const SigmaF &s) {
  const float det = s.suu * s.svv - s.suv * s.suv;
  if (det <= kFerrEps) {
    const float n2 = ru * ru + rv * rv;
    const float floor2 = kMinFerrPx * kMinFerrPx;
    return (floor2 > 0.0f) ? (n2 / floor2) : n2;
  }
  return (s.svv * ru * ru - 2.0f * s.suv * ru * rv + s.suu * rv * rv) / det;
}

} // namespace sio_geo_flow
