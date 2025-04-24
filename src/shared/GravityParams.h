/**
 * \file GravityParams.h
 */
#ifndef EULER_KOKKOS_SHARED_GRAVITY_PARAMS_H_
#define EULER_KOKKOS_SHARED_GRAVITY_PARAMS_H_

#include <utils/config/ConfigMap.h>
#include <shared/real_type.h>

namespace euler_kokkos
{

/**
 * Constant uniform gravity parameters.
 *
 * Well-balanced gravity MUSCL reconstruction is adapted from:
 * "Hydrostatic equilibrium preservation in MHD numerical simulation with stratified atmospheres.
 * Explicit Godunov-type schemes with MUSCL reconstruction", G. Krause, A&A 631 (2019), p. A68.
 * https://doi.org/10.1051/0004-6361/201936387
 */
struct GravityParams
{
  //! turn on/off gravity at run run-time
  bool enabled;

  //! enable Hancock predictor (no well-balanced gravity)
  bool hancock_predictor_enabled;

  //! enable well-balanced gravity in MUSCL reconstruction
  bool well_balanced_reconstruction_enabled;

  GravityParams(ConfigMap const & configMap)
    : enabled(configMap.getBool("gravity", "enabled", false))
    , hancock_predictor_enabled(configMap.getBool("gravity", "hancock_predictor_enabled", false))
    , well_balanced_reconstruction_enabled(
        configMap.getBool("gravity", "well_balanced_reconstruction_enabled", false))
  {}

}; // struct GravityParams

} // namespace euler_kokkos

#endif // EULER_KOKKOS_SHARED_GRAVITY_PARAMS_H_
