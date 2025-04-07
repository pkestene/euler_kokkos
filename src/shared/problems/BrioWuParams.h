#ifndef BRIOWU_PARAMS_H_
#define BRIOWU_PARAMS_H_

#include "utils/config/ConfigMap.h"

namespace euler_kokkos
{

///
/// Brio-Wu shock tube problem.
///
/// Reference:
/// - https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node192.html
/// - https://www.astro.princeton.edu/~jstone/Athena/tests/brio-wu/Brio-Wu.html
struct BrioWuParams
{

  enum Direction : int
  {
    X = 0,
    Y = 1,
    Z = 2
  };

  // Brio-Wu problem parameters (left and right state)
  real_t rhoL;
  real_t pL;
  real_t uL;
  real_t ByL;

  real_t rhoR;
  real_t pR;
  real_t uR;
  real_t ByR;

  real_t Bx;

  Direction direction;

  real_t xd; // discontinuity location

  BrioWuParams(ConfigMap & configMap)
  {
    rhoL = configMap.getFloat("brio-wu", "rhoL", 1.0);
    pL = configMap.getFloat("brio-wu", "pL", 1.0);
    uL = configMap.getFloat("brio-wu", "uL", 0.0);
    ByL = configMap.getFloat("brio-wu", "ByL", 1.0);

    rhoR = configMap.getFloat("brio-wu", "rhoR", 0.125);
    pR = configMap.getFloat("brio-wu", "pR", 0.1);
    uR = configMap.getFloat("brio-wu", "uR", 0.0);
    ByR = configMap.getFloat("brio-wu", "ByR", -1.0);

    Bx = configMap.getFloat("brio-wu", "Bx", 0.75);

    direction = static_cast<Direction>(configMap.getInteger("brio-wu", "direction", Direction::X));

    if (direction == Direction::X)
    {
      const auto xmin = configMap.getFloat("mesh", "xmin", 0.0);
      const auto xmax = configMap.getFloat("mesh", "xmax", 1.0);
      xd = configMap.getFloat("brio-wu", "xd", (xmin + xmax) / 2.0);
    }
    else if (direction == Direction::Y)
    {
      const auto ymin = configMap.getFloat("mesh", "ymin", 0.0);
      const auto ymax = configMap.getFloat("mesh", "ymax", 1.0);
      xd = configMap.getFloat("brio-wu", "xd", (ymin + ymax) / 2.0);
    }
    else
    {
      const auto zmin = configMap.getFloat("mesh", "zmin", 0.0);
      const auto zmax = configMap.getFloat("mesh", "zmax", 1.0);
      xd = configMap.getFloat("brio-wu", "xd", (zmin + zmax) / 2.0);
    }
  }

}; // struct BrioWuParams

} // namespace euler_kokkos

#endif // BRIOWU_PARAMS_H_
