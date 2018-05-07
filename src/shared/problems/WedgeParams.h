#ifndef WEDGE_PARAMS_H_
#define WEDGE_PARAMS_H_

#include "utils/config/ConfigMap.h"

#include <math.h> // for M_PI

/**
 * A small structure to hold parameters passed to a Kokkos functor,
 * used in border condition routine for the Wedge (Double Mach reflection) test.
 *
 * See http://amroc.sourceforge.net/examples/euler/2d/html/ramp_n.htm
 *
 * This border condition is time-dependent.
 */
struct WedgeParams {

  real_t x_f;     //! initial shock front position
  real_t angle_f; //! angle in radian of the shock front with x-axis
  real_t slope_f; //! slope dy/dx on the shock front

  real_t shock_speed; //! shock wave speed projected on x-axis

  real_t delta_x; //! how much the shock front has moved at time t

  //! conservative variable in the inflow region
  real_t rho1, e_tot1, rho_u1, rho_v1, rho_w1;
  
  //! conservative variable in the pre-shock region
  real_t rho2, e_tot2, rho_u2, rho_v2, rho_w2;
  
  WedgeParams (ConfigMap& configMap, real_t t)
  {
    real_t gamma0 = configMap.getFloat("hydro","gamma0", 1.4);

    x_f     = configMap.getFloat("wedge", "front_x", 0.1);
    angle_f = configMap.getFloat("wedge", "front_angle", M_PI/3.0);
    slope_f = tan(angle_f);

    shock_speed = configMap.getFloat("wedge", "shock_speed", 10.0);
    shock_speed /= cos(M_PI/2.0-angle_f);

    delta_x = shock_speed*t;

    // post-shock region
    rho1 = configMap.getFloat("wedge", "rho1", 8.0);

    real_t p1 = configMap.getFloat("wedge", "p1", 116.5);
    real_t u1 = configMap.getFloat("wedge", "u1", 8.25*cos(angle_f-M_PI/2.0));
    real_t v1 = configMap.getFloat("wedge", "v1", 8.25*sin(angle_f-M_PI/2.0));
    real_t w1 = configMap.getFloat("wedge", "w1",  0.0);

    rho_u1 = rho1 * u1;
    rho_v1 = rho1 * v1;
    rho_w1 = rho1 * w1;

    e_tot1 = p1 / (gamma0-1.0) +
      0.5 * rho1 * ( u1*u1 + v1*v1 + w1*w1 );

    // pre-shock region
    rho2 = configMap.getFloat("wedge", "rho2", 1.4);

    real_t p2 = configMap.getFloat("wedge", "p2", 1.0);
    real_t u2 = configMap.getFloat("wedge", "u2", 0.0);
    real_t v2 = configMap.getFloat("wedge", "v2", 0.0);
    real_t w2 = configMap.getFloat("wedge", "w2", 0.0);
    
    rho_u2 = rho2 * u2;
    rho_v2 = rho2 * v2;
    rho_w2 = rho2 * w2;

    e_tot2 = p2 / (gamma0-1.0) +
      0.5 * rho2 * ( u2*u2 + v2*v2 + w2*w2 );

  } // WedgeParams constructor
  
}; // struct WedgeParams

#endif // WEDGE_PARAMS_H_
