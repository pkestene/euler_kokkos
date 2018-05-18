#ifndef RAYLEIGH_TAYLOR_INSTABILITY_PARAMS_H_
#define RAYLEIGH_TAYLOR_INSTABILITY_PARAMS_H_

#include "utils/config/ConfigMap.h"

/**
 * Rayleigh-Taylor test parameters.
 */
struct RayleighTaylorInstabilityParams {

  // rayleigh taylorinstability test
  real_t amplitude;
  real_t d0,d1; //density below / above
  real_t gx, gy, gz; // uniform initial gravity field
  real_t bx, by, bz; // uniform initial magnetic field

  
  RayleighTaylorInstabilityParams(ConfigMap& configMap)
  {
    
    amplitude  = configMap.getFloat("rayleigh_taylor","amplitude", 0.01);
    d0  = configMap.getFloat("rayleigh_taylor","d0", 1.0);
    d1  = configMap.getFloat("rayleigh_taylor","d1", 2.0);

    gx  = configMap.getFloat("rayleigh_taylor","gx",  0.0);
    gy  = configMap.getFloat("rayleigh_taylor","gy", -0.1);
    gz  = configMap.getFloat("rayleigh_taylor","gz",  0.0);
    
    bx  = configMap.getFloat("rayleigh_taylor","bx", 0.0);
    by  = configMap.getFloat("rayleigh_taylor","by", 0.0);
    bz  = configMap.getFloat("rayleigh_taylor","bz", 0.0);

  }

}; // struct RayleighTaylorInstabilityParams

#endif // RAYLEIGH_TAYLOR_INSTABILITY_PARAMS_H_
