#ifndef ROTOR_PARAMS_H_
#define ROTOR_PARAMS_H_

#include <math.h>

#include "utils/config/ConfigMap.h"


struct RotorParams {

  // rotor problem parameters
  real_t r0, r1, u0, p0, b0;

  RotorParams(ConfigMap& configMap)
  {
    
    r0 = configMap.getFloat("rotor","r0",0.1);
    r1 = configMap.getFloat("rotor","r1",0.115);
    u0 = configMap.getFloat("rotor","u0",2.0);
    p0 = configMap.getFloat("rotor","p0",1.0);
    b0 = configMap.getFloat("rotor","b0",5.0/sqrt(4*M_PI));
    
  }

}; // struct RotorParams

#endif // ROTOR_PARAMS_H_
