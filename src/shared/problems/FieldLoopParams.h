#ifndef FIELD_LOOP_PARAMS_H_
#define FIELD_LOOP_PARAMS_H_

#include <math.h>

#include "utils/config/ConfigMap.h"


struct FieldLoopParams {

  // field loop problem parameters
  real_t radius;
  real_t density_in;
  real_t amplitude;
  real_t vflow;
  double amp;
  int    seed;

  FieldLoopParams(ConfigMap& configMap)
  {
    
    radius    = configMap.getFloat  ("FieldLoop","radius"   ,  1.0);
    density_in= configMap.getFloat  ("FieldLoop","density_in", 1.0);
    amplitude = configMap.getFloat  ("FieldLoop","amplitude",  1.0);
    vflow     = configMap.getFloat  ("FieldLoop","vflow"    ,  1.0);
    amp       = configMap.getFloat  ("FieldLoop","amp",        0.01);
    seed      = configMap.getInteger("FieldLoop","seed",       0);
    
  }

}; // struct FieldLoopParams

#endif // FIELD_LOOP_PARAMS_H_
