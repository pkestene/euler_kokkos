#ifndef BLAST_PARAMS_H_
#define BLAST_PARAMS_H_

#include "utils/config/ConfigMap.h"

struct BlastParams {

  // blast problem parameters
  real_t blast_radius;
  real_t blast_center_x;
  real_t blast_center_y;
  real_t blast_center_z;
  real_t blast_density_in;
  real_t blast_density_out;
  real_t blast_pressure_in;
  real_t blast_pressure_out;

  BlastParams(ConfigMap& configMap)
  {

    double xmin = configMap.getFloat("mesh", "xmin", 0.0);
    double ymin = configMap.getFloat("mesh", "ymin", 0.0);
    double zmin = configMap.getFloat("mesh", "zmin", 0.0);

    double xmax = configMap.getFloat("mesh", "xmax", 1.0);
    double ymax = configMap.getFloat("mesh", "ymax", 1.0);
    double zmax = configMap.getFloat("mesh", "zmax", 1.0);
    
    blast_radius   = configMap.getFloat("blast","radius", (xmin+xmax)/2.0/10);
    blast_center_x = configMap.getFloat("blast","center_x", (xmin+xmax)/2);
    blast_center_y = configMap.getFloat("blast","center_y", (ymin+ymax)/2);
    blast_center_z = configMap.getFloat("blast","center_z", (zmin+zmax)/2);
    blast_density_in  = configMap.getFloat("blast","density_in", 1.0);
    blast_density_out = configMap.getFloat("blast","density_out", 1.2);
    blast_pressure_in  = configMap.getFloat("blast","pressure_in", 10.0);
    blast_pressure_out = configMap.getFloat("blast","pressure_out", 0.1);
    
  }

}; // struct BlastParams

#endif // BLAST_PARAMS_H_
