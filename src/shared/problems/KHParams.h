#ifndef KELVIN_HELMHOLTZ_PARAMS_H_
#define KELVIN_HELMHOLTZ_PARAMS_H_

#include "utils/config/ConfigMap.h"

//#include <cstdlib> // for srand

/**
 * A small structure to hold parameters passed to a Kokkos functor,
 * for initializing the Kelvin-Helmholtz instability init condition.
 *
 * p_sine, p_sine_robertson and p_rand specifiy which type of perturbation is
 * used to seed the instability.
 */
struct KHParams {

  // Kelvin-Helmholtz problem parameters
  real_t d_in;  //! density in
  real_t d_out; //! density out
  real_t pressure;
  bool p_sine; //! sinus perturbation
  bool p_sine_rob; //! sinus perturbation "a la Robertson"
  bool p_rand; //! random perturbation

  real_t vflow_in;
  real_t vflow_out;

  int seed;
  real_t amplitude; //! perturbation amplitude
  real_t outer_size;
  real_t inner_size;

  // for sine perturbation "a la Robertson"
  int    mode;
  real_t w0;
  real_t delta;
  
  KHParams(ConfigMap& configMap)
  {

    d_in  = configMap.getFloat("KH", "d_in", 1.0);
    d_out = configMap.getFloat("KH", "d_out", 1.0);

    pressure = configMap.getFloat("KH", "pressure", 10.0);

    p_sine     = configMap.getBool("KH", "perturbation_sine", false);
    p_sine_rob = configMap.getBool("KH", "perturbation_sine_robertson", true);
    p_rand     = configMap.getBool("KH", "perturbation_rand", false);

    vflow_in  = configMap.getFloat("KH", "vflow_in",  -0.5);
    vflow_out = configMap.getFloat("KH", "vflow_out",  0.5);

    if (p_rand) {
      // choose a different random seed per mpi rank
      seed = configMap.getInteger("KH", "rand_seed", 12);

#ifdef USE_MPI
      //srand( seed * (mpiRank+1) );

      // get MPI rank in MPI_COMM_WORLD
      // TODO : pass communicator to the constructor (?)
      int mpiRank = 1;
      MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
      seed *= (mpiRank+1);
#endif // USE_MPI
      
    }

    amplitude = configMap.getFloat("KH", "amplitude", 0.1);


    if (p_sine_rob or p_sine) {
      
      // perturbation mode number
      inner_size = configMap.getFloat("KH","inner_size", 0.2);

      mode       = configMap.getInteger("KH", "mode", 2);
      w0         = configMap.getFloat("KH", "w0", 0.1);
      delta      = configMap.getFloat("KH", "delta", 0.03);
    }
    
    
  } // KHParams

}; // struct KHParams

#endif // KELVIN_HELMHOLTZ_PARAMS_H_
