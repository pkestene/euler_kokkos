#ifndef JET_PARAMS_H_
#define JET_PARAMS_H_

#include "utils/config/ConfigMap.h"

#include <math.h> // for M_PI

/**
 * A small structure to hold parameters passed to a Kokkos functor,
 * used in border condition routine for the Jet test case.
 *
 * ref:
 * "On positivity-preserving high order discontinuous Galerkin schemes for
 * compressible Euler equations on rectangular meshes", Xiangxiong Zhang, 
 * Chi-Wang Shu, Journal of Computational Physics, Volume 229, Issue 23,
 * 20 November 2010, Pages 8918-8934
 * http://www.sciencedirect.com/science/article/pii/S0021999110004535
 *
 */
struct JetParams {

  // jet hydro state
  real_t rho_jet; //! density of injected fluid (jet)
  real_t u_jet;   //! x velocity of the jet
  real_t v_jet;   //! y velocity of the jet
  real_t w_jet;   //! z velocity of the jet
  real_t p_jet;   //! pressure in the jet

  // bulk hydro state
  real_t rho_bulk; //! density of injected fluid (bulk)
  real_t u_bulk;   //! x velocity of the bulk
  real_t v_bulk;   //! y velocity of the bulk
  real_t w_bulk;   //! z velocity of the bulk
  real_t p_bulk;   //! pressure in the bulk
  
  //! conservative variables in the inflow region (jet)
  real_t rho1, rho_u1, rho_v1, rho_w1, e_tot1;
  
  //! conservative variable in the bulk
  real_t rho2, rho_u2, rho_v2, rho_w2, e_tot2;

  //! jet position center
  real_t pos_jet;

  //! jet width
  real_t width_jet;
  
  JetParams (ConfigMap& configMap)
  {
    real_t gamma0 = configMap.getFloat("hydro","gamma0", 5.0/3.0);

    // read jet parameters
    rho_jet = configMap.getFloat("jet", "rho_jet", 5.0);
    u_jet   = configMap.getFloat("jet", "u_jet", 800.0);
    v_jet   = configMap.getFloat("jet", "v_jet", 0.0);
    w_jet   = configMap.getFloat("jet", "w_jet", 0.0);
    p_jet   = configMap.getFloat("jet", "p_jet", 0.4127);

    rho1 = rho_jet;
    rho_u1 = rho1 * u_jet;
    rho_v1 = rho1 * v_jet;
    rho_w1 = rho1 * w_jet;
    e_tot1 = p_jet / (gamma0-1.0) +
      0.5 * rho1 * ( u_jet*u_jet +
		     v_jet*v_jet +
		     w_jet*w_jet );

    pos_jet = configMap.getFloat("jet", "pos_jet", 0.0);
    width_jet = configMap.getFloat("jet", "width_jet", 0.1);
    
    // read bulk (ambiant) region
    rho_bulk = configMap.getFloat("jet", "rho_bulk", 0.5);
    u_bulk   = configMap.getFloat("jet", "u_bulk", 0.0);
    v_bulk   = configMap.getFloat("jet", "v_bulk", 0.0);
    w_bulk   = configMap.getFloat("jet", "w_bulk", 0.0);
    p_bulk   = configMap.getFloat("jet", "p_bulk", 0.4127);
    
    rho2   = rho_bulk,
    rho_u2 = rho2 * u_bulk;
    rho_v2 = rho2 * v_bulk;
    rho_w2 = rho2 * w_bulk;
    e_tot2 = p_bulk / (gamma0-1.0) +
      0.5 * rho2 * ( u_bulk*u_bulk +
		     v_bulk*v_bulk +
		     w_bulk*w_bulk );

  } // JetParams constructor
  
}; // struct JetParams

#endif // JET_PARAMS_H_
