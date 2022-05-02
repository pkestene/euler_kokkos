#ifndef EULER_EQUATIONS_H_
#define EULER_EQUATIONS_H_

#include <map>
#include <array>

#include "shared/real_type.h"
#include "shared/enums.h"
#include "shared/HydroState.h"

namespace euler_kokkos {

/**
 * This structure gather useful information (variable names,
 * flux functions, ...) for the compressible Euler equations system
 * in both 2D / 3D.
 *
 * Inspired by code dflo (https://github.com/cpraveen/dflo)
 */
template<int dim>
struct EulerEquations {};

/**
 * 2D specialization of the Euler Equation system.
 */
template <>
struct EulerEquations<2>
{

  //! numeric constant 1/2
  static constexpr real_t ONE_HALF = 0.5;

  //! small pressure safe-guard
  static constexpr real_t smallp = 1e-7;

  //! number of variables: density(1) + energy(1) + momentum(2)
  static const int nbvar = 2+2;

  //! type alias to a small array holding hydrodynamics state variables
  using HydroState = HydroState2d;

  //! enum
  // static const int ID = 0; // density
  // static const int IP = 1; // Pressure (when used in primitive variables)
  // static const int IE = 1; // Energy
  // static const int IU = 2; // momentum along X
  // static const int IV = 3; // momentum along Y

  //! velocity gradient tensor number of components
  static const int nbvar_grad = 2*2;

  enum gradient_index_t {

    U_X = (int) gradientV_IDS_2d::U_X,
    U_Y = (int) gradientV_IDS_2d::U_Y,

    V_X = (int) gradientV_IDS_2d::V_X,
    V_Y = (int) gradientV_IDS_2d::V_Y

  };

  //! alias typename to an array holding gradient velocity tensor components
  using GradTensor = Kokkos::Array<real_t,nbvar_grad>;

  //! just a dim-dimension vector
  using Vector = Kokkos::Array<real_t,2>;

  //! variables names as a std::map
  static std::map<int, std::string>
  get_variable_names()
  {

    std::map<int, std::string> names;

    names[ID] = "rho";
    names[IP] = "energy";
    names[IU] = "rho_vx"; // momentum component X
    names[IV] = "rho_vy"; // momentum component Y

    return names;

  } // get_variable_names

  /**
   * Compute pressure.
   *
   * \param[in] q vector of conservative variables.
   * \param[in] gamma0
   *
   * \return pressure
   */
  static
  KOKKOS_INLINE_FUNCTION
  real_t compute_pressure(const HydroState& q, real_t gamma0)
  {

    // 0.5 * rho * (u^2+v^2)
    real_t ekin = 0.5 * (q[IU]*q[IU] + q[IV]*q[IV]) / q[ID];

    // pressure
    real_t pressure = (gamma0-1.0)*(q[IE] - ekin);

    return pressure > smallp*q[ID] ? pressure : smallp*q[ID];

  } // compute_pressure

  /**
   * Compute speed of sound.
   *
   * \param[in] vector of primitive variables
   * \param[in] gamma0
   *
   * \return speed of sound
   */
  static
  KOKKOS_INLINE_FUNCTION
  real_t compute_speed_of_sound(const HydroState& w, real_t gamma0)
  {

    return w[IP] * gamma0 / w[ID];

  } // compute_speed_of_sound

  /**
   * Convert from conservative to primitive variables.
   *
   * \param[in] q vector of conservative variables.
   * \param[out] q vector of primitive variables.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void convert_to_primitive(const HydroState& q,
			    HydroState&       w,
			    real_t            gamma0)
  {

    const real_t rho = fmax(q[ID], 1e-8);

    // 0.5 * rho * (u^2+v^2)
    const real_t ekin = 0.5 * (q[IU]*q[IU] + q[IV]*q[IV]) / rho;

    // pressure
    real_t pressure = (gamma0-1.0)*(q[IE] - ekin);

    pressure = fmax( pressure ,  smallp*rho);

    w[ID] = q[ID];
    w[IU] = q[IU]/q[ID];
    w[IV] = q[IV]/q[ID];
    w[IP] = pressure;

  } // convert_to_primitive

  /**
   * Flux expression in the Euler equations system written in conservative
   * form along direction X.
   *
   * \param[in] q conservative variables \f$ (\rho, \rho u, \rho v, E) \f$
   * \param[in] p is pressure
   * \param[out] flux vector \f$ (\rho u, \rho u^2+p, \rho u v, u(E+p) ) \f$
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_x(const HydroState& q, real_t p, HydroState& flux)
  {
    flux[ID] = q[IU];                 // rho u
    flux[IU] = q[IU]*q[IU]/q[ID]+p;   // rho u^2 + p
    flux[IV] = q[IU]*q[IV]/q[ID];     // rho u v
    flux[IE] = q[IU]/q[ID]*(q[IE]+p); // u (E+p)
  };

  /**
   * Flux expression in the Euler equations system written in conservative
   * form along direction Y.
   *
   * \param[in] q conservative variables \f$ (\rho, \rho u, \rho v, E) \f$
   * \param[in] p is pressure
   * \param[out] flux vector \f$ (\rho v, \rho v u, \rho v^2+p, v(E+p) ) \f$
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_y(const HydroState& q, real_t p, HydroState& flux)
  {
    flux[ID] = q[IV];                 // rho v
    flux[IU] = q[IV]*q[IU]/q[ID];     // rho v u
    flux[IV] = q[IV]*q[IV]/q[ID]+p;   // rho v^2 + p
    flux[IE] = q[IV]/q[ID]*(q[IE]+p); // v (E+p)
  };

  /**
   * Viscous term as a flux along direction X.
   *
   * \param[in] g is the velocity gradient tensor
   * \param[in] v is the hydrodynamics velocity vector
   * \param[in] f is a vector (gradient of diffusive term)
   * \param[in] mu is dynamics viscosity (mu = rho * nu)
   * \param[out]
   *
   * note that the diffusive term f represents thermal + entropy diffusion
   * as in ASH / CHORUS code.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_visc_x(const GradTensor& g,
		   const Vector& v,
		   const Vector& f,
		   real_t mu,
		   HydroState& flux)
  {
    real_t tau_xx = 2*mu*(g[U_X]-ONE_HALF*(g[U_X]+g[V_Y]));
    real_t tau_xy =   mu*(g[V_X] + g[U_Y]);

    flux[ID] = 0.0;
    flux[IU] = tau_xx;
    flux[IV] = tau_xy;
    flux[IE] = v[IX]*tau_xx + v[IY]*tau_xy + f[IX];
  };

  /**
   * Viscous term as a flux along direction Y.
   *
   * \param[in] g is the velocity gradient tensor
   * \param[in] v is the hydrodynamics velocity vector
   * \param[in] f is a vector (gradient of diffusive term)
   * \param[in] mu is dynamics viscosity (mu = rho * nu)
   * \param[out]
   *
   * note that the diffusive term f represents thermal + entropy diffusion
   * as in ASH / CHORUS code.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_visc_y(const GradTensor& g,
		   const Vector& v,
		   const Vector& f,
		   real_t mu,
		   HydroState& flux)
  {
    real_t tau_yy = 2*mu*(g[V_Y]-ONE_HALF*(g[U_X]+g[V_Y]));
    real_t tau_xy =   mu*(g[V_X] + g[U_Y]);

    flux[ID] = 0.0;
    flux[IU] = tau_xy;
    flux[IV] = tau_yy;
    flux[IE] = v[IX]*tau_xy + v[IY]*tau_yy + f[IY];
  };

  /**
   * Compute characteristic variables by multiply input vector
   * by L (left eigenvalue matrix of Euler Jacobian). Computation is done in place
   *
   * The formulas defining eigen matrix are detailled in doc/euler/euler_equations.tex
   * and also copied here using python syntax
   *
   * # left eigenvectors (R^-1)
   * Lx = np.array([[beta*(phi2+u*c), -beta*(g1*u+c), -beta*g1*v, beta*g1],
   *                [1.0-phi2/c2, g1*u/c2, g1*v/c2, -g1/c2],
   *                [-v,0,1,0],
   *                [beta*(phi2-u*c), -beta*(g1*u-c), -beta*g1*v, beta*g1]])
   *
   * Ly = np.array([[beta*(phi2+v*c), -beta*g1*u, -beta*(g1*v+c), beta*g1],
   *                [1.0-phi2/c2, g1*u/c2, g1*v/c2, -g1/c2],
   *                [-u,1,0,0],
   *                [beta*(phi2-v*c), -beta*g1*u, -beta*(g1*v-c), beta*g1]])
   *
   * If \f$ \Lambda \f$ is the diagonal matrix with eigenvalues u-c, u, u, u+c, one has the
   * following eigen decomposition:
   * \f$ \Lambda = L  A(U)  R \f$
   * where \f$ A(U) = \partial F / \partial U \f$.
   *
   * \tparam dir allows to select which flux function / Jacobian to use : A(U) or B(U)
   *
   * \param[in,out] data input vector of conservative variables and characteristics var in output
   * \param[in] q local hydro state vector of conservative variables
   * \param[in] c speed of sound
   * \param[in] gamma0 is the heat capacity ratio
   * \param[out] out output vector of characteristics variables
   */
  template<int dir>
  static
  KOKKOS_INLINE_FUNCTION
  void cons_to_charac(HydroState& data,
		      const HydroState& q,
		      const real_t c,
		      const real_t gamma0)
  {
    HydroState tmp;

    // some useful intermediate values
    const real_t u = q[IU]/q[ID];
    const real_t v = q[IV]/q[ID];
    const real_t c2 = c*c;

    const real_t g1 = gamma0-1.0;

    const real_t beta = 1.0/2/c2;

    const real_t V2=u*u+v*v;

    // enthalpy
    const real_t H = 0.5*V2 + c2/g1;

    // also equal to g1*V2
    const real_t phi2 = g1*H-c2;

    if (dir == IX) {

      // compute matrix vector multiply: tmp = Lx . data
      tmp[ID] = data[ID]*beta*(phi2+u*c) + data[IU]*(-beta*(g1*u+c)) + data[IV]*(-beta*g1*v) + data[IE]*beta*g1;
      tmp[IU] = data[ID]*(1.0-phi2/c2)   + data[IU]*g1*u/c2          + data[IV]*g1*v/c2      + data[IE]*(-g1/c2);
      tmp[IV] = data[ID]*(-v)                                      + data[IV];
      tmp[IE] = data[ID]*beta*(phi2-u*c) + data[IU]*(-beta*(g1*u-c)) + data[IV]*(-beta*g1*v) + data[IE]*beta*g1;

    } else if (dir == IY) {

      // compute matrix vector multiply: tmp = Ly . data
      tmp[ID] = data[ID]*beta*(phi2+v*c) + data[IU]*(-beta*g1*u) + data[IV]*(-beta*(g1*v+c)) + data[IE]*beta*g1;
      tmp[IU] = data[ID]*(1.0-phi2/c2)   + data[IU]*g1*u/c2      + data[IV]*g1*v/c2          + data[IE]*(-g1/c2);
      tmp[IV] = data[ID]*(-u)            + data[IU];
      tmp[IE] = data[ID]*beta*(phi2-v*c) + data[IU]*(-beta*g1*u) + data[IV]*(-beta*(g1*v-c)) + data[IE]*beta*g1;

    }

    data[ID] = tmp[ID];
    data[IE] = tmp[IE];
    data[IU] = tmp[IU];
    data[IV] = tmp[IV];

  } // cons_to_charac

  /**
   * Transform from characteristic variables to conservative by multiply input vector
   * by R (right eigenvalue matrix of Euler Jacobian). Computation is done in place
   *
   * The formulas defining eigen matrix are detailled in doc/euler/euler_equations.tex
   * and also copied here using python syntax
   *
   * # right eigenvectors
   * Rx = np.array([[1,    1,    0, 1],
   *                [u-c,  u,    0, u+c],
   *                [v,    v,    1, v],
   *                [H-u*c,V2/2, v, H+u*c]])
   *
   * Ry = np.array([[1,    1,    0, 1],
   *                [u,    u,    1, u],
   *                [v-c,  v,    0, v+c],
   *                [H-v*c,V2/2, u, H+v*c]])
   *
   *
   *
   * If \f$ \Lambda \f$ is the diagonal matrix with eigenvalues u-c, u, u, u+c, one has the
   * following eigen decomposition:
   * \f$ \Lambda = L  A(U)  R \f$
   * where \f$ A(U) = \partial F / \partial U \f$.
   *
   * \tparam dir allows to select which flux function / Jacobian to use : A(U) or B(U)
   *
   * \param[in,out] on input vector of characteristics variables, on output conservative var
   * \param[in] q local hydro state vector of conservative variables
   * \param[in] c speed of sound
   * \param[in] gamma0 is the heat capacity ratio
   */
  template<int dir>
  static
  KOKKOS_INLINE_FUNCTION
  void charac_to_cons(HydroState& data,
		      const HydroState& q,
		      const real_t c,
		      const real_t gamma0)
  {

    HydroState tmp;

    // some useful intermediate values
    const real_t u = q[IU]/q[ID];
    const real_t v = q[IV]/q[ID];
    const real_t c2 = c*c;

    const real_t g1 = gamma0-1.0;

    //const real_t beta = 1.0/2/c2;

    const real_t V2=u*u+v*v;

    // enthalpy
    const real_t H = 0.5*V2 + c2/g1;

    // also equal to g1*V2
    //const real_t phi2 = g1*H-c2;

    if (dir == IX) {

      // compute matrix vector multiply: tmp = Rx . data
      tmp[ID] = data[ID]         + data[IU]                   + data[IE];
      tmp[IU] = data[ID]*(u-c)   + data[IU]*u                 + data[IE]*(u+c);
      tmp[IV] = data[ID]*v       + data[IU]*v    + data[IV]   + data[IE]*v;
      tmp[IE] = data[ID]*(H-u*c) + data[IU]*V2/2 + data[IV]*v + data[IE]*(H+u*c);

    } else if (dir == IY) {

      // compute matrix vector multiply: tmp = Ry . data
      tmp[ID] = data[ID]         + data[IU]                   + data[IE];
      tmp[IU] = data[ID]*u       + data[IU]*u    + data[IV]   + data[IE]*u;
      tmp[IV] = data[ID]*(v-c)   + data[IU]*v                 + data[IE]*(v+c);
      tmp[IE] = data[ID]*(H-v*c) + data[IU]*V2/2 + data[IV]*u + data[IE]*(H+v*c);

    }

    data[ID] = tmp[ID];
    data[IE] = tmp[IE];
    data[IU] = tmp[IU];
    data[IV] = tmp[IV];

  } // charac_to_cons

}; //struct EulerEquations<2>

/**
 * 3D specialization of the Euler Equation system.
 */
template <>
struct EulerEquations<3>
{

  //! numeric constant 1/3
  static constexpr real_t ONE_THIRD = 1.0/3;

  //! small pressure safe-guard
  static constexpr real_t smallp = 1e-7;

  //! number of variables: density(1) + energy(1) + momentum(3)
  static const int nbvar = 2+3;

  //! type alias to a small array holding hydrodynamics state variables
  using HydroState = HydroState3d;

  //! enum
  // enum varIDS {
  //   ID = 0, // density
  //   IP = 1, // Pressure (when used in primitive variables)
  //   IE = 1, // Energy
  //   IU = 2, // momentum along X
  //   IV = 3, // momentum along Y
  //   IW = 4, // momentum along Z
  // };

  //! velocity gradient tensor number of components
  static const int nbvar_grad = 3*3;

  enum gradient_index_t {

    U_X = (int) gradientV_IDS_3d::U_X,
    U_Y = (int) gradientV_IDS_3d::U_Y,
    U_Z = (int) gradientV_IDS_3d::U_Z,

    V_X = (int) gradientV_IDS_3d::V_X,
    V_Y = (int) gradientV_IDS_3d::V_Y,
    V_Z = (int) gradientV_IDS_3d::V_Z,

    W_X = (int) gradientV_IDS_3d::W_X,
    W_Y = (int) gradientV_IDS_3d::W_Y,
    W_Z = (int) gradientV_IDS_3d::W_Z

  };

  //! alias typename to an array holding gradient velocity tensor components
  using GradTensor = Kokkos::Array<real_t,nbvar_grad>;

  //! just a dim-dimension vector
  using Vector = Kokkos::Array<real_t,3>;

  //! variables names as a std::map
  static std::map<int, std::string>
  get_variable_names()
  {

    std::map<int, std::string> names;

    names[ID] = "rho";
    names[IE] = "energy";
    names[IU] = "rho_vx"; // momentum component X
    names[IV] = "rho_vy"; // momentum component Y
    names[IW] = "rho_vz"; // momentum component Z

    return names;

  } // get_variable_names

  /**
   * Compute pressure.
   *
   * \param[in] q vector of conservative variables.
   * \param[in] gamma0
   *
   * \return pressure
   */
  static
  KOKKOS_INLINE_FUNCTION
  real_t compute_pressure(const HydroState& q, real_t gamma0)
  {

    // 0.5 * rho * (u^2+v^2+w^2)
    real_t ekin = 0.5 * (q[IU]*q[IU] + q[IV]*q[IV] + q[IW]*q[IW]) / q[ID];

    // pressure
    real_t pressure = (gamma0-1.0)*(q[IE] - ekin);

    return pressure > smallp*q[ID] ? pressure : smallp*q[ID];

  } // compute_pressure

  /**
   * Compute speed of sound.
   *
   * \param[in] vector of primitive variables
   * \param[in] gamma0
   *
   * \return speed of sound
   */
  static
  KOKKOS_INLINE_FUNCTION
  real_t compute_speed_of_sound(const HydroState& w, real_t gamma0)
  {

    return w[IP] * gamma0 / w[ID];

  } // compute_speed_of_sound

  /**
   * Convert from conservative to primitive variables.
   *
   * \param[in] q vector of conservative variables.
   * \param[out] q vector of primitive variables.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void convert_to_primitive(const HydroState& q,
			    HydroState&       w,
			    real_t            gamma0)
  {

    // 0.5 * rho * (u^2+v^2+w^2)
    real_t ekin = 0.5 * (q[IU]*q[IU] + q[IV]*q[IV] + q[IW]*q[IW]) / q[ID];

    // pressure
    real_t pressure = (gamma0-1.0)*(q[IE] - ekin);

    pressure = fmax( pressure ,  smallp*q[ID]);

    w[ID] = q[ID];
    w[IU] = q[IU]/q[ID];
    w[IV] = q[IV]/q[ID];
    w[IW] = q[IW]/q[ID];
    w[IP] = pressure;

  } // convert_to_primitive

  /**
   * Flux expression in the Euler equations system written in conservative
   * form along direction X.
   *
   * \param[in] q conservative var \f$ (\rho, \rho u, \rho v, \rho w, E) \f$
   * \param[in] p is pressure
   * \param[out] flux \f$ (\rho u, \rho u^2+p, \rho u v, \rho u w, u(E+p) ) \f$
   *
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_x(const HydroState& q, real_t p, HydroState& flux)
  {
    real_t u = q[IU]/q[ID];

    flux[ID] =   q[IU];     //   rho u
    flux[IU] = u*q[IU]+p;   // u rho u + p
    flux[IV] = u*q[IV];     // u rho v
    flux[IW] = u*q[IW];     // u rho w
    flux[IE] = u*(q[IE]+p); // u (E+p)
  };

  /**
   * Flux expression in the Euler equations system written in conservative
   * form along direction Y.
   *
   * \param[in] q conservative var \f$ (\rho, \rho u, \rho v, \rho w, E) \f$
   * \param[in] p is pressure
   * \param[out] flux \f$ (\rho v, \rho v u, \rho v^2+p, \rho v w, v(E+p) ) \f$
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_y(const HydroState& q, real_t p, HydroState& flux)
  {
    real_t v = q[IV]/q[ID];

    flux[ID] =   q[IV];     //   rho v
    flux[IU] = v*q[IU];     // v rho u
    flux[IV] = v*q[IV]+p;   // v rho v + p
    flux[IW] = v*q[IW];     // v rho w
    flux[IE] = v*(q[IE]+p); // v (E+p)
  };

  /**
   * Flux expression in the Euler equations system written in conservative
   * form along direction Z.
   *
   * \param[in] q conservative var \f$ (\rho, \rho u, \rho v, \rho w, E) \f$
   * \param[in] p is pressure
   * \param[out] flux \f$ (\rho v, \rho v u, \rho v^2+p, \rho v w, v(E+p) ) \f$
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_z(const HydroState& q, real_t p, HydroState& flux)
  {
    real_t w = q[IW]/q[ID];

    flux[ID] =   q[IW];     //   rho w
    flux[IU] = w*q[IU];     // w rho u
    flux[IV] = w*q[IV];     // w rho v
    flux[IW] = w*q[IW]+p;   // w rho w + p
    flux[IE] = w*(q[IE]+p); // w (E+p)
  };

  /**
   * Viscous term as a flux along direction X.
   *
   * \param[in] g is the velocity gradient tensor
   * \param[in] v is the hydrodynamics velocity vector
   * \param[in] f is a vector (gradient of diffusive term)
   * \param[in] mu is dynamics viscosity (mu = rho * nu)
   * \param[out]
   *
   * note that the diffusive term f represents thermal + entropy diffusion
   * as in ASH / CHORUS code.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_visc_x(const GradTensor& g,
		   const Vector& v, const Vector& f,
		   real_t mu,
		   HydroState& flux)
  {
    real_t tau_xx = 2*mu*(g[U_X]-ONE_THIRD*(g[U_X]+g[V_Y]+g[W_Z]));
    real_t tau_yx =   mu*(g[V_X] + g[U_Y]);
    real_t tau_zx =   mu*(g[W_X] + g[U_Z]);

    flux[ID] = 0.0;
    flux[IU] = tau_xx;
    flux[IV] = tau_yx;
    flux[IW] = tau_zx;
    flux[IE] = v[IX]*tau_xx + v[IY]*tau_yx + v[IZ]*tau_zx + f[IX];
  };

  /**
   * Viscous term as a flux along direction Y.
   *
   * \param[in] g is the velocity gradient tensor
   * \param[in] v is the hydrodynamics velocity vector
   * \param[in] f is a vector (gradient of diffusive term)
   * \param[in] mu is dynamics viscosity (mu = rho * nu)
   * \param[out]
   *
   * note that the diffusive term f represents thermal + entropy diffusion
   * as in ASH / CHORUS code.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_visc_y(const GradTensor& g,
		   const Vector& v, const Vector& f,
		   real_t mu,
		   HydroState& flux)
  {
    real_t tau_xy =   mu*(g[U_Y] + g[V_X]);
    real_t tau_yy = 2*mu*(g[V_Y]-ONE_THIRD*(g[U_X]+g[V_Y]+g[W_Z]));
    real_t tau_zy =   mu*(g[W_Y] + g[V_Z]);

    flux[ID] = 0.0;
    flux[IU] = tau_xy;
    flux[IV] = tau_yy;
    flux[IW] = tau_zy;
    flux[IE] = v[IX]*tau_xy + v[IY]*tau_yy + v[IZ]*tau_zy + f[IY];
  };

  /**
   * Viscous term as a flux along direction Z.
   *
   * \param[in] g is the velocity gradient tensor
   * \param[in] v is the hydrodynamics velocity vector
   * \param[in] f is a vector (gradient of diffusive term)
   * \param[in] mu is dynamics viscosity (mu = rho * nu)
   * \param[out]
   *
   * note that the diffusive term f represents thermal + entropy diffusion
   * as in ASH / CHORUS code.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_visc_z(const GradTensor& g,
		   const Vector& v, const Vector& f,
		   real_t mu,
		   HydroState& flux)
  {
    real_t tau_xz =   mu*(g[U_Z] + g[W_X]);
    real_t tau_yz =   mu*(g[V_Z] + g[W_Y]);
    real_t tau_zz = 2*mu*(g[W_Z]-ONE_THIRD*(g[U_X]+g[V_Y]+g[W_Z]));

    flux[ID] = 0.0;
    flux[IU] = tau_xz;
    flux[IV] = tau_yz;
    flux[IW] = tau_zz;
    flux[IE] = v[IX]*tau_xz + v[IY]*tau_yz + v[IZ]*tau_zz + f[IZ];
  };

  /**
   * Compute characteristic variables by multiply input vector
   * by L (left eigenvalue matrix of Euler Jacobian). Computation is done in place.
   *
   * The formulas defining eigen matrix are detailled in doc/euler/euler_equations.tex
   * and also copied here using python syntax
   *
   * # left eigenvectors (R^-1)
   * Lx = np.array([[beta*(phi2+u*c), -beta*(g1*u+c), -beta*g1*v,    -beta*g1*w,     beta*g1],
   *                [1.0-phi2/c2,     g1*u/c2,        g1*v/c2,       g1*w/c2,        -g1/c2],
   *                [-v,              0,              1,             0,              0],
   *                [-w,              0,              0,             1,              0],
   *                [beta*(phi2-u*c), -beta*(g1*u-c), -beta*g1*v,    -beta*g1*w,     beta*g1]])
   *
   * Ly = np.array([[beta*(phi2+v*c), -beta*g1*u,    -beta*(g1*v+c), -beta*g1*w,     beta*g1],
   *                [1.0-phi2/c2,     g1*u/c2,       g1*v/c2,        g1*w/c2,        -g1/c2],
   *                [-u,              1,             0,              0,              0],
   *                [-w,              0,             0,              1,              0],
   *                [beta*(phi2-v*c), -beta*g1*u,    -beta*(g1*v-c), -beta*g1*w,     beta*g1]])
   *
   * Lz = np.array([[beta*(phi2+w*c), -beta*g1*u, -beta*g1*v,        -beta*(g1*w+c), beta*g1],
   *                [1.0-phi2/c2,     g1*u/c2,    g1*v/c2,           g1*w/c2,        -g1/c2],
   *                [-u,              1,          0,                 0,              0],
   *                [-v,              0,          1,                 0,              0],
   *                [beta*(phi2-w*c), -beta*g1*u, -beta*g1*v,        -beta*(g1*w-c), beta*g1]])
   *
   * If \f$ \Lambda \f$ is the diagonal matrix with eigenvalues u-c, u, u, u, u+c, one has the
   * following eigen decomposition:
   * \f$ \Lambda = L  A(U)  R \f$
   * where \f$ A(U) = \partial F / \partial U \f$.
   *
   * \tparam dir allows to select which flux function / Jacobian to use : Ax(U), Ay(U) or Az(U)
   *
   * \param[in,out] on input vector of conservative variables, on output characteristics var
   * \param[in] q local hydro state vector of conservative variables
   * \param[in] c speed of sound
   * \param[in] gamma0 is the heat capacity ratio
   */
  template<int dir>
  static
  KOKKOS_INLINE_FUNCTION
  void cons_to_charac(HydroState& data,
		      const HydroState& q,
		      const real_t c,
		      const real_t gamma0)
  {
    HydroState tmp;

    // some useful intermediate values
    const real_t u = q[IU]/q[ID];
    const real_t v = q[IV]/q[ID];
    const real_t w = q[IW]/q[ID];
    const real_t c2 = c*c;

    const real_t g1 = gamma0-1.0;

    const real_t beta = 1.0/2/c2;

    const real_t V2=u*u+v*v+w*w;

    // enthalpy
    const real_t H = 0.5*V2 + c2/g1;

    // also equal to g1*V2
    const real_t phi2 = g1*H-c2;

    if (dir == IX) {

      // compute matrix vector multiply: tmp = Lx . data
      tmp[ID] = data[ID]*beta*(phi2+u*c) + data[IU]*(-beta*(g1*u+c)) + data[IV]*(-beta*g1*v) + data[IW]*(-beta*g1*w) + data[IE]*beta*g1;
      tmp[IU] = data[ID]*(1.0-phi2/c2)   + data[IU]*g1*u/c2          + data[IV]*g1*v/c2      + data[IW]*g1*w/c2      + data[IE]*(-g1/c2);
      tmp[IV] = data[ID]*(-v)                                        + data[IV];
      tmp[IW] = data[ID]*(-w)                                                                + data[IW];
      tmp[IE] = data[ID]*beta*(phi2-u*c) + data[IU]*(-beta*(g1*u-c)) + data[IV]*(-beta*g1*v) + data[IW]*(-beta*g1*w) + data[IE]*beta*g1;

    } else if (dir == IY) {

      // compute matrix vector multiply: tmp = Ly . data
      tmp[ID] = data[ID]*beta*(phi2+v*c) + data[IU]*(-beta*g1*u) + data[IV]*(-beta*(g1*v+c)) + data[IW]*(-beta*g1*w) + data[IE]*beta*g1;
      tmp[IU] = data[ID]*(1.0-phi2/c2)   + data[IU]*g1*u/c2      + data[IV]*g1*v/c2          + data[IW]*g1*w/c2      + data[IE]*(-g1/c2);
      tmp[IV] = data[ID]*(-u)            + data[IU];
      tmp[IW] = data[ID]*(-w)                                                                + data[IW];
      tmp[IE] = data[ID]*beta*(phi2-v*c) + data[IU]*(-beta*g1*u) + data[IV]*(-beta*(g1*v-c)) + data[IW]*(-beta*g1*w) + data[IE]*beta*g1;

    } else if (dir == IZ) {

      // compute matrix vector multiply: tmp = Lz . data
      tmp[ID] = data[ID]*beta*(phi2+w*c) + data[IU]*(-beta*g1*u) + data[IV]*(-beta*g1*v)     + data[IW]*(-beta*(g1*w+c)) + data[IE]*beta*g1;
      tmp[IU] = data[ID]*(1.0-phi2/c2)   + data[IU]*g1*u/c2      + data[IV]*g1*v/c2          + data[IW]*g1*w/c2          + data[IE]*(-g1/c2);
      tmp[IV] = data[ID]*(-u)            + data[IU];
      tmp[IW] = data[ID]*(-v)                                    + data[IV];
      tmp[IE] = data[ID]*beta*(phi2-w*c) + data[IU]*(-beta*g1*u) + data[IV]*(-beta*g1*v)     + data[IW]*(-beta*(g1*w-c)) + data[IE]*beta*g1;

    }

    data[ID] = tmp[ID];
    data[IE] = tmp[IE];
    data[IU] = tmp[IU];
    data[IV] = tmp[IV];
    data[IW] = tmp[IW];

  } // cons_to_charac

  /**
   * Transform from characteristic variables to conservative by multiply input vector
   * by R (right eigenvalue matrix of Euler Jacobian). Computation done in place.
   *
   * The formulas defining eigen matrix are detailled in doc/euler/euler_equations.tex
   * and also copied here using python syntax
   *
   * # right eigenvectors
   * Rx = np.array([[1,    1,    0, 0, 1],
   *                [u-c,  u,    0, 0, u+c],
   *                [v,    v,    1, 0, v],
   *                [w,    w,    0, 1, w],
   *                [H-u*c,V2/2, v, w, H+u*c]])
   *
   * Ry = np.array([[1,    1,    0, 0, 1],
   *                [u,    u,    1, 0, u],
   *                [v-c,  v,    0, 0, v+c],
   *                [w,    w,    0, 1, w],
   *                [H-v*c,V2/2, u, w, H+v*c]])
   *
   * Rz = np.array([[1,    1,    0, 0, 1],
   *                [u,    u,    1, 0, u],
   *                [v  ,  v,    0, 1, v],
   *                [w-c,  w,    0, 0, w+c],
   *                [H-w*c,V2/2, u, v, H+w*c]])
   *
   *
   * If \f$ \Lambda \f$ is the diagonal matrix with eigenvalues u-c, u, u, u, u+c, one has the
   * following eigen decomposition:
   * \f$ \Lambda = L  A(U)  R \f$
   * where \f$ A(U) = \partial F / \partial U \f$.
   *
   * \tparam dir allows to select which flux function / Jacobian to use : Ax(U), Ay(U) or Az(U)
   *
   * \param[in,out] on input vector of characteristics variables, on output conservative vars
   * \param[in] q local hydro state vector of conservative variables
   * \param[in] c speed of sound
   * \param[in] gamma0 is the heat capacity ratio
   */
  template<int dir>
  static
  KOKKOS_INLINE_FUNCTION
  void charac_to_cons(HydroState& data,
		      const HydroState& q,
		      const real_t c,
		      const real_t gamma0)
  {
    HydroState tmp;

    // some useful intermediate values
    const real_t u = q[IU]/q[ID];
    const real_t v = q[IV]/q[ID];
    const real_t w = q[IW]/q[ID];
    const real_t c2 = c*c;

    const real_t g1 = gamma0-1.0;

    //const real_t beta = 1.0/2/c2;

    const real_t V2=u*u+v*v+w*w;

    // enthalpy
    const real_t H = 0.5*V2 + c2/g1;

    // also equal to g1*V2
    //const real_t phi2 = g1*H-c2;

    if (dir == IX) {

      // compute matrix vector multiply: tmp = Rx . data
      tmp[ID] = data[ID]         + data[IU]                                + data[IE];
      tmp[IU] = data[ID]*(u-c)   + data[IU]*u                              + data[IE]*(u+c);
      tmp[IV] = data[ID]*v       + data[IU]*v    + data[IV]                + data[IE]*v;
      tmp[IW] = data[ID]*w       + data[IU]*w                 + data[IW]   + data[IE]*w;
      tmp[IE] = data[ID]*(H-u*c) + data[IU]*V2/2 + data[IV]*v + data[IW]*w + data[IE]*(H+u*c);

    } else if (dir == IY) {

      // compute matrix vector multiply: tmp = Ry . data
      tmp[ID] = data[ID]         + data[IU]                                + data[IE];
      tmp[IU] = data[ID]*u       + data[IU]*u    + data[IV]                + data[IE]*u;
      tmp[IV] = data[ID]*(v-c)   + data[IU]*v                              + data[IE]*(v+c);
      tmp[IW] = data[ID]*w       + data[IU]*w                 + data[IW]   + data[IE]*w;
      tmp[IE] = data[ID]*(H-v*c) + data[IU]*V2/2 + data[IV]*u + data[IW]*w + data[IE]*(H+v*c);

    } else if (dir == IZ) {

      // compute matrix vector multiply: tmp = Rz . data
      tmp[ID] = data[ID]         + data[IU]                                + data[IE];
      tmp[IU] = data[ID]*u       + data[IU]*u    + data[IV]                + data[IE]*u;
      tmp[IV] = data[ID]*v       + data[IU]*v                 + data[IW]   + data[IE]*v;
      tmp[IW] = data[ID]*(w-c)   + data[IU]*w                              + data[IE]*(w+c);
      tmp[IE] = data[ID]*(H-w*c) + data[IU]*V2/2 + data[IV]*u + data[IW]*v + data[IE]*(H+w*c);

    }

    data[ID] = tmp[ID];
    data[IE] = tmp[IE];
    data[IU] = tmp[IU];
    data[IV] = tmp[IV];
    data[IW] = tmp[IW];

  } // charac_to_cons

}; //struct EulerEquations<3>

} // namespace euler_kokkos

#endif // EULER_EQUATIONS_H_
