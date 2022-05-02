/**
 * All possible Riemann solvers or so.
 */
#ifndef RIEMANN_SOLVERS_H_
#define RIEMANN_SOLVERS_H_

#include <math.h>

#include "HydroParams.h"
#include "HydroState.h"

namespace euler_kokkos {

/**
 * Compute cell fluxes from the Godunov state
 * \param[in]  qgdnv input primitive variables Godunov state
 * \param[out] flux  output flux vector
 */
template <class HydroState>
KOKKOS_INLINE_FUNCTION
void cmpflx(const HydroState& qgdnv,
	    HydroState& flux,
	    const HydroParams& params)
{
  real_t gamma0 = params.settings.gamma0;

  // Compute fluxes
  // Mass density
  flux[ID] = qgdnv[ID] * qgdnv[IU];

  // Normal momentum
  flux[IU] = flux[ID] * qgdnv[IU] + qgdnv[IP];

  // Transverse momentum 1
  flux[IV] = flux[ID] * qgdnv[IV];

  if (std::is_same<HydroState,HydroState3d>::value)
    flux[IW] = flux[ID] * qgdnv[IW];

  // Total energy
  real_t entho = ONE_F / (gamma0 - ONE_F);
  real_t ekin;
  ekin = HALF_F * qgdnv[ID] * (qgdnv[IU]*qgdnv[IU] +
			       qgdnv[IV]*qgdnv[IV]);
  if (std::is_same<HydroState,HydroState3d>::value)
    ekin += HALF_F * qgdnv[ID] * (qgdnv[IW]*qgdnv[IW]);

  real_t etot = qgdnv[IP] * entho + ekin;
  flux[IP] = qgdnv[IU] * (etot + qgdnv[IP]);

} // cmpflx

/**
 * Riemann solver, equivalent to riemann_approx in RAMSES (see file
 * godunov_utils.f90 in RAMSES).
 *
 * @param[in] qleft  : input left  state (primitive variables)
 * @param[in] qright : input right state (primitive variables)
 * @param[out] qgdnv : output Godunov state
 * @param[out] flux  : output flux
 */
template<class HydroState>
KOKKOS_INLINE_FUNCTION
void riemann_approx(const HydroState& qleft,
		    const HydroState& qright,
		    HydroState& qgdnv,
		    HydroState& flux,
		    const HydroParams& params)
{
  real_t gamma0  = params.settings.gamma0;
  real_t gamma6  = params.settings.gamma6;
  real_t smallr  = params.settings.smallr;
  real_t smallc  = params.settings.smallc;
  real_t smallp  = params.settings.smallp;
  real_t smallpp = params.settings.smallpp;

  // Pressure, density and velocity
  real_t rl = fmax(qleft [ID], smallr);
  real_t ul =      qleft [IU];
  real_t pl = fmax(qleft [IP], rl*smallp);
  real_t rr = fmax(qright[ID], smallr);
  real_t ur =      qright[IU];
  real_t pr = fmax(qright[IP], rr*smallp);

  // Lagrangian sound speed
  real_t cl = gamma0*pl*rl;
  real_t cr = gamma0*pr*rr;

  // First guess
  real_t wl = SQRT(cl);
  real_t wr = SQRT(cr);
  real_t pstar = fmax(((wr*pl+wl*pr)+wl*wr*(ul-ur))/(wl+wr), (real_t) ZERO_F);
  real_t pold = pstar;
  real_t conv = ONE_F;

  // Newton-Raphson iterations to find pstar at the required accuracy
  for(int iter = 0; (iter < 10 /*niter_riemann*/) && (conv > 1e-6); ++iter)
    {
      real_t wwl = SQRT(cl*(ONE_F+gamma6*(pold-pl)/pl));
      real_t wwr = SQRT(cr*(ONE_F+gamma6*(pold-pr)/pr));
      real_t ql = 2.0f*wwl*wwl*wwl/(wwl*wwl+cl);
      real_t qr = 2.0f*wwr*wwr*wwr/(wwr*wwr+cr);
      real_t usl = ul-(pold-pl)/wwl;
      real_t usr = ur+(pold-pr)/wwr;
      real_t delp = fmax(qr*ql/(qr+ql)*(usl-usr),-pold);

      pold = pold+delp;
      conv = FABS(delp/(pold+smallpp));	 // Convergence indicator
    }

  // Star region pressure
  // for a two-shock Riemann problem
  pstar = pold;
  wl = SQRT(cl*(ONE_F+gamma6*(pstar-pl)/pl));
  wr = SQRT(cr*(ONE_F+gamma6*(pstar-pr)/pr));

  // Star region velocity
  // for a two shock Riemann problem
  real_t ustar = HALF_F * (ul + (pl-pstar)/wl + ur - (pr-pstar)/wr);

  // Left going or right going contact wave
  real_t sgnm = COPYSIGN(ONE_F, ustar);

  // Left or right unperturbed state
  real_t ro, uo, po, wo;
  if(sgnm > ZERO_F)
    {
      ro = rl;
      uo = ul;
      po = pl;
      wo = wl;
    }
  else
    {
      ro = rr;
      uo = ur;
      po = pr;
      wo = wr;
    }
  real_t co = fmax(smallc, SQRT(FABS(gamma0*po/ro)));

  // Star region density (Shock, fmax prevents vacuum formation in star region)
  real_t rstar = fmax((real_t) (ro/(ONE_F+ro*(po-pstar)/(wo*wo))), (real_t) (smallr));
  // Star region sound speed
  real_t cstar = fmax(smallc, SQRT(FABS(gamma0*pstar/rstar)));

  // Compute rarefaction head and tail speed
  real_t spout  = co    - sgnm*uo;
  real_t spin   = cstar - sgnm*ustar;
  // Compute shock speed
  real_t ushock = wo/ro - sgnm*uo;

  if(pstar >= po)
    {
      spin  = ushock;
      spout = ushock;
    }

  // Sample the solution at x/t=0
  real_t scr = fmax(spout-spin, smallc+FABS(spout+spin));
  real_t frac = HALF_F * (ONE_F + (spout + spin)/scr);

  if (frac != frac) /* Not a Number */
    frac = 0.0;
  else
    frac = frac >= 1.0 ? 1.0 : frac <= 0.0 ? 0.0 : frac;

  qgdnv[ID] = frac*rstar + (ONE_F-frac)*ro;
  qgdnv[IU] = frac*ustar + (ONE_F-frac)*uo;
  qgdnv[IP] = frac*pstar + (ONE_F-frac)*po;

  if(spout < ZERO_F)
    {
      qgdnv[ID] = ro;
      qgdnv[IU] = uo;
      qgdnv[IP] = po;
    }

  if(spin > ZERO_F)
    {
      qgdnv[ID] = rstar;
      qgdnv[IU] = ustar;
      qgdnv[IP] = pstar;
    }

  // transverse velocity
  if(sgnm > ZERO_F)
    {
      qgdnv[IV] = qleft[IV];
      if (std::is_same<HydroState,HydroState3d>::value)
	qgdnv[IW] = qleft[IW];
    }
  else
    {
      qgdnv[IV] = qright[IV];
      if (std::is_same<HydroState,HydroState3d>::value)
	qgdnv[IW] = qright[IW];
    }

  cmpflx<HydroState>(qgdnv, flux, params);

} // riemann_approx

/**
 * Riemann solver, equivalent to riemann_llf in RAMSES (see file
 * godunov_utils.f90 in RAMSES).
 *
 * LLF = Local Lax-Friedrich.
 *
 * Reference : E.F. Toro, Riemann solvers and numerical methods for
 * fluid dynamics, Springer, chapter 10 (The HLL and HLLC Riemann solver).
 * See section 10.5.1, equation 10.43 which gives the expression of S+ denoted here
 * as cmax.
 *
 * @param[in] qleft  : input left  state (primitive variables)
 * @param[in] qright : input right state (primitive variables)
 * @param[out] qgdnv : output Godunov state
 * @param[out] flux  : output flux
 */
template<class HydroState>
KOKKOS_INLINE_FUNCTION
void riemann_llf(const HydroState& qleft,
		 const HydroState& qright,
		 HydroState& qgdnv,
		 HydroState& flux,
		 const HydroParams& params)
{

  // 1D LLF Riemann solver

  // constants
  real_t gamma0 = params.settings.gamma0;
  real_t smallr = params.settings.smallr;
  real_t smallp = params.settings.smallp;

  const real_t entho = ONE_F / (gamma0 - ONE_F);

  //============================
  // Compute maximum wave speed
  //============================
  real_t rl=FMAX(qleft [ID],smallr);
  real_t ul=     qleft [IU];
  real_t pl=FMAX(qleft [IP],rl*smallp);

  real_t rr=FMAX(qright[ID],smallr);
  real_t ur=     qright[IU];
  real_t pr=FMAX(qright[IP],rr*smallp);

  real_t cl= SQRT(gamma0*pl/rl);
  real_t cr= SQRT(gamma0*pr/rr);

  real_t cmax = FMAX(FABS(ul)+cl,FABS(ur)+cr);

  // Compute average velocity
  qgdnv[IU] = HALF_F*(qleft[IU]+qright[IU]);

  //================================
  // Compute conservative variables
  //================================
  HydroState uleft, uright;
  // mass density
  uleft [ID] = qleft [ID];
  uright[ID] = qright[ID];

  // total energy
  uleft [IP] = qleft [IP]*entho + HALF_F*qleft [ID]*qleft [IU]*qleft [IU];
  uright[IP] = qright[IP]*entho + HALF_F*qright[ID]*qright[IU]*qright[IU];

  uleft [IP] += HALF_F*qleft [ID]*qleft [IV]*qleft [IV];
  uright[IP] += HALF_F*qright[ID]*qright[IV]*qright[IV];

  if (std::is_same<HydroState,HydroState3d>::value) {
    uleft [IP] += HALF_F*qleft [ID]*qleft [IW]*qleft [IW];
    uright[IP] += HALF_F*qright[ID]*qright[IW]*qright[IW];
  }

  // normal momentum
  uleft [IU] = qleft [ID]*qleft [IU];
  uright[IU] = qright[ID]*qright[IU];

  // transverse momentum
  uleft [IV] = qleft [ID]*qleft [IV];
  uright[IV] = qright[ID]*qright[IV];

  if (std::is_same<HydroState,HydroState3d>::value) {
    uleft [IW] = qleft [ID]*qleft [IW];
    uright[IW] = qright[ID]*qright[IW];
  }

  //===============================
  // Compute left and right fluxes
  //===============================
  HydroState fleft, fright;
  // mass density
  fleft [ID] = uleft [ID] * qleft [IU];
  fright[ID] = uright[ID] * qright[IU];

  // total energy
  fleft [IP] = qleft [IU] * ( uleft [IP] + qleft [IP]);
  fright[IP] = qright[IU] * ( uright[IP] + qright[IP]);

  // normal momentum
  fleft [IU] = qleft [IP] +   uleft [IU] * qleft [IU];
  fright[IU] = qright[IP] +   uright[IU] * qright[IU];

  // transverse momentum
  fleft [IV] = fleft [ID] * qleft [IV];
  fright[IV] = fright[ID] * qright[IV];

  if (std::is_same<HydroState,HydroState3d>::value) {
    fleft [IW] = fleft [ID] * qleft [IW];
    fright[IW] = fright[ID] * qright[IW];
  }

  //==============================
  // Compute Lax-Friedrich fluxes
  //==============================
  for (int nVar=0; nVar < HYDRO_2D_NBVAR; nVar++) {
    flux[nVar] = HALF_F * ( fleft[nVar] + fright[nVar] - cmax*(uright[nVar] - uleft[nVar]) );
  }
  if (std::is_same<HydroState,HydroState3d>::value) {
    flux[IW] = HALF_F * ( fleft[IW] + fright[IW] - cmax*(uright[IW] - uleft[IW]) );
  }

} // riemann_llf

/**
 * Riemann solver, equivalent to riemann_hll in RAMSES (see file
 * godunov_utils.f90 in RAMSES).
 *
 * This is the HYDRO only version. The MHD version is in file riemann_mhd.h
 *
 * Reference : E.F. Toro, Riemann solvers and numerical methods for
 * fluid dynamics, Springer, chapter 10 (The HLL and HLLC Riemann solver).
 *
 * @param[in] qleft  : input left  state (primitive variables)
 * @param[in] qright : input right state (primitive variables)
 * @param[out] qgdnv : output Godunov state
 * @param[out] flux  : output flux
 */
template<class HydroState>
KOKKOS_INLINE_FUNCTION
void riemann_hll(const HydroState& qleft,
		 const HydroState& qright,
		 HydroState& qgdnv,
		 HydroState& flux,
		 const HydroParams& params)
{

  // 1D HLL Riemann solver

  // constants
  real_t gamma0 = params.settings.gamma0;
  real_t smallr = params.settings.smallr;
  real_t smallp = params.settings.smallp;
  //real_t smallc = params.settings.smallc;

  //const real_t smallp = smallc*smallc/gamma0;
  const real_t entho = ONE_F / (gamma0 - ONE_F);

  // Maximum wave speed
  real_t rl=FMAX(qleft [ID],smallr);
  real_t ul=     qleft [IU];
  real_t pl=FMAX(qleft [IP],rl*smallp);

  real_t rr=FMAX(qright[ID],smallr);
  real_t ur=     qright[IU];
  real_t pr=FMAX(qright[IP],rr*smallp);

  real_t cl= SQRT(gamma0*pl/rl);
  real_t cr= SQRT(gamma0*pr/rr);

  real_t SL = FMIN(FMIN(ul,ur)-FMAX(cl,cr),(real_t) ZERO_F);
  real_t SR = FMAX(FMAX(ul,ur)+FMAX(cl,cr),(real_t) ZERO_F);

  // Compute average velocity
  qgdnv[IU] = HALF_F*(qleft[IU]+qright[IU]);

  // Compute conservative variables
  HydroState uleft, uright;
  uleft [ID] = qleft [ID];
  uright[ID] = qright[ID];
  uleft [IP] = qleft [IP]*entho + HALF_F*qleft [ID]*qleft [IU]*qleft [IU];
  uright[IP] = qright[IP]*entho + HALF_F*qright[ID]*qright[IU]*qright[IU];
  uleft [IP] += HALF_F*qleft [ID]*qleft [IV]*qleft [IV];
  uright[IP] += HALF_F*qright[ID]*qright[IV]*qright[IV];
  if (std::is_same<HydroState,HydroState3d>::value) {
    uleft [IP] += HALF_F*qleft [ID]*qleft [IW]*qleft [IW];
    uright[IP] += HALF_F*qright[ID]*qright[IW]*qright[IW];
  }
  uleft [IU] = qleft [ID]*qleft [IU];
  uright[IU] = qright[ID]*qright[IU];

  // Other advected quantities
  uleft [IV] = qleft [ID]*qleft [IV];
  uright[IV] = qright[ID]*qright[IV];
  if (std::is_same<HydroState,HydroState3d>::value) {
    uleft [IW] = qleft [ID]*qleft [IW];
    uright[IW] = qright[ID]*qright[IW];
  }

  // Compute left and right fluxes
  HydroState fleft, fright;
  fleft [ID] = uleft [IU];
  fright[ID] = uright[IU];
  fleft [IP] = qleft [IU] * ( uleft [IP] + qleft [IP]);
  fright[IP] = qright[IU] * ( uright[IP] + qright[IP]);
  fleft [IU] = qleft [IP] +   uleft [IU] * qleft [IU];
  fright[IU] = qright[IP] +   uright[IU] * qright[IU];

  // Other advected quantities
  fleft [IV] = fleft [ID] * qleft [IV];
  fright[IV] = fright[ID] * qright[IV];
  if (std::is_same<HydroState,HydroState3d>::value) {
    fleft [IW] = fleft [ID] * qleft [IW];
    fright[IW] = fright[ID] * qright[IW];
  }

  // Compute HLL fluxes
  for (int nVar=0; nVar < HYDRO_2D_NBVAR; nVar++) {
    flux[nVar] = (SR * fleft[nVar] - SL * fright[nVar] +
		  SR * SL * (uright[nVar] - uleft[nVar]) ) / (SR-SL);
  }
  if (std::is_same<HydroState,HydroState3d>::value) {
    flux[IW] = (SR * fleft[IW] - SL * fright[IW] +
		SR * SL * (uright[IW] - uleft[IW]) ) / (SR-SL);
  }

} // riemann_hll

/**
 * Riemann solver HLLC
 *
 * @param[in] qleft  : input left  state (primitive variables)
 * @param[in] qright : input right state (primitive variables)
 * @param[out] qgdnv : output Godunov state
 * @param[out] flux  : output flux
 */
template<class HydroState>
KOKKOS_INLINE_FUNCTION
void riemann_hllc(const HydroState& qleft,
		  const HydroState& qright,
		  HydroState& qgdnv,
		  HydroState& flux,
		  const HydroParams& params)
{
  UNUSED(qgdnv);

  real_t gamma0 = params.settings.gamma0;
  real_t smallr = params.settings.smallr;
  real_t smallp = params.settings.smallp;
  real_t smallc = params.settings.smallc;

  const real_t entho = ONE_F / (gamma0 - ONE_F);

  // Left variables
  real_t rl = fmax(qleft[ID], smallr);
  real_t pl = fmax(qleft[IP], rl*smallp);
  real_t ul =      qleft[IU];

  real_t ecinl = HALF_F*rl*ul*ul;
  ecinl += HALF_F*rl*qleft[IV]*qleft[IV];
  if (std::is_same<HydroState,HydroState3d>::value)
    ecinl += HALF_F*rl*qleft[IW]*qleft[IW];

  real_t etotl = pl*entho+ecinl;
  real_t ptotl = pl;

  // Right variables
  real_t rr = fmax(qright[ID], smallr);
  real_t pr = fmax(qright[IP], rr*smallp);
  real_t ur =      qright[IU];

  real_t ecinr = HALF_F*rr*ur*ur;
  ecinr += HALF_F*rr*qright[IV]*qright[IV];
  if (std::is_same<HydroState,HydroState3d>::value)
    ecinl += HALF_F*rr*qright[IW]*qright[IW];

  real_t etotr = pr*entho+ecinr;
  real_t ptotr = pr;

  // Find the largest eigenvalues in the normal direction to the interface
  real_t cfastl = SQRT(fmax(gamma0*pl/rl,smallc*smallc));
  real_t cfastr = SQRT(fmax(gamma0*pr/rr,smallc*smallc));

  // Compute HLL wave speed
  real_t SL = fmin(ul,ur) - fmax(cfastl,cfastr);
  real_t SR = fmax(ul,ur) + fmax(cfastl,cfastr);

  // Compute lagrangian sound speed
  real_t rcl = rl*(ul-SL);
  real_t rcr = rr*(SR-ur);

  // Compute acoustic star state
  real_t ustar    = (rcr*ur   +rcl*ul   +  (ptotl-ptotr))/(rcr+rcl);
  real_t ptotstar = (rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl);

  // Left star region variables
  real_t rstarl    = rl*(SL-ul)/(SL-ustar);
  real_t etotstarl = ((SL-ul)*etotl-ptotl*ul+ptotstar*ustar)/(SL-ustar);

  // Right star region variables
  real_t rstarr    = rr*(SR-ur)/(SR-ustar);
  real_t etotstarr = ((SR-ur)*etotr-ptotr*ur+ptotstar*ustar)/(SR-ustar);

  // Sample the solution at x/t=0
  real_t ro, uo, ptoto, etoto;
  if (SL > ZERO_F) {
    ro=rl;
    uo=ul;
    ptoto=ptotl;
    etoto=etotl;
  } else if (ustar > ZERO_F) {
    ro=rstarl;
    uo=ustar;
    ptoto=ptotstar;
    etoto=etotstarl;
  } else if (SR > ZERO_F) {
    ro=rstarr;
    uo=ustar;
    ptoto=ptotstar;
    etoto=etotstarr;
  } else {
    ro=rr;
    uo=ur;
    ptoto=ptotr;
    etoto=etotr;
  }

  // Compute the Godunov flux
  flux[ID] = ro*uo;
  flux[IU] = ro*uo*uo+ptoto;
  flux[IP] = (etoto+ptoto)*uo;
  if (flux[ID] > ZERO_F) {
    flux[IV] = flux[ID]*qleft[IV];
  } else {
    flux[IV] = flux[ID]*qright[IV];
  }

  if (std::is_same<HydroState,HydroState3d>::value) {
    if (flux[ID] > ZERO_F) {
      flux[IW] = flux[ID]*qleft[IW];
    } else {
      flux[IW] = flux[ID]*qright[IW];
    }
  }

} // riemann_hllc

/**
 * Wrapper function calling the actual riemann solver.
 */
KOKKOS_INLINE_FUNCTION
void riemann_hydro(const HydroState2d& qleft,
		   const HydroState2d& qright,
		   HydroState2d& qgdnv,
		   HydroState2d& flux,
		   const HydroParams& params)
{

  if        (params.riemannSolverType == RIEMANN_APPROX) {

    riemann_approx<HydroState2d>(qleft,qright,qgdnv,flux,params);

  } else if (params.riemannSolverType == RIEMANN_HLL) {

    riemann_hll<HydroState2d>  (qleft,qright,qgdnv,flux,params);

  } else if (params.riemannSolverType == RIEMANN_HLLC) {

    riemann_hllc<HydroState2d>  (qleft,qright,qgdnv,flux,params);

  } else if (params.riemannSolverType == RIEMANN_LLF) {

    riemann_llf<HydroState2d>  (qleft,qright,qgdnv,flux,params);

  }

} // riemann_hydro

/**
 * Wrapper function calling the actual riemann solver.
 */
KOKKOS_INLINE_FUNCTION
void riemann_hydro(const HydroState3d& qleft,
		   const HydroState3d& qright,
		   HydroState3d& qgdnv,
		   HydroState3d& flux,
		   const HydroParams& params)
{

  if        (params.riemannSolverType == RIEMANN_APPROX) {

    riemann_approx<HydroState3d>(qleft,qright,qgdnv,flux,params);

  } else if (params.riemannSolverType == RIEMANN_HLLC) {

    riemann_hllc<HydroState3d>  (qleft,qright,qgdnv,flux,params);

  } else if (params.riemannSolverType == RIEMANN_LLF) {

    riemann_llf<HydroState3d>  (qleft,qright,qgdnv,flux,params);

  }

} // riemann_hydro

} // namespace euler_kokkos

#endif // RIEMANN_SOLVERS_H_
