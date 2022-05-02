/**
 * All possible Riemann solvers or so for MHD.
 */
#ifndef RIEMANN_SOLVERS_MHD_H_
#define RIEMANN_SOLVERS_MHD_H_

#include <math.h>

#include "HydroParams.h"
#include "HydroState.h"
#include "mhd_utils.h"

namespace euler_kokkos {

/**
 * MHD HLL Riemann solver
 *
 * qleft, qright and flux have now NVAR_MHD=8 components.
 *
 * The following code is adapted from Dumses.
 *
 * @param[in] qleft  : input left state
 * @param[in] qright : input right state
 * @param[out] flux  : output flux
 *
 */
KOKKOS_INLINE_FUNCTION
void riemann_hll(MHDState &qleft,
		 MHDState &qright,
		 MHDState &flux,
		 const HydroParams& params)
{

  // enforce continuity of normal component
  real_t bx_mean = 0.5 * ( qleft[IBX] + qright[IBX] );

  qleft[IBX]  = bx_mean;
  qright[IBX] = bx_mean;

  MHDState uleft,  fleft;
  MHDState uright, fright;

  find_mhd_flux(qleft ,uleft ,fleft , params);
  find_mhd_flux(qright,uright,fright, params);

  // find the largest eigenvalue in the normal direction to the interface
  real_t cfleft  = find_speed_fast<IX>(qleft,params);
  real_t cfright = find_speed_fast<IX>(qright,params);

  real_t vleft =qleft[IU];
  real_t vright=qright[IU];
  real_t sl=fmin ( fmin (vleft,vright) - fmax (cfleft,cfright) , 0.0);
  real_t sr=fmax ( fmax (vleft,vright) + fmax (cfleft,cfright) , 0.0);

  // the hll flux
  flux[ID] = (sr*fleft[ID]-sl*fright[ID]+
	      sr*sl*(uright[ID]-uleft[ID]))/(sr-sl);
  flux[IP] = (sr*fleft[IP]-sl*fright[IP]+
	      sr*sl*(uright[IP]-uleft[IP]))/(sr-sl);
  flux[IU] = (sr*fleft[IU]-sl*fright[IU]+
	      sr*sl*(uright[IU]-uleft[IU]))/(sr-sl);
  flux[IV] = (sr*fleft[IV]-sl*fright[IV]+
	      sr*sl*(uright[IV]-uleft[IV]))/(sr-sl);
  flux[IW] = (sr*fleft[IW]-sl*fright[IW]+
	      sr*sl*(uright[IW]-uleft[IW]))/(sr-sl);
  flux[IBX] = (sr*fleft[IBX]-sl*fright[IBX]+
	       sr*sl*(uright[IBX]-uleft[IBX]))/(sr-sl);
  flux[IBY] = (sr*fleft[IBY]-sl*fright[IBY]+
	       sr*sl*(uright[IBY]-uleft[IBY]))/(sr-sl);
  flux[IBZ] = (sr*fleft[IBZ]-sl*fright[IBZ]+
	       sr*sl*(uright[IBZ]-uleft[IBZ]))/(sr-sl);


} // riemann_hll

/*
 * MHD LLF (Local Lax-Friedrich) Riemann solver
 *
 * qleft, qright and flux have now NVAR_MHD=8 components.
 *
 * The following code is adapted from Dumses.
 *
 * @param[in] qleft  : input left state
 * @param[in] qright : input right state
 * @param[out] flux  : output flux
 *
 */
KOKKOS_INLINE_FUNCTION
void riemann_llf(MHDState &qleft,
		 MHDState &qright,
		 MHDState &flux,
		 const HydroParams &params)
{

  // enforce continuity of normal component
  real_t bx_mean = HALF_F * ( qleft[IA] + qright[IA] );
  qleft [IA] = bx_mean;
  qright[IA] = bx_mean;

  MHDState uleft,  fleft;
  MHDState uright, fright;

  find_mhd_flux(qleft ,uleft ,fleft ,params);
  find_mhd_flux(qright,uright,fright,params);

  // compute mean flux
  for (int iVar=0; iVar<MHD_NBVAR; iVar++)
    flux[iVar] = (fleft[iVar]+fright[iVar])/2;

  // find the largest eigenvalue in the normal direction to the interface
  real_t cleft  = find_speed_info(qleft ,params);
  real_t cright = find_speed_info(qright,params);

  real_t vel_info = FMAX(cleft,cright);

  // the Local Lax-Friedrich flux
  for (int iVar=0; iVar<MHD_NBVAR; iVar++)
    flux[iVar] -= vel_info*(uright[iVar]-uleft[iVar])/2;

} // riemann_llf

/**
 * Riemann solver, equivalent to riemann_hlld in RAMSES/DUMSES (see file
 * godunov_utils.f90 in RAMSES/DUMSES).
 *
 * Reference :
 * <A HREF="http://www.sciencedirect.com/science/article/B6WHY-4FY3P80-7/2/426234268c96dcca8a828d098b75fe4e">
 * Miyoshi & Kusano, 2005, JCP, 208, 315 </A>
 *
 * \warning This version of HLLD integrates the pressure term in
 * flux[IU] (as in RAMSES). This will need to be modified in the
 * future (as it is done in DUMSES) to handle cylindrical / spherical
 * coordinate systems. For example, one could add a new ouput named qStar
 * to store star state, and that could be used to compute geometrical terms
 * outside this routine.
 *
 * @param[in] qleft : input left state
 * @param[in] qright : input right state
 * @param[out] flux  : output flux
 */
KOKKOS_INLINE_FUNCTION
void riemann_hlld(MHDState &qleft,
		  MHDState &qright,
		  MHDState &flux,
		  const HydroParams& params)
{

  // Constants
  const real_t gamma0 = params.settings.gamma0;
  const real_t entho = 1.0 / (gamma0 - 1.0);

  // Enforce continuity of normal component of magnetic field
  real_t a    = 0.5 * ( qleft[IBX] + qright[IBX] );
  real_t sgnm = (a >= 0) ? ONE_F : -ONE_F;

  qleft [IBX]  = a;
  qright[IBX]  = a;

  // ISOTHERMAL
  real_t cIso = params.settings.cIso;
  if (cIso > 0) {
    // recompute pressure
    qleft [IP] = qleft [ID]*cIso*cIso;
    qright[IP] = qright[ID]*cIso*cIso;
  } // end ISOTHERMAL

    // left variables
  real_t rl, pl, ul, vl, wl, bl, cl;
  rl = qleft[ID]; //rl = fmax(qleft[ID], static_cast<real_t>(gParams.smallr)    );
  pl = qleft[IP]; //pl = fmax(qleft[IP], static_cast<real_t>(rl*gParams.smallp) );
  ul = qleft[IU];  vl = qleft[IV];  wl = qleft[IW];
  bl = qleft[IBY];  cl = qleft[IBZ];
  real_t ecinl = 0.5 * (ul*ul + vl*vl + wl*wl) * rl;
  real_t emagl = 0.5 * ( a*a  + bl*bl + cl*cl);
  real_t etotl = pl*entho + ecinl + emagl;
  real_t ptotl = pl + emagl;
  real_t vdotbl= ul*a + vl*bl + wl*cl;

  // right variables
  real_t rr, pr, ur, vr, wr, br, cr;
  rr = qright[ID]; //rr = fmax(qright[ID], static_cast<real_t>( gParams.smallr) );
  pr = qright[IP]; //pr = fmax(qright[IP], static_cast<real_t>( rr*gParams.smallp) );
  ur = qright[IU];  vr=qright[IV];  wr = qright[IW];
  br = qright[IBY];  cr=qright[IBZ];
  real_t ecinr = 0.5 * (ur*ur + vr*vr + wr*wr) * rr;
  real_t emagr = 0.5 * ( a*a  + br*br + cr*cr);
  real_t etotr = pr*entho + ecinr + emagr;
  real_t ptotr = pr + emagr;
  real_t vdotbr= ur*a + vr*br + wr*cr;

  // find the largest eigenvalues in the normal direction to the interface
  real_t cfastl = find_speed_fast<IX>(qleft,params);
  real_t cfastr = find_speed_fast<IX>(qright,params);

  // compute hll wave speed
  real_t sl = fmin(ul,ur) - fmax(cfastl,cfastr);
  real_t sr = fmax(ul,ur) + fmax(cfastl,cfastr);

  // compute lagrangian sound speed
  real_t rcl = rl * (ul-sl);
  real_t rcr = rr * (sr-ur);

  // compute acoustic star state
  real_t ustar   = (rcr*ur   +rcl*ul   +  (ptotl-ptotr))/(rcr+rcl);
  real_t ptotstar= (rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl);

  // left star region variables
  real_t estar;
  real_t rstarl, el;
  rstarl = rl*(sl-ul)/(sl-ustar);
  estar  = rl*(sl-ul)*(sl-ustar)-a*a;
  el     = rl*(sl-ul)*(sl-ul   )-a*a;
  real_t vstarl, wstarl;
  real_t bstarl, cstarl;
  // not very good (should use a small energy cut-off !!!)
  if(a*a>0 and fabs(estar/(a*a)-ONE_F)<=1e-8) {
    vstarl=vl;
    bstarl=bl;
    wstarl=wl;
    cstarl=cl;
  } else {
    vstarl=vl-a*bl*(ustar-ul)/estar;
    bstarl=bl*el/estar;
    wstarl=wl-a*cl*(ustar-ul)/estar;
    cstarl=cl*el/estar;
  }
  real_t vdotbstarl = ustar*a+vstarl*bstarl+wstarl*cstarl;
  real_t etotstarl  = ((sl-ul)*etotl-ptotl*ul+ptotstar*ustar+a*(vdotbl-vdotbstarl))/(sl-ustar);
  real_t sqrrstarl  = sqrt(rstarl);
  real_t calfvenl   = fabs(a)/sqrrstarl; /* sqrrstarl should never be zero, but it might happen if border conditions are not OK !!!!!! */
  real_t sal        = ustar-calfvenl;

  // right star region variables
  real_t rstarr, er;
  rstarr = rr*(sr-ur)/(sr-ustar);
  estar  = rr*(sr-ur)*(sr-ustar)-a*a;
  er     = rr*(sr-ur)*(sr-ur   )-a*a;
  real_t vstarr, wstarr;
  real_t bstarr, cstarr;
  // not very good (should use a small energy cut-off !!!)
  if(a*a>0 and FABS(estar/(a*a)-ONE_F)<=1e-8) {
    vstarr=vr;
    bstarr=br;
    wstarr=wr;
    cstarr=cr;
  } else {
    vstarr=vr-a*br*(ustar-ur)/estar;
    bstarr=br*er/estar;
    wstarr=wr-a*cr*(ustar-ur)/estar;
    cstarr=cr*er/estar;
  }
  real_t vdotbstarr = ustar*a+vstarr*bstarr+wstarr*cstarr;
  real_t etotstarr  = ((sr-ur)*etotr-ptotr*ur+ptotstar*ustar+a*(vdotbr-vdotbstarr))/(sr-ustar);
  real_t sqrrstarr  = sqrt(rstarr);
  real_t calfvenr   = fabs(a)/sqrrstarr; /* sqrrstarr should never be zero, but it might happen if border conditions are not OK !!!!!! */
  real_t sar        = ustar+calfvenr;

  // double star region variables
  real_t vstarstar     = (sqrrstarl*vstarl+sqrrstarr*vstarr+
			  sgnm*(bstarr-bstarl)) / (sqrrstarl+sqrrstarr);
  real_t wstarstar     = (sqrrstarl*wstarl+sqrrstarr*wstarr+
			  sgnm*(cstarr-cstarl)) / (sqrrstarl+sqrrstarr);
  real_t bstarstar     = (sqrrstarl*bstarr+sqrrstarr*bstarl+
			  sgnm*sqrrstarl*sqrrstarr*(vstarr-vstarl)) /
    (sqrrstarl+sqrrstarr);
  real_t cstarstar     = (sqrrstarl*cstarr+sqrrstarr*cstarl+
			  sgnm*sqrrstarl*sqrrstarr*(wstarr-wstarl)) /
    (sqrrstarl+sqrrstarr);
  real_t vdotbstarstar = ustar*a+vstarstar*bstarstar+wstarstar*cstarstar;
  real_t etotstarstarl = etotstarl-sgnm*sqrrstarl*(vdotbstarl-vdotbstarstar);
  real_t etotstarstarr = etotstarr+sgnm*sqrrstarr*(vdotbstarr-vdotbstarstar);

  // sample the solution at x/t=0
  real_t ro, uo, vo, wo, bo, co, ptoto, etoto, vdotbo;
  if(sl>0) { // flow is supersonic, return upwind variables
    ro=rl;
    uo=ul;
    vo=vl;
    wo=wl;
    bo=bl;
    co=cl;
    ptoto=ptotl;
    etoto=etotl;
    vdotbo=vdotbl;
  } else if (sal>0) {
    ro=rstarl;
    uo=ustar;
    vo=vstarl;
    wo=wstarl;
    bo=bstarl;
    co=cstarl;
    ptoto=ptotstar;
    etoto=etotstarl;
    vdotbo=vdotbstarl;
  } else if (ustar>0) {
    ro=rstarl;
    uo=ustar;
    vo=vstarstar;
    wo=wstarstar;
    bo=bstarstar;
    co=cstarstar;
    ptoto=ptotstar;
    etoto=etotstarstarl;
    vdotbo=vdotbstarstar;
  } else if (sar>0) {
    ro=rstarr;
    uo=ustar;
    vo=vstarstar;
    wo=wstarstar;
    bo=bstarstar;
    co=cstarstar;
    ptoto=ptotstar;
    etoto=etotstarstarr;
    vdotbo=vdotbstarstar;
  } else if (sr>0) {
    ro=rstarr;
    uo=ustar;
    vo=vstarr;
    wo=wstarr;
    bo=bstarr;
    co=cstarr;
    ptoto=ptotstar;
    etoto=etotstarr;
    vdotbo=vdotbstarr;
  } else { // flow is supersonic, return upwind variables
    ro=rr;
    uo=ur;
    vo=vr;
    wo=wr;
    bo=br;
    co=cr;
    ptoto=ptotr;
    etoto=etotr;
    vdotbo=vdotbr;
  }

  // compute the godunov flux
  flux[ID] = ro*uo;
  flux[IP] = (etoto+ptoto)*uo-a*vdotbo;
  flux[IU] = ro*uo*uo-a*a+ptoto; /* *** WARNING *** : ptoto used here (this is only valid for cartesian geometry) ! */
  flux[IV] = ro*uo*vo-a*bo;
  flux[IW] = ro*uo*wo-a*co;
  flux[IBX] = 0.0;
  flux[IBY] = bo*uo-a*vo;
  flux[IBZ] = co*uo-a*wo;

} // riemann_hlld

/**
 * Wrapper function calling the actual riemann solver for MHD.
 */
KOKKOS_INLINE_FUNCTION
void riemann_mhd(MHDState& qleft,
		 MHDState& qright,
		 MHDState& flux,
		 const HydroParams& params)
{
  if (params.riemannSolverType == RIEMANN_HLLD) {

    riemann_hlld(qleft,qright,flux,params);

  } else if (params.riemannSolverType == RIEMANN_HLL) {

    riemann_hll(qleft,qright,flux,params);

  } else if (params.riemannSolverType == RIEMANN_LLF) {

    riemann_llf(qleft,qright,flux,params);

  }

} // riemann_mhd

/**
 * 2D magnetic riemann solver of type HLLD
 *
 */
KOKKOS_INLINE_FUNCTION
real_t mag_riemann2d_hlld(const MHDState (&qLLRR)[4],
			  real_t eLLRR[4],
			  const HydroParams& params)
{

  // alias reference to input arrays
  const MHDState &qLL = qLLRR[ILL];
  const MHDState &qRL = qLLRR[IRL];
  const MHDState &qLR = qLLRR[ILR];
  const MHDState &qRR = qLLRR[IRR];

  real_t &ELL = eLLRR[ILL];
  real_t &ERL = eLLRR[IRL];
  real_t &ELR = eLLRR[ILR];
  real_t &ERR = eLLRR[IRR];
  //real_t ELL,ERL,ELR,ERR;

  const real_t &rLL=qLL[ID]; const real_t &pLL=qLL[IP];
  const real_t &uLL=qLL[IU]; const real_t &vLL=qLL[IV];
  const real_t &aLL=qLL[IBX]; const real_t &bLL=qLL[IBY] ; const real_t &cLL=qLL[IBZ];

  const real_t &rLR=qLR[ID]; const real_t &pLR=qLR[IP];
  const real_t &uLR=qLR[IU]; const real_t &vLR=qLR[IV];
  const real_t &aLR=qLR[IBX]; const real_t &bLR=qLR[IBY] ; const real_t &cLR=qLR[IBZ];

  const real_t &rRL=qRL[ID]; const real_t &pRL=qRL[IP];
  const real_t &uRL=qRL[IU]; const real_t &vRL=qRL[IV];
  const real_t &aRL=qRL[IBX]; const real_t &bRL=qRL[IBY] ; const real_t &cRL=qRL[IBZ];

  const real_t &rRR=qRR[ID]; const real_t &pRR=qRR[IP];
  const real_t &uRR=qRR[IU]; const real_t &vRR=qRR[IV];
  const real_t &aRR=qRR[IBX]; const real_t &bRR=qRR[IBY] ; const real_t &cRR=qRR[IBZ];

  // Compute 4 fast magnetosonic velocity relative to x direction
  real_t cFastLLx = find_speed_fast<IX>(qLL,params);
  real_t cFastLRx = find_speed_fast<IX>(qLR,params);
  real_t cFastRLx = find_speed_fast<IX>(qRL,params);
  real_t cFastRRx = find_speed_fast<IX>(qRR,params);

  // Compute 4 fast magnetosonic velocity relative to y direction
  real_t cFastLLy = find_speed_fast<IY>(qLL,params);
  real_t cFastLRy = find_speed_fast<IY>(qLR,params);
  real_t cFastRLy = find_speed_fast<IY>(qRL,params);
  real_t cFastRRy = find_speed_fast<IY>(qRR,params);

  // TODO : write a find_speed that computes the 2 speeds together (in
  // a single routine -> factorize computation of cFastLLx and cFastLLy

  real_t SL = FMIN4(uLL,uLR,uRL,uRR) - FMAX4(cFastLLx,cFastLRx,cFastRLx,cFastRRx);
  real_t SR = FMAX4(uLL,uLR,uRL,uRR) + FMAX4(cFastLLx,cFastLRx,cFastRLx,cFastRRx);
  real_t SB = FMIN4(vLL,vLR,vRL,vRR) - FMAX4(cFastLLy,cFastLRy,cFastRLy,cFastRRy);
  real_t ST = FMAX4(vLL,vLR,vRL,vRR) + FMAX4(cFastLLy,cFastLRy,cFastRLy,cFastRRy);

  /*ELL = uLL*bLL - vLL*aLL;
    ELR = uLR*bLR - vLR*aLR;
    ERL = uRL*bRL - vRL*aRL;
    ERR = uRR*bRR - vRR*aRR;*/

  real_t PtotLL = pLL + HALF_F * (aLL*aLL + bLL*bLL + cLL*cLL);
  real_t PtotLR = pLR + HALF_F * (aLR*aLR + bLR*bLR + cLR*cLR);
  real_t PtotRL = pRL + HALF_F * (aRL*aRL + bRL*bRL + cRL*cRL);
  real_t PtotRR = pRR + HALF_F * (aRR*aRR + bRR*bRR + cRR*cRR);

  real_t rcLLx = rLL * (uLL-SL); real_t rcRLx = rRL *(SR-uRL);
  real_t rcLRx = rLR * (uLR-SL); real_t rcRRx = rRR *(SR-uRR);
  real_t rcLLy = rLL * (vLL-SB); real_t rcLRy = rLR *(ST-vLR);
  real_t rcRLy = rRL * (vRL-SB); real_t rcRRy = rRR *(ST-vRR);

  real_t ustar = (rcLLx*uLL + rcLRx*uLR + rcRLx*uRL + rcRRx*uRR +
		  (PtotLL - PtotRL + PtotLR - PtotRR) ) / (rcLLx + rcLRx +
							   rcRLx + rcRRx);
  real_t vstar = (rcLLy*vLL + rcLRy*vLR + rcRLy*vRL + rcRRy*vRR +
		  (PtotLL - PtotLR + PtotRL - PtotRR) ) / (rcLLy + rcLRy +
							   rcRLy + rcRRy);

  real_t rstarLLx = rLL * (SL-uLL) / (SL-ustar);
  real_t BstarLL  = bLL * (SL-uLL) / (SL-ustar);
  real_t rstarLLy = rLL * (SB-vLL) / (SB-vstar);
  real_t AstarLL  = aLL * (SB-vLL) / (SB-vstar);
  real_t rstarLL  = rLL * (SL-uLL) / (SL-ustar)
    *                     (SB-vLL) / (SB-vstar);
  real_t EstarLLx = ustar * BstarLL - vLL   * aLL;
  real_t EstarLLy = uLL   * bLL     - vstar * AstarLL;
  real_t EstarLL  = ustar * BstarLL - vstar * AstarLL;

  real_t rstarLRx = rLR * (SL-uLR) / (SL-ustar);
  real_t BstarLR  = bLR * (SL-uLR) / (SL-ustar);
  real_t rstarLRy = rLR * (ST-vLR) / (ST-vstar);
  real_t AstarLR  = aLR * (ST-vLR) / (ST-vstar);
  real_t rstarLR  = rLR * (SL-uLR) / (SL-ustar) * (ST-vLR) / (ST-vstar);
  real_t EstarLRx = ustar * BstarLR - vLR   * aLR;
  real_t EstarLRy = uLR   * bLR     - vstar * AstarLR;
  real_t EstarLR  = ustar * BstarLR - vstar * AstarLR;

  real_t rstarRLx = rRL * (SR-uRL) / (SR-ustar);
  real_t BstarRL  = bRL * (SR-uRL) / (SR-ustar);
  real_t rstarRLy = rRL * (SB-vRL) / (SB-vstar);
  real_t AstarRL  = aRL * (SB-vRL) / (SB-vstar);
  real_t rstarRL  = rRL * (SR-uRL) / (SR-ustar) * (SB-vRL) / (SB-vstar);
  real_t EstarRLx = ustar * BstarRL - vRL   * aRL;
  real_t EstarRLy = uRL   * bRL     - vstar * AstarRL;
  real_t EstarRL  = ustar * BstarRL - vstar * AstarRL;

  real_t rstarRRx = rRR * (SR-uRR) / (SR-ustar);
  real_t BstarRR  = bRR * (SR-uRR) / (SR-ustar);
  real_t rstarRRy = rRR * (ST-vRR) / (ST-vstar);
  real_t AstarRR  = aRR * (ST-vRR) / (ST-vstar);
  real_t rstarRR  = rRR * (SR-uRR) / (SR-ustar) * (ST-vRR) / (ST-vstar);
  real_t EstarRRx = ustar * BstarRR - vRR   * aRR;
  real_t EstarRRy = uRR   * bRR     - vstar * AstarRR;
  real_t EstarRR  = ustar * BstarRR - vstar * AstarRR;

  real_t calfvenL = FMAX5(FABS(aLR)/SQRT(rstarLRx), FABS(AstarLR)/SQRT(rstarLR),
			  FABS(aLL)/SQRT(rstarLLx), FABS(AstarLL)/SQRT(rstarLL),
			  params.settings.smallc);
  real_t calfvenR = FMAX5(FABS(aRR)/SQRT(rstarRRx), FABS(AstarRR)/SQRT(rstarRR),
			  FABS(aRL)/SQRT(rstarRLx), FABS(AstarRL)/SQRT(rstarRL),
			  params.settings.smallc);
  real_t calfvenB = FMAX5(FABS(bLL)/SQRT(rstarLLy), FABS(BstarLL)/SQRT(rstarLL),
			  FABS(bRL)/SQRT(rstarRLy), FABS(BstarRL)/SQRT(rstarRL),
			  params.settings.smallc);
  real_t calfvenT = FMAX5(FABS(bLR)/SQRT(rstarLRy), FABS(BstarLR)/SQRT(rstarLR),
			  FABS(bRR)/SQRT(rstarRRy), FABS(BstarRR)/SQRT(rstarRR),
			  params.settings.smallc);

  real_t SAL = FMIN(ustar - calfvenL, (real_t) ZERO_F);
  real_t SAR = FMAX(ustar + calfvenR, (real_t) ZERO_F);
  real_t SAB = FMIN(vstar - calfvenB, (real_t) ZERO_F);
  real_t SAT = FMAX(vstar + calfvenT, (real_t) ZERO_F);

  real_t AstarT = (SAR*AstarRR - SAL*AstarLR) / (SAR-SAL);
  real_t AstarB = (SAR*AstarRL - SAL*AstarLL) / (SAR-SAL);

  real_t BstarR = (SAT*BstarRR - SAB*BstarRL) / (SAT-SAB);
  real_t BstarL = (SAT*BstarLR - SAB*BstarLL) / (SAT-SAB);

  // finally get emf E
  real_t E=0, tmpE=0;

  // the following part is slightly different from the original fortran
  // code since it has to much different branches
  // which generate to much branch divergence in CUDA !!!

  // compute sort of boolean (don't know if signbit is available)
  int SB_pos = (int) (1+COPYSIGN(ONE_F,SB))/2, SB_neg = 1-SB_pos;
  int ST_pos = (int) (1+COPYSIGN(ONE_F,ST))/2, ST_neg = 1-ST_pos;
  int SL_pos = (int) (1+COPYSIGN(ONE_F,SL))/2, SL_neg = 1-SL_pos;
  int SR_pos = (int) (1+COPYSIGN(ONE_F,SR))/2, SR_neg = 1-SR_pos;

  // else
  tmpE = (SAL*SAB*EstarRR-SAL*SAT*EstarRL -
	  SAR*SAB*EstarLR+SAR*SAT*EstarLL)/(SAR-SAL)/(SAT-SAB) -
    SAT*SAB/(SAT-SAB)*(AstarT-AstarB) +
    SAR*SAL/(SAR-SAL)*(BstarR-BstarL);
  E += (SB_neg * ST_pos * SL_neg * SR_pos) * tmpE;

  // SB>0
  tmpE = (SAR*EstarLLx-SAL*EstarRLx+SAR*SAL*(bRL-bLL))/(SAR-SAL);
  tmpE = SL_pos*ELL + SL_neg*SR_neg*ERL + SL_neg*SR_pos*tmpE;
  E += SB_pos * tmpE;

  // ST<0
  tmpE = (SAR*EstarLRx-SAL*EstarRRx+SAR*SAL*(bRR-bLR))/(SAR-SAL);
  tmpE = SL_pos*ELR + SL_neg*SR_neg*ERR + SL_neg*SR_pos*tmpE;
  E += (SB_neg * ST_neg) * tmpE;

  // SL>0
  tmpE = (SAT*EstarLLy-SAB*EstarLRy-SAT*SAB*(aLR-aLL))/(SAT-SAB);
  E += (SB_neg * ST_pos * SL_pos) * tmpE;

  // SR<0
  tmpE = (SAT*EstarRLy-SAB*EstarRRy-SAT*SAB*(aRR-aRL))/(SAT-SAB);
  E += (SB_neg * ST_pos * SL_neg * SR_neg) * tmpE;


  /*
    if(SB>ZERO_F) {
    if(SL>ZERO_F) {
    E=ELL;
    } else if(SR<ZERO_F) {
    E=ERL;
    } else {
    E=(SAR*EstarLLx-SAL*EstarRLx+SAR*SAL*(bRL-bLL))/(SAR-SAL);
    }
    } else if (ST<ZERO_F) {
    if(SL>ZERO_F) {
    E=ELR;
    } else if(SR<ZERO_F) {
    E=ERR;
    } else {
    E=(SAR*EstarLRx-SAL*EstarRRx+SAR*SAL*(bRR-bLR))/(SAR-SAL);
    }
    } else if (SL>ZERO_F) {
    E=(SAT*EstarLLy-SAB*EstarLRy-SAT*SAB*(aLR-aLL))/(SAT-SAB);
    } else if (SR<ZERO_F) {
    E=(SAT*EstarRLy-SAB*EstarRRy-SAT*SAB*(aRR-aRL))/(SAT-SAB);
    } else {
    E = (SAL*SAB*EstarRR-SAL*SAT*EstarRL -
    SAR*SAB*EstarLR+SAR*SAT*EstarLL)/(SAR-SAL)/(SAT-SAB) -
    SAT*SAB/(SAT-SAB)*(AstarT-AstarB) +
    SAR*SAL/(SAR-SAL)*(BstarR-BstarL);
    }
  */

  return E;

} // mag_riemann2d_hlld

/**
 * Compute emf from qEdge state vector via a 2D magnetic Riemann
 * solver (see routine cmp_mag_flux in DUMSES).
 *
 * @param[in] qEdge array containing input states qRT, qLT, qRB, qLB
 * @param[in] xPos x position in space (only needed for shearing box correction terms).
 * @return emf
 *
 * template parameters:
 *
 * @tparam emfDir plays the role of xdim/lor in DUMSES routine
 * cmp_mag_flx, i.e. define which EMF will be computed (how to define
 * parallel/orthogonal velocity). emfDir identifies the orthogonal direction.
 *
 * \note the global parameter magRiemannSolver is used to choose the
 * 2D magnetic Riemann solver.
 *
 * TODO: make xPos parameter non-optional
 */
template <EmfDir emfDir>
KOKKOS_INLINE_FUNCTION
real_t compute_emf(MHDState (&qEdge)[4],
		   const HydroParams& params,
		   real_t xPos=0)
{

  // define alias reference to input arrays
  MHDState &qRT = qEdge[IRT];
  MHDState &qLT = qEdge[ILT];
  MHDState &qRB = qEdge[IRB];
  MHDState &qLB = qEdge[ILB];

  // defines alias reference to intermediate state before applying a
  // magnetic Riemann solver
  MHDState qLLRR[4];
  MHDState &qLL = qLLRR[ILL];
  MHDState &qRL = qLLRR[IRL];
  MHDState &qLR = qLLRR[ILR];
  MHDState &qRR = qLLRR[IRR];

  // density
  qLL[ID] = qRT[ID];
  qRL[ID] = qLT[ID];
  qLR[ID] = qRB[ID];
  qRR[ID] = qLB[ID];

  // pressure
  // ISOTHERMAL
  real_t cIso = params.settings.cIso;
  if (cIso > 0) {
    qLL[IP] = qLL[ID]*cIso*cIso;
    qRL[IP] = qRL[ID]*cIso*cIso;
    qLR[IP] = qLR[ID]*cIso*cIso;
    qRR[IP] = qRR[ID]*cIso*cIso;
  } else {
    qLL[IP] = qRT[IP];
    qRL[IP] = qLT[IP];
    qLR[IP] = qRB[IP];
    qRR[IP] = qLB[IP];
  }

  // iu, iv : parallel velocity indexes
  // iw     : orthogonal velocity index
  // ia, ib, ic : idem for magnetic field
  //int iu, iv, iw, ia, ib, ic;
  if (emfDir == EMFZ) {

    //iu = IU; iv = IV; iw = IW;
    //ia = IA; ib = IB, ic = IC;

    // First parallel velocity
    qLL[IU] = qRT[IU];
    qRL[IU] = qLT[IU];
    qLR[IU] = qRB[IU];
    qRR[IU] = qLB[IU];

    // Second parallel velocity
    qLL[IV] = qRT[IV];
    qRL[IV] = qLT[IV];
    qLR[IV] = qRB[IV];
    qRR[IV] = qLB[IV];

    // First parallel magnetic field (enforce continuity)
    qLL[IBX] = HALF_F * ( qRT[IBX] + qLT[IBX] );
    qRL[IBX] = HALF_F * ( qRT[IBX] + qLT[IBX] );
    qLR[IBX] = HALF_F * ( qRB[IBX] + qLB[IBX] );
    qRR[IBX] = HALF_F * ( qRB[IBX] + qLB[IBX] );

    // Second parallel magnetic field (enforce continuity)
    qLL[IBY] = HALF_F * ( qRT[IBY] + qRB[IBY] );
    qRL[IBY] = HALF_F * ( qLT[IBY] + qLB[IBY] );
    qLR[IBY] = HALF_F * ( qRT[IBY] + qRB[IBY] );
    qRR[IBY] = HALF_F * ( qLT[IBY] + qLB[IBY] );

    // Orthogonal velocity
    qLL[IW] = qRT[IW];
    qRL[IW] = qLT[IW];
    qLR[IW] = qRB[IW];
    qRR[IW] = qLB[IW];

    // Orthogonal magnetic Field
    qLL[IBZ] = qRT[IBZ];
    qRL[IBZ] = qLT[IBZ];
    qLR[IBZ] = qRB[IBZ];
    qRR[IBZ] = qLB[IBZ];

  } else if (emfDir == EMFY) {

    //iu = IW; iv = IU; iw = IV;
    //ia = IC; ib = IA, ic = IB;

    // First parallel velocity
    qLL[IU] = qRT[IW];
    qRL[IU] = qLT[IW];
    qLR[IU] = qRB[IW];
    qRR[IU] = qLB[IW];

    // Second parallel velocity
    qLL[IV] = qRT[IU];
    qRL[IV] = qLT[IU];
    qLR[IV] = qRB[IU];
    qRR[IV] = qLB[IU];

    // First parallel magnetic field (enforce continuity)
    qLL[IBX] = HALF_F * ( qRT[IBZ] + qLT[IBZ] );
    qRL[IBX] = HALF_F * ( qRT[IBZ] + qLT[IBZ] );
    qLR[IBX] = HALF_F * ( qRB[IBZ] + qLB[IBZ] );
    qRR[IBX] = HALF_F * ( qRB[IBZ] + qLB[IBZ] );

    // Second parallel magnetic field (enforce continuity)
    qLL[IBY] = HALF_F * ( qRT[IBX] + qRB[IBX] );
    qRL[IBY] = HALF_F * ( qLT[IBX] + qLB[IBX] );
    qLR[IBY] = HALF_F * ( qRT[IBX] + qRB[IBX] );
    qRR[IBY] = HALF_F * ( qLT[IBX] + qLB[IBX] );

    // Orthogonal velocity
    qLL[IW] = qRT[IV];
    qRL[IW] = qLT[IV];
    qLR[IW] = qRB[IV];
    qRR[IW] = qLB[IV];

    // Orthogonal magnetic Field
    qLL[IBZ] = qRT[IBY];
    qRL[IBZ] = qLT[IBY];
    qLR[IBZ] = qRB[IBY];
    qRR[IBZ] = qLB[IBY];

  } else { // emfDir == EMFX

    //iu = IV; iv = IW; iw = IU;
    //ia = IB; ib = IC, ic = IA;

    // First parallel velocity
    qLL[IU] = qRT[IV];
    qRL[IU] = qLT[IV];
    qLR[IU] = qRB[IV];
    qRR[IU] = qLB[IV];

    // Second parallel velocity
    qLL[IV] = qRT[IW];
    qRL[IV] = qLT[IW];
    qLR[IV] = qRB[IW];
    qRR[IV] = qLB[IW];

    // First parallel magnetic field (enforce continuity)
    qLL[IBX] = HALF_F * ( qRT[IBY] + qLT[IBY] );
    qRL[IBX] = HALF_F * ( qRT[IBY] + qLT[IBY] );
    qLR[IBX] = HALF_F * ( qRB[IBY] + qLB[IBY] );
    qRR[IBX] = HALF_F * ( qRB[IBY] + qLB[IBY] );

    // Second parallel magnetic field (enforce continuity)
    qLL[IBY] = HALF_F * ( qRT[IBZ] + qRB[IBZ] );
    qRL[IBY] = HALF_F * ( qLT[IBZ] + qLB[IBZ] );
    qLR[IBY] = HALF_F * ( qRT[IBZ] + qRB[IBZ] );
    qRR[IBY] = HALF_F * ( qLT[IBZ] + qLB[IBZ] );

    // Orthogonal velocity
    qLL[IW] = qRT[IU];
    qRL[IW] = qLT[IU];
    qLR[IW] = qRB[IU];
    qRR[IW] = qLB[IU];

    // Orthogonal magnetic Field
    qLL[IBZ] = qRT[IBX];
    qRL[IBZ] = qLT[IBX];
    qLR[IBZ] = qRB[IBX];
    qRR[IBZ] = qLB[IBX];
  }


  // Compute final fluxes

  // vx*by - vy*bx at the four edge centers
  real_t eLLRR[4];
  real_t &ELL = eLLRR[ILL];
  real_t &ERL = eLLRR[IRL];
  real_t &ELR = eLLRR[ILR];
  real_t &ERR = eLLRR[IRR];

  ELL = qLL[IU]*qLL[IBY] - qLL[IV]*qLL[IBX];
  ERL = qRL[IU]*qRL[IBY] - qRL[IV]*qRL[IBX];
  ELR = qLR[IU]*qLR[IBY] - qLR[IV]*qLR[IBX];
  ERR = qRR[IU]*qRR[IBY] - qRR[IV]*qRR[IBX];

  real_t emf=0;
  // mag_riemann2d<>
  //if (params.magRiemannSolver == MAG_HLLD) {
  emf = mag_riemann2d_hlld(qLLRR, eLLRR, params);
  // } else if (params.magRiemannSolver == MAG_HLLA) {
  //   emf = mag_riemann2d_hlla(qLLRR, eLLRR);
  // } else if (params.magRiemannSolver == MAG_HLLF) {
  //   emf = mag_riemann2d_hllf(qLLRR, eLLRR);
  // } else if (params.magRiemannSolver == MAG_LLF) {
  //   emf = mag_riemann2d_llf(qLLRR, eLLRR);
  // }

  /* upwind solver in case of the shearing box */
  // if ( /* cartesian */ (params.settings.Omega0>0) /* and not fargo */ ) {
  //   if (emfDir==EMFX) {
  // 	real_t shear = -1.5 * params.Omega0 * xPos;
  // 	if (shear>0) {
  // 	  emf += shear * qLL[IBY];
  // 	} else {
  // 	  emf += shear * qRR[IBY];
  // 	}
  //   }
  //   if (emfDir==EMFZ) {
  // 	real_t shear = -1.5 * params.Omega0 * (xPos - params[ID]x/2);
  // 	if (shear>0) {
  // 	  emf -= shear * qLL[IBX];
  // 	} else {
  // 	  emf -= shear * qRR[IBX];
  // 	}
  //   }
  // }

  return emf;

} // compute_emf

} // namespace euler_kokkos

#endif // RIEMANN_SOLVERS_MHD_H_
