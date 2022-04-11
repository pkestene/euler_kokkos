#ifndef MHD_BASE_FUNCTOR_3D_H_
#define MHD_BASE_FUNCTOR_3D_H_

#include "shared/kokkos_shared.h"

#include "shared/HydroParams.h"
#include "shared/HydroState.h"

namespace euler_kokkos { namespace muscl {

/**
 * Base class to derive actual kokkos functor.
 * params is passed by copy.
 */
class MHDBaseFunctor3D
{

public:

  using HydroState = MHDState;
  using DataArray  = DataArray3d;

  MHDBaseFunctor3D(HydroParams params) : params(params) {};
  virtual ~MHDBaseFunctor3D() {};

  HydroParams params;
  const int nbvar = params.nbvar;

  // utility routines used in various computational kernels

  KOKKOS_INLINE_FUNCTION
  void swapValues(real_t *a, real_t *b) const
  {

    real_t tmp = *a;

    *a = *b;
    *b = tmp;

  } // swapValues

  /**
   * Copy data(i,j,k) into q.
   */
  KOKKOS_INLINE_FUNCTION
  void get_state(DataArray data, int i, int j, int k, MHDState& q) const
  {

    q[ID]  = data(i,j,k, ID);
    q[IP]  = data(i,j,k, IP);
    q[IU]  = data(i,j,k, IU);
    q[IV]  = data(i,j,k, IV);
    q[IW]  = data(i,j,k, IW);
    q[IBX] = data(i,j,k, IBX);
    q[IBY] = data(i,j,k, IBY);
    q[IBZ] = data(i,j,k, IBZ);

  } // get_state

  /**
   * Copy q into data(i,j,k).
   */
  KOKKOS_INLINE_FUNCTION
  void set_state(DataArray data, int i, int j, int k, const MHDState& q) const
  {

    data(i,j,k, ID)  = q[ID];
    data(i,j,k, IP)  = q[IP];
    data(i,j,k, IU)  = q[IU];
    data(i,j,k, IV)  = q[IV];
    data(i,j,k, IW)  = q[IW];
    data(i,j,k, IBX) = q[IBX];
    data(i,j,k, IBY) = q[IBY];
    data(i,j,k, IBZ) = q[IBZ];

  } // set_state

  /**
   *
   */
  KOKKOS_INLINE_FUNCTION
  void get_magField(const DataArray& data, int i, int j, int k, BField& b) const
  {

    b[IBFX] = data(i,j,k, IBX);
    b[IBFY] = data(i,j,k, IBY);
    b[IBFZ] = data(i,j,k, IBZ);

  } // get_magField

  /**
   * Equation of state:
   * compute pressure p and speed of sound c, from density rho and
   * internal energy eint using the "calorically perfect gas" equation
   * of state : \f$ eint=\frac{p}{\rho (\gamma-1)} \f$
   * Recall that \f$ \gamma \f$ is equal to the ratio of specific heats
   *  \f$ \left[ c_p/c_v \right] \f$.
   *
   * @param[in]  rho  density
   * @param[in]  eint internal energy
   * @param[out] p    pressure
   * @param[out] c    speed of sound
   */
  KOKKOS_INLINE_FUNCTION
  void eos(real_t rho,
	   real_t eint,
	   real_t* p,
	   real_t* c) const
  {
    real_t gamma0 = params.settings.gamma0;
    real_t smallp = params.settings.smallp;

    *p = fmax((gamma0 - ONE_F) * rho * eint, rho * smallp);
    *c = sqrt(gamma0 * (*p) / rho);

  } // eos

  /**
   * Convert conservative variables (rho, rho*u, rho*v, rho*w, e, bx, by, bz)
   * to primitive variables (rho,u,v,w,p,bx,by,bz
   *)
   * @param[in]  u  conservative variables array
   * @param[in]  magFieldNeighbors face-centered magnetic fields in neighboring cells.
   * @param[out] c  local speed of sound
   * @param[out] q  primitive    variables array (allocated in calling routine, size is constant NBVAR)
   */
  KOKKOS_INLINE_FUNCTION
  void constoprim_mhd(const MHDState& u,
		      const real_t magFieldNeighbors[3],
		      real_t &c,
		      MHDState &q) const
  {
    real_t smallr = params.settings.smallr;

    // compute density
    q[ID] = fmax(u[ID], smallr);

    // compute velocities
    q[IU] = u[IU] / q[ID];
    q[IV] = u[IV] / q[ID];
    q[IW] = u[IW] / q[ID];

    // compute cell-centered magnetic field
    q[IBX] = 0.5 * ( u[IBX] + magFieldNeighbors[0] );
    q[IBY] = 0.5 * ( u[IBY] + magFieldNeighbors[1] );
    q[IBZ] = 0.5 * ( u[IBZ] + magFieldNeighbors[2] );

    // compute specific kinetic energy and magnetic energy
    real_t eken = 0.5 * (q[IU] *q[IU]  + q[IV] *q[IV]  + q[IW] *q[IW] );
    real_t emag = 0.5 * (q[IBX]*q[IBX] + q[IBY]*q[IBY] + q[IBZ]*q[IBZ]);

    // compute pressure

    if (params.settings.cIso > 0) { // isothermal

      q[IP] = q[ID] * (params.settings.cIso) * (params.settings.cIso);
      c     =  params.settings.cIso;

    } else {

      real_t eint = (u[IP] - emag) / q[ID] - eken;

      q[IP] = fmax((params.settings.gamma0-1.0) * q[ID] * eint,
		 q[ID] * params.settings.smallp);

      // if (q[IP] < 0) {
      // 	printf("MHD pressure neg !!!\n");
      // }

      // compute speed of sound (should be removed as it is useless, hydro
      // legacy)
      c = sqrt(params.settings.gamma0 * q[IP] / q[ID]);
    }

  } // constoprim_mhd

    /**
     * Compute primitive variables slopes (dqX,dqY) for one component from q and its neighbors.
     * This routine is only used in the 3D UNSPLIT integration and slope_type = 0,1 and 2.
     *
     * Only slope_type 1 and 2 are supported.
     *
     * \param[in]  q       : current primitive variable
     * \param[in]  qPlusX  : value in the next neighbor cell along XDIR
     * \param[in]  qMinusX : value in the previous neighbor cell along XDIR
     * \param[in]  qPlusY  : value in the next neighbor cell along YDIR
     * \param[in]  qMinusY : value in the previous neighbor cell along YDIR
     * \param[in]  qPlusZ  : value in the next neighbor cell along ZDIR
     * \param[in]  qMinusZ : value in the previous neighbor cell along ZDIR
     * \param[out] dqX     : reference to an array returning the X slopes
     * \param[out] dqY     : reference to an array returning the Y slopes
     * \param[out] dqZ     : reference to an array returning the Z slopes
     *
     */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_hydro_3d_scalar(real_t q,
				     real_t qPlusX,
				     real_t qMinusX,
				     real_t qPlusY,
				     real_t qMinusY,
				     real_t qPlusZ,
				     real_t qMinusZ,
				     real_t *dqX,
				     real_t *dqY,
				     real_t *dqZ) const
  {
    real_t slope_type = params.settings.slope_type;

    real_t dlft, drgt, dcen, dsgn, slop, dlim;

    // slopes in first coordinate direction
    dlft = slope_type*(q      - qMinusX);
    drgt = slope_type*(qPlusX - q      );
    dcen = 0.5 * (qPlusX - qMinusX);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqX = dsgn * fmin( dlim, FABS(dcen) );

    // slopes in second coordinate direction
    dlft = slope_type*(q      - qMinusY);
    drgt = slope_type*(qPlusY - q      );
    dcen = 0.5 * (qPlusY - qMinusY);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqY = dsgn * fmin( dlim, FABS(dcen) );

    // slopes in second coordinate direction
    dlft = slope_type*(q      - qMinusZ);
    drgt = slope_type*(qPlusZ - q      );
    dcen = 0.5 * (qPlusZ - qMinusZ);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqZ = dsgn * fmin( dlim, FABS(dcen) );

  } // slope_unsplit_hydro_3d_scalar


  /**
   * Compute primitive variables slope (vector dq) from q and its neighbors.
   * This routine is only used in the 3D UNSPLIT integration and slope_type = 0,1,2 and 3.
   *
   * Note that slope_type is a global variable, located in symbol memory when
   * using the GPU version.
   *
   * Loosely adapted from RAMSES/hydro/umuscl.f90: subroutine uslope
   * Interface is changed to become cellwise.
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in]  qNb     : array to primitive variable vector state in the neighborhood
   * \param[out] dq      : reference to an array returning the X,Y  and Z slopes
   *
   *
   */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_hydro_3d(const MHDState & q      ,
			      const MHDState & qPlusX ,
			      const MHDState & qMinusX,
			      const MHDState & qPlusY ,
			      const MHDState & qMinusY,
			      const MHDState & qPlusZ ,
			      const MHDState & qMinusZ,
			      MHDState (&dq)[3]) const
  {
    real_t slope_type = params.settings.slope_type;

    MHDState &dqX = dq[IX];
    MHDState &dqY = dq[IY];
    MHDState &dqZ = dq[IZ];

    if (slope_type==0) {

      dqX[ID]  = ZERO_F; dqY[ID]  = ZERO_F; dqZ[ID]  = ZERO_F;
      dqX[IP]  = ZERO_F; dqY[IP]  = ZERO_F; dqZ[IP]  = ZERO_F;
      dqX[IU]  = ZERO_F; dqY[IU]  = ZERO_F; dqZ[IU]  = ZERO_F;
      dqX[IV]  = ZERO_F; dqY[IV]  = ZERO_F; dqZ[IV]  = ZERO_F;
      dqX[IW]  = ZERO_F; dqY[IW]  = ZERO_F; dqZ[IW]  = ZERO_F;
      dqX[IBX] = ZERO_F; dqY[IBX] = ZERO_F; dqZ[IBX] = ZERO_F;
      dqX[IBY] = ZERO_F; dqY[IBY] = ZERO_F; dqZ[IBY] = ZERO_F;
      dqX[IBZ] = ZERO_F; dqY[IBZ] = ZERO_F; dqZ[IBZ] = ZERO_F;

      return;
    }

    if (slope_type==1 or
	slope_type==2) {  // minmod or average

      slope_unsplit_hydro_3d_scalar(q[ID],qPlusX[ID],qMinusX[ID],qPlusY[ID],qMinusY[ID],qPlusZ[ID],qMinusZ[ID], &(dqX[ID]), &(dqY[ID]), &(dqZ[ID]));
      slope_unsplit_hydro_3d_scalar(q[IP],qPlusX[IP],qMinusX[IP],qPlusY[IP],qMinusY[IP],qPlusZ[IP],qMinusZ[IP], &(dqX[IP]), &(dqY[IP]), &(dqZ[IP]));
      slope_unsplit_hydro_3d_scalar(q[IU],qPlusX[IU],qMinusX[IU],qPlusY[IU],qMinusY[IU],qPlusZ[IU],qMinusZ[IU], &(dqX[IU]), &(dqY[IU]), &(dqZ[IU]));
      slope_unsplit_hydro_3d_scalar(q[IV],qPlusX[IV],qMinusX[IV],qPlusY[IV],qMinusY[IV],qPlusZ[IV],qMinusZ[IV], &(dqX[IV]), &(dqY[IV]), &(dqZ[IV]));
      slope_unsplit_hydro_3d_scalar(q[IW],qPlusX[IW],qMinusX[IW],qPlusY[IW],qMinusY[IW],qPlusZ[IW],qMinusZ[IW], &(dqX[IW]), &(dqY[IW]), &(dqZ[IW]));
      slope_unsplit_hydro_3d_scalar(q[IBX],qPlusX[IBX],qMinusX[IBX],qPlusY[IBX],qMinusY[IBX],qPlusZ[IBX],qMinusZ[IBX], &(dqX[IBX]), &(dqY[IBX]), &(dqZ[IBX]));
      slope_unsplit_hydro_3d_scalar(q[IBY],qPlusX[IBY],qMinusX[IBY],qPlusY[IBY],qMinusY[IBY],qPlusZ[IBY],qMinusZ[IBY], &(dqX[IBY]), &(dqY[IBY]), &(dqZ[IBY]));
      slope_unsplit_hydro_3d_scalar(q[IBZ],qPlusX[IBZ],qMinusX[IBZ],qPlusY[IBZ],qMinusY[IBZ],qPlusZ[IBZ],qMinusY[IBZ], &(dqX[IBZ]), &(dqY[IBZ]), &(dqZ[IBZ]));

    }

  } // slope_unsplit_hydro_3d


  /**
   * slope_unsplit_mhd_3d computes only magnetic field slopes in 3D; hydro
   * slopes are always computed in slope_unsplit_hydro_3d.
   *
   * Compute magnetic field slopes (vector dbf) from bf (face-centered)
   * and its neighbors.
   *
   * Note that slope_type is a global variable, located in symbol memory when
   * using the GPU version.
   *
   * Loosely adapted from RAMSES and DUMSES mhd/umuscl.f90: subroutine uslope
   * Interface is changed to become cellwise.
   *
   * \param[in]  bf  : face centered magnetic field in current
   * and neighboring cells. There are 15 values (5 values for bf_x along
   * y and z, 5 for bf_y along x and z, 5 for bf_z along x and y).
   *
   * \param[out] dbf : reference to an array returning magnetic field slopes
   *
   * \note This routine is called inside trace_unsplit_mhd_3d
   */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_mhd_3d(const real_t (&bfNeighbors)[15],
			    real_t (&dbf)[3][3]) const
  {
    /* layout for face centered magnetic field */
    const real_t &bfx        = bfNeighbors[0];
    const real_t &bfx_yplus  = bfNeighbors[1];
    const real_t &bfx_yminus = bfNeighbors[2];
    const real_t &bfx_zplus  = bfNeighbors[3];
    const real_t &bfx_zminus = bfNeighbors[4];

    const real_t &bfy        = bfNeighbors[5];
    const real_t &bfy_xplus  = bfNeighbors[6];
    const real_t &bfy_xminus = bfNeighbors[7];
    const real_t &bfy_zplus  = bfNeighbors[8];
    const real_t &bfy_zminus = bfNeighbors[9];

    const real_t &bfz        = bfNeighbors[10];
    const real_t &bfz_xplus  = bfNeighbors[11];
    const real_t &bfz_xminus = bfNeighbors[12];
    const real_t &bfz_yplus  = bfNeighbors[13];
    const real_t &bfz_yminus = bfNeighbors[14];


    real_t (&dbfX)[3] = dbf[IX];
    real_t (&dbfY)[3] = dbf[IY];
    real_t (&dbfZ)[3] = dbf[IZ];

    // default values for magnetic field slopes
    for (int nVar=0; nVar<3; ++nVar) {
      dbfX[nVar] = ZERO_F;
      dbfY[nVar] = ZERO_F;
      dbfZ[nVar] = ZERO_F;
    }

    /*
     * face-centered magnetic field slopes
     */
    // 1D transverse TVD slopes for face-centered magnetic fields
    real_t xslope_type = FMIN(params.settings.slope_type, 2.0);
    real_t dlft, drgt, dcen, dsgn, slop, dlim;
    {
      // Bx along direction Y
      dlft = xslope_type * (bfx       - bfx_yminus);
      drgt = xslope_type * (bfx_yplus - bfx       );
      dcen = HALF_F      * (bfx_yplus - bfx_yminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dbfY[IX] = dsgn * FMIN( dlim, FABS(dcen) );
      // Bx along direction Z
      dlft = xslope_type * (bfx       - bfx_zminus);
      drgt = xslope_type * (bfx_zplus - bfx       );
      dcen = HALF_F      * (bfx_zplus - bfx_zminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dbfZ[IX] = dsgn * FMIN( dlim, FABS(dcen) );

      // By along direction X
      dlft = xslope_type * (bfy       - bfy_xminus);
      drgt = xslope_type * (bfy_xplus - bfy       );
      dcen = HALF_F      * (bfy_xplus - bfy_xminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if( (dlft*drgt) <= ZERO_F )
	dlim=ZERO_F;
      dbfX[IY] = dsgn * FMIN( dlim, FABS(dcen) );
      // By along direction Z
      dlft = xslope_type * (bfy       - bfy_zminus);
      drgt = xslope_type * (bfy_zplus - bfy       );
      dcen = HALF_F      * (bfy_zplus - bfy_zminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if( (dlft*drgt) <= ZERO_F )
	dlim=ZERO_F;
      dbfZ[IY] = dsgn * FMIN( dlim, FABS(dcen) );

      // Bz along direction X
      dlft = xslope_type * (bfz       - bfz_xminus);
      drgt = xslope_type * (bfz_xplus - bfz       );
      dcen = HALF_F      * (bfz_xplus - bfz_xminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if( (dlft*drgt) <= ZERO_F )
	dlim=ZERO_F;
      dbfX[IZ] = dsgn * FMIN( dlim, FABS(dcen) );
      // Bz along direction Y
      dlft = xslope_type * (bfz       - bfz_yminus);
      drgt = xslope_type * (bfz_yplus - bfz       );
      dcen = HALF_F      * (bfz_yplus - bfz_yminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if( (dlft*drgt) <= ZERO_F )
	dlim=ZERO_F;
      dbfY[IZ] = dsgn * FMIN( dlim, FABS(dcen) );

    }

  } // slope_unsplit_mhd_3d


  /**
   * This another implementation of trace computations simpler than
   * trace_unsplit_mhd_3d.
   *
   * By simpler, we mean to design a device function that could lead to better
   * ressource utilization and thus better performances (hopefully).
   *
   * To achieve this goal, several modifications are brought (compared to
   * trace_unsplit_mhd_3d) :
   * - hydro slopes (call to slope_unsplit_hydro_3d is done outside)
   * - face-centered magnetic field slopes is done outside and before, so it is
   *   an input now
   * - electric field computation is done outside and before (probably in a
   *   separate CUDA kernel as for the GPU version), so it is now an input
   *
   *
   */
  KOKKOS_INLINE_FUNCTION
  void trace_unsplit_mhd_3d_simpler(const MHDState q,
				    MHDState (&dq)[THREE_D],
				    const real_t (&bfNb)[THREE_D*2], /* 2 faces per direction*/
				    const real_t (&dbf)[12],
				    const real_t (&elecFields)[THREE_D][2][2],
				    real_t dtdx,
				    real_t dtdy,
				    real_t dtdz,
				    real_t xPos,
				    MHDState (&qm)[THREE_D],
				    MHDState (&qp)[THREE_D],
				    MHDState (&qEdge)[4][3]) const
  {

    // inputs
    // alias to electric field components
    const real_t (&Ex)[2][2] = elecFields[IX];
    const real_t (&Ey)[2][2] = elecFields[IY];
    const real_t (&Ez)[2][2] = elecFields[IZ];

    // outputs
    // alias for q on cell edge (as defined in DUMSES trace3d routine)
    MHDState &qRT_X = qEdge[0][0];
    MHDState &qRB_X = qEdge[1][0];
    MHDState &qLT_X = qEdge[2][0];
    MHDState &qLB_X = qEdge[3][0];

    MHDState &qRT_Y = qEdge[0][1];
    MHDState &qRB_Y = qEdge[1][1];
    MHDState &qLT_Y = qEdge[2][1];
    MHDState &qLB_Y = qEdge[3][1];

    MHDState &qRT_Z = qEdge[0][2];
    MHDState &qRB_Z = qEdge[1][2];
    MHDState &qLT_Z = qEdge[2][2];
    MHDState &qLB_Z = qEdge[3][2];

    real_t gamma  = params.settings.gamma0;
    real_t smallR = params.settings.smallr;
    real_t smallp = params.settings.smallp;
    real_t Omega0 = params.settings.Omega0;
    real_t dx     = params.dx;

    // Edge centered electric field in X, Y and Z directions
    const real_t &ELL = Ex[0][0];
    const real_t &ELR = Ex[0][1];
    const real_t &ERL = Ex[1][0];
    const real_t &ERR = Ex[1][1];

    const real_t &FLL = Ey[0][0];
    const real_t &FLR = Ey[0][1];
    const real_t &FRL = Ey[1][0];
    const real_t &FRR = Ey[1][1];

    const real_t &GLL = Ez[0][0];
    const real_t &GLR = Ez[0][1];
    const real_t &GRL = Ez[1][0];
    const real_t &GRR = Ez[1][1];

    // Cell centered values
    real_t r = q[ID];
    real_t p = q[IP];
    real_t u = q[IU];
    real_t v = q[IV];
    real_t w = q[IW];
    real_t A = q[IBX];
    real_t B = q[IBY];
    real_t C = q[IBZ];

    // Face centered variables
    real_t AL =  bfNb[0];
    real_t AR =  bfNb[1];
    real_t BL =  bfNb[2];
    real_t BR =  bfNb[3];
    real_t CL =  bfNb[4];
    real_t CR =  bfNb[5];

    // Cell centered TVD slopes in X direction
    real_t& drx = dq[IX][ID];  drx *= HALF_F;
    real_t& dpx = dq[IX][IP];  dpx *= HALF_F;
    real_t& dux = dq[IX][IU];  dux *= HALF_F;
    real_t& dvx = dq[IX][IV];  dvx *= HALF_F;
    real_t& dwx = dq[IX][IW];  dwx *= HALF_F;
    real_t& dCx = dq[IX][IBZ];  dCx *= HALF_F;
    real_t& dBx = dq[IX][IBY];  dBx *= HALF_F;

    // Cell centered TVD slopes in Y direction
    real_t& dry = dq[IY][ID];  dry *= HALF_F;
    real_t& dpy = dq[IY][IP];  dpy *= HALF_F;
    real_t& duy = dq[IY][IU];  duy *= HALF_F;
    real_t& dvy = dq[IY][IV];  dvy *= HALF_F;
    real_t& dwy = dq[IY][IW];  dwy *= HALF_F;
    real_t& dCy = dq[IY][IBZ];  dCy *= HALF_F;
    real_t& dAy = dq[IY][IBX];  dAy *= HALF_F;

    // Cell centered TVD slopes in Z direction
    real_t& drz = dq[IZ][ID];  drz *= HALF_F;
    real_t& dpz = dq[IZ][IP];  dpz *= HALF_F;
    real_t& duz = dq[IZ][IU];  duz *= HALF_F;
    real_t& dvz = dq[IZ][IV];  dvz *= HALF_F;
    real_t& dwz = dq[IZ][IW];  dwz *= HALF_F;
    real_t& dAz = dq[IZ][IBX];  dAz *= HALF_F;
    real_t& dBz = dq[IZ][IBY];  dBz *= HALF_F;


    // Face centered TVD slopes in transverse direction
    real_t dALy = HALF_F * dbf[0];
    real_t dALz = HALF_F * dbf[1];
    real_t dBLx = HALF_F * dbf[2];
    real_t dBLz = HALF_F * dbf[3];
    real_t dCLx = HALF_F * dbf[4];
    real_t dCLy = HALF_F * dbf[5];

    real_t dARy = HALF_F * dbf[6];
    real_t dARz = HALF_F * dbf[7];
    real_t dBRx = HALF_F * dbf[8];
    real_t dBRz = HALF_F * dbf[9];
    real_t dCRx = HALF_F * dbf[10];
    real_t dCRy = HALF_F * dbf[11];

    // Cell centered slopes in normal direction
    real_t dAx = HALF_F * (AR - AL);
    real_t dBy = HALF_F * (BR - BL);
    real_t dCz = HALF_F * (CR - CL);

    // Source terms (including transverse derivatives)
    real_t sr0, su0, sv0, sw0, sp0, sA0, sB0, sC0;
    real_t sAL0, sAR0, sBL0, sBR0, sCL0, sCR0;

    if (true /*cartesian*/) {

      sr0 = (-u*drx-dux*r)              *dtdx + (-v*dry-dvy*r)              *dtdy + (-w*drz-dwz*r)              *dtdz;
      su0 = (-u*dux-(dpx+B*dBx+C*dCx)/r)*dtdx + (-v*duy+B*dAy/r)            *dtdy + (-w*duz+C*dAz/r)            *dtdz;
      sv0 = (-u*dvx+A*dBx/r)            *dtdx + (-v*dvy-(dpy+A*dAy+C*dCy)/r)*dtdy + (-w*dvz+C*dBz/r)            *dtdz;
      sw0 = (-u*dwx+A*dCx/r)            *dtdx + (-v*dwy+B*dCy/r)            *dtdy + (-w*dwz-(dpz+A*dAz+B*dBz)/r)*dtdz;
      sp0 = (-u*dpx-dux*gamma*p)        *dtdx + (-v*dpy-dvy*gamma*p)        *dtdy + (-w*dpz-dwz*gamma*p)        *dtdz;
      sA0 =                                     (u*dBy+B*duy-v*dAy-A*dvy)   *dtdy + (u*dCz+C*duz-w*dAz-A*dwz)   *dtdz;
      sB0 = (v*dAx+A*dvx-u*dBx-B*dux)   *dtdx +                                     (v*dCz+C*dvz-w*dBz-B*dwz)   *dtdz;
      sC0 = (w*dAx+A*dwx-u*dCx-C*dux)   *dtdx + (w*dBy+B*dwy-v*dCy-C*dvy)   *dtdy;
      if (Omega0>0) {
	real_t shear = -1.5 * Omega0 *xPos;
	sr0 = sr0 -  shear*dry*dtdy;
	su0 = su0 -  shear*duy*dtdy;
	sv0 = sv0 -  shear*dvy*dtdy;
	sw0 = sw0 -  shear*dwy*dtdy;
	sp0 = sp0 -  shear*dpy*dtdy;
	sA0 = sA0 -  shear*dAy*dtdy;
	sB0 = sB0 + (shear*dAx - 1.5 * Omega0 * A *dx)*dtdx + shear*dBz*dtdz;
	sC0 = sC0 -  shear*dCy*dtdy;
      }

      // Face-centered B-field
      sAL0 = +(GLR-GLL)*dtdy*HALF_F -(FLR-FLL)*dtdz*HALF_F;
      sAR0 = +(GRR-GRL)*dtdy*HALF_F -(FRR-FRL)*dtdz*HALF_F;
      sBL0 = -(GRL-GLL)*dtdx*HALF_F +(ELR-ELL)*dtdz*HALF_F;
      sBR0 = -(GRR-GLR)*dtdx*HALF_F +(ERR-ERL)*dtdz*HALF_F;
      sCL0 = +(FRL-FLL)*dtdx*HALF_F -(ERL-ELL)*dtdy*HALF_F;
      sCR0 = +(FRR-FLR)*dtdx*HALF_F -(ERR-ELR)*dtdy*HALF_F;

    } // end cartesian

    // Update in time the  primitive variables
    r = r + sr0;
    u = u + su0;
    v = v + sv0;
    w = w + sw0;
    p = p + sp0;
    A = A + sA0;
    B = B + sB0;
    C = C + sC0;

    AL = AL + sAL0;
    AR = AR + sAR0;
    BL = BL + sBL0;
    BR = BR + sBR0;
    CL = CL + sCL0;
    CR = CR + sCR0;

    // Face averaged right state at left interface
    qp[0][ID] = r - drx;
    qp[0][IU] = u - dux;
    qp[0][IV] = v - dvx;
    qp[0][IW] = w - dwx;
    qp[0][IP] = p - dpx;
    qp[0][IBX] = AL;
    qp[0][IBY] = B - dBx;
    qp[0][IBZ] = C - dCx;
    qp[0][ID] = FMAX(smallR,  qp[0][ID]);
    qp[0][IP] = FMAX(smallp /** qp[0][ID]*/, qp[0][IP]);

    // Face averaged left state at right interface
    qm[0][ID] = r + drx;
    qm[0][IU] = u + dux;
    qm[0][IV] = v + dvx;
    qm[0][IW] = w + dwx;
    qm[0][IP] = p + dpx;
    qm[0][IBX] = AR;
    qm[0][IBY] = B + dBx;
    qm[0][IBZ] = C + dCx;
    qm[0][ID] = FMAX(smallR,  qm[0][ID]);
    qm[0][IP] = FMAX(smallp /** qm[0][ID]*/, qm[0][IP]);

    // Face averaged top state at bottom interface
    qp[1][ID] = r - dry;
    qp[1][IU] = u - duy;
    qp[1][IV] = v - dvy;
    qp[1][IW] = w - dwy;
    qp[1][IP] = p - dpy;
    qp[1][IBX] = A - dAy;
    qp[1][IBY] = BL;
    qp[1][IBZ] = C - dCy;
    qp[1][ID] = FMAX(smallR,  qp[1][ID]);
    qp[1][IP] = FMAX(smallp /** qp[1][ID]*/, qp[1][IP]);

    // Face averaged bottom state at top interface
    qm[1][ID] = r + dry;
    qm[1][IU] = u + duy;
    qm[1][IV] = v + dvy;
    qm[1][IW] = w + dwy;
    qm[1][IP] = p + dpy;
    qm[1][IBX] = A + dAy;
    qm[1][IBY] = BR;
    qm[1][IBZ] = C + dCy;
    qm[1][ID] = FMAX(smallR,  qm[1][ID]);
    qm[1][IP] = FMAX(smallp /** qm[1][ID]*/, qm[1][IP]);

    // Face averaged front state at back interface
    qp[2][ID] = r - drz;
    qp[2][IU] = u - duz;
    qp[2][IV] = v - dvz;
    qp[2][IW] = w - dwz;
    qp[2][IP] = p - dpz;
    qp[2][IBX] = A - dAz;
    qp[2][IBY] = B - dBz;
    qp[2][IBZ] = CL;
    qp[2][ID] = FMAX(smallR,  qp[2][ID]);
    qp[2][IP] = FMAX(smallp /** qp[2][ID]*/, qp[2][IP]);

    // Face averaged back state at front interface
    qm[2][ID] = r + drz;
    qm[2][IU] = u + duz;
    qm[2][IV] = v + dvz;
    qm[2][IW] = w + dwz;
    qm[2][IP] = p + dpz;
    qm[2][IBX] = A + dAz;
    qm[2][IBY] = B + dBz;
    qm[2][IBZ] = CR;
    qm[2][ID] = FMAX(smallR,  qm[2][ID]);
    qm[2][IP] = FMAX(smallp /** qm[2][ID]*/, qm[2][IP]);

    // X-edge averaged right-top corner state (RT->LL)
    qRT_X[ID] = r + (+dry+drz);
    qRT_X[IU] = u + (+duy+duz);
    qRT_X[IV] = v + (+dvy+dvz);
    qRT_X[IW] = w + (+dwy+dwz);
    qRT_X[IP] = p + (+dpy+dpz);
    qRT_X[IBX] = A + (+dAy+dAz);
    qRT_X[IBY] = BR+ (   +dBRz);
    qRT_X[IBZ] = CR+ (+dCRy   );
    qRT_X[ID] = FMAX(smallR,  qRT_X[ID]);
    qRT_X[IP] = FMAX(smallp /** qRT_X[ID]*/, qRT_X[IP]);

    // X-edge averaged right-bottom corner state (RB->LR)
    qRB_X[ID] = r + (+dry-drz);
    qRB_X[IU] = u + (+duy-duz);
    qRB_X[IV] = v + (+dvy-dvz);
    qRB_X[IW] = w + (+dwy-dwz);
    qRB_X[IP] = p + (+dpy-dpz);
    qRB_X[IBX] = A + (+dAy-dAz);
    qRB_X[IBY] = BR+ (   -dBRz);
    qRB_X[IBZ] = CL+ (+dCLy   );
    qRB_X[ID] = FMAX(smallR,  qRB_X[ID]);
    qRB_X[IP] = FMAX(smallp /** qRB_X[ID]*/, qRB_X[IP]);

    // X-edge averaged left-top corner state (LT->RL)
    qLT_X[ID] = r + (-dry+drz);
    qLT_X[IU] = u + (-duy+duz);
    qLT_X[IV] = v + (-dvy+dvz);
    qLT_X[IW] = w + (-dwy+dwz);
    qLT_X[IP] = p + (-dpy+dpz);
    qLT_X[IBX] = A + (-dAy+dAz);
    qLT_X[IBY] = BL+ (   +dBLz);
    qLT_X[IBZ] = CR+ (-dCRy   );
    qLT_X[ID] = FMAX(smallR,  qLT_X[ID]);
    qLT_X[IP] = FMAX(smallp /** qLT_X[ID]*/, qLT_X[IP]);

    // X-edge averaged left-bottom corner state (LB->RR)
    qLB_X[ID] = r + (-dry-drz);
    qLB_X[IU] = u + (-duy-duz);
    qLB_X[IV] = v + (-dvy-dvz);
    qLB_X[IW] = w + (-dwy-dwz);
    qLB_X[IP] = p + (-dpy-dpz);
    qLB_X[IBX] = A + (-dAy-dAz);
    qLB_X[IBY] = BL+ (   -dBLz);
    qLB_X[IBZ] = CL+ (-dCLy   );
    qLB_X[ID] = FMAX(smallR,  qLB_X[ID]);
    qLB_X[IP] = FMAX(smallp /** qLB_X[ID]*/, qLB_X[IP]);

    // Y-edge averaged right-top corner state (RT->LL)
    qRT_Y[ID] = r + (+drx+drz);
    qRT_Y[IU] = u + (+dux+duz);
    qRT_Y[IV] = v + (+dvx+dvz);
    qRT_Y[IW] = w + (+dwx+dwz);
    qRT_Y[IP] = p + (+dpx+dpz);
    qRT_Y[IBX] = AR+ (   +dARz);
    qRT_Y[IBY] = B + (+dBx+dBz);
    qRT_Y[IBZ] = CR+ (+dCRx   );
    qRT_Y[ID] = FMAX(smallR,  qRT_Y[ID]);
    qRT_Y[IP] = FMAX(smallp /** qRT_Y[ID]*/, qRT_Y[IP]);

    // Y-edge averaged right-bottom corner state (RB->LR)
    qRB_Y[ID] = r + (+drx-drz);
    qRB_Y[IU] = u + (+dux-duz);
    qRB_Y[IV] = v + (+dvx-dvz);
    qRB_Y[IW] = w + (+dwx-dwz);
    qRB_Y[IP] = p + (+dpx-dpz);
    qRB_Y[IBX] = AR+ (   -dARz);
    qRB_Y[IBY] = B + (+dBx-dBz);
    qRB_Y[IBZ] = CL+ (+dCLx   );
    qRB_Y[ID] = FMAX(smallR,  qRB_Y[ID]);
    qRB_Y[IP] = FMAX(smallp /** qRB_Y[ID]*/, qRB_Y[IP]);

    // Y-edge averaged left-top corner state (LT->RL)
    qLT_Y[ID] = r + (-drx+drz);
    qLT_Y[IU] = u + (-dux+duz);
    qLT_Y[IV] = v + (-dvx+dvz);
    qLT_Y[IW] = w + (-dwx+dwz);
    qLT_Y[IP] = p + (-dpx+dpz);
    qLT_Y[IBX] = AL+ (   +dALz);
    qLT_Y[IBY] = B + (-dBx+dBz);
    qLT_Y[IBZ] = CR+ (-dCRx   );
    qLT_Y[ID] = FMAX(smallR,  qLT_Y[ID]);
    qLT_Y[IP] = FMAX(smallp /** qLT_Y[ID]*/, qLT_Y[IP]);

    // Y-edge averaged left-bottom corner state (LB->RR)
    qLB_Y[ID] = r + (-drx-drz);
    qLB_Y[IU] = u + (-dux-duz);
    qLB_Y[IV] = v + (-dvx-dvz);
    qLB_Y[IW] = w + (-dwx-dwz);
    qLB_Y[IP] = p + (-dpx-dpz);
    qLB_Y[IBX] = AL+ (   -dALz);
    qLB_Y[IBY] = B + (-dBx-dBz);
    qLB_Y[IBZ] = CL+ (-dCLx   );
    qLB_Y[ID] = FMAX(smallR,  qLB_Y[ID]);
    qLB_Y[IP] = FMAX(smallp /** qLB_Y[ID]*/, qLB_Y[IP]);

    // Z-edge averaged right-top corner state (RT->LL)
    qRT_Z[ID] = r + (+drx+dry);
    qRT_Z[IU] = u + (+dux+duy);
    qRT_Z[IV] = v + (+dvx+dvy);
    qRT_Z[IW] = w + (+dwx+dwy);
    qRT_Z[IP] = p + (+dpx+dpy);
    qRT_Z[IBX] = AR+ (   +dARy);
    qRT_Z[IBY] = BR+ (+dBRx   );
    qRT_Z[IBZ] = C + (+dCx+dCy);
    qRT_Z[ID] = FMAX(smallR,  qRT_Z[ID]);
    qRT_Z[IP] = FMAX(smallp /** qRT_Z[ID]*/, qRT_Z[IP]);

    // Z-edge averaged right-bottom corner state (RB->LR)
    qRB_Z[ID] = r + (+drx-dry);
    qRB_Z[IU] = u + (+dux-duy);
    qRB_Z[IV] = v + (+dvx-dvy);
    qRB_Z[IW] = w + (+dwx-dwy);
    qRB_Z[IP] = p + (+dpx-dpy);
    qRB_Z[IBX] = AR+ (   -dARy);
    qRB_Z[IBY] = BL+ (+dBLx   );
    qRB_Z[IBZ] = C + (+dCx-dCy);
    qRB_Z[ID] = FMAX(smallR,  qRB_Z[ID]);
    qRB_Z[IP] = FMAX(smallp /** qRB_Z[ID]*/, qRB_Z[IP]);

    // Z-edge averaged left-top corner state (LT->RL)
    qLT_Z[ID] = r + (-drx+dry);
    qLT_Z[IU] = u + (-dux+duy);
    qLT_Z[IV] = v + (-dvx+dvy);
    qLT_Z[IW] = w + (-dwx+dwy);
    qLT_Z[IP] = p + (-dpx+dpy);
    qLT_Z[IBX] = AL+ (   +dALy);
    qLT_Z[IBY] = BR+ (-dBRx   );
    qLT_Z[IBZ] = C + (-dCx+dCy);
    qLT_Z[ID] = FMAX(smallR,  qLT_Z[ID]);
    qLT_Z[IP] = FMAX(smallp /** qLT_Z[ID]*/, qLT_Z[IP]);

    // Z-edge averaged left-bottom corner state (LB->RR)
    qLB_Z[ID] = r + (-drx-dry);
    qLB_Z[IU] = u + (-dux-duy);
    qLB_Z[IV] = v + (-dvx-dvy);
    qLB_Z[IW] = w + (-dwx-dwy);
    qLB_Z[IP] = p + (-dpx-dpy);
    qLB_Z[IBX] = AL+ (   -dALy);
    qLB_Z[IBY] = BL+ (-dBLx   );
    qLB_Z[IBZ] = C + (-dCx-dCy);
    qLB_Z[ID] = FMAX(smallR,  qLB_Z[ID]);
    qLB_Z[IP] = FMAX(smallp /** qLB_Z[ID]*/, qLB_Z[IP]);

  } // trace_unsplit_mhd_3d_simpler

}; // class MHDBaseFunctor3D

} // namespace muscl

} // namespace euler_kokkos

#endif // MHD_BASE_FUNCTOR_3D_H_
