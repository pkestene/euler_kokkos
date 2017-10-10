#ifndef HYDRO_BASE_FUNCTOR_3D_H_
#define HYDRO_BASE_FUNCTOR_3D_H_

#include "shared/kokkos_shared.h"

#include "shared/HydroParams.h"
#include "shared/HydroState.h"

namespace euler_kokkos { namespace muscl {

/**
 * Base class to derive actual kokkos functor for hydro 3D.
 * params is passed by copy.
 */
class HydroBaseFunctor3D
{

public:

  using HydroState = HydroState3d;
  using DataArray  = DataArray3d;
  
HydroBaseFunctor3D(HydroParams params) : params(params) {};
  virtual ~HydroBaseFunctor3D() {};

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
    
    *p = FMAX((gamma0 - ONE_F) * rho * eint, rho * smallp);
    *c = SQRT(gamma0 * (*p) / rho);
    
  } // eos
  
  /**
   * Convert conservative variables (rho, rho*u, rho*v, e) to 
   * primitive variables (rho,u,v,p)
   * @param[in]  u  conservative variables array
   * @param[out] q  primitive    variables array (allocated in calling routine, size is constant nbvar)
   * @param[out] c  local speed of sound
   */
  KOKKOS_INLINE_FUNCTION
  void computePrimitives(const HydroState& u,
			 real_t* c,
			 HydroState& q) const
  {
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;
    real_t smallp = params.settings.smallp;
    
    real_t d, p, ux, uy, uz;
    
    d = fmax(u[ID], smallr);
    ux = u[IU] / d;
    uy = u[IV] / d;
    uz = u[IW] / d;
    
    real_t eken = HALF_F * (ux*ux + uy*uy + uz*uz);
    real_t e = u[IP] / d - eken;
    
    // compute pressure and speed of sound
    p = fmax((gamma0 - 1.0) * d * e, d * smallp);
    *c = sqrt(gamma0 * (p) / d);
    
    q[ID] = d;
    q[IP] = p;
    q[IU] = ux;
    q[IV] = uy;
    q[IW] = uz;
    
  } // computePrimitive

  /**
   * Trace computations for unsplit Godunov scheme.
   *
   * \param[in] q          : Primitive variables state.
   * \param[in] qNeighbors : state in the neighbor cells (2 neighbors
   * per dimension, in the following order x+, x-, y+, y-, z+, z-)
   * \param[in] c          : local sound speed.
   * \param[in] dtdx       : dt over dx
   * \param[out] qm        : qm state (one per dimension)
   * \param[out] qp        : qp state (one per dimension)
   */
  KOKKOS_INLINE_FUNCTION
  void trace_unsplit_3d(const HydroState& q, 
			const HydroState& qNeighbors_0,
			const HydroState& qNeighbors_1,
			const HydroState& qNeighbors_2,
			const HydroState& qNeighbors_3,
			const HydroState& qNeighbors_4,
			const HydroState& qNeighbors_5,
			real_t c, 
			real_t dtdx, 
			real_t dtdy,
			real_t dtdz,
			HydroState& qm_x,
			HydroState& qm_y,
			HydroState& qm_z,
			HydroState& qp_x,
			HydroState& qp_y,
			HydroState& qp_z) const
  {
    
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;
    
    // first compute slopes
    HydroState dqX, dqY, dqZ;
    dqX[ID] = 0.0;
    dqX[IP] = 0.0;
    dqX[IU] = 0.0;
    dqX[IV] = 0.0;
    dqX[IW] = 0.0;
    
    dqY[ID] = 0.0;
    dqY[IP] = 0.0;
    dqY[IU] = 0.0;
    dqY[IV] = 0.0;
    dqY[IW] = 0.0;

    dqZ[ID] = 0.0;
    dqZ[IP] = 0.0;
    dqZ[IU] = 0.0;
    dqZ[IV] = 0.0;
    dqZ[IW] = 0.0;

    slope_unsplit_hydro_3d(q, 
			   qNeighbors_0, qNeighbors_1, 
			   qNeighbors_2, qNeighbors_3,
			   qNeighbors_4, qNeighbors_5,
			   dqX, dqY, dqZ);
      
    // Cell centered values
    real_t r =  q[ID];
    real_t p =  q[IP];
    real_t u =  q[IU];
    real_t v =  q[IV];
    real_t w =  q[IW];
      
    // TVD slopes in all directions
    real_t drx = dqX[ID];
    real_t dpx = dqX[IP];
    real_t dux = dqX[IU];
    real_t dvx = dqX[IV];
    real_t dwx = dqX[IW];
      
    real_t dry = dqY[ID];
    real_t dpy = dqY[IP];
    real_t duy = dqY[IU];
    real_t dvy = dqY[IV];
    real_t dwy = dqY[IW];

    real_t drz = dqZ[ID];
    real_t dpz = dqZ[IP];
    real_t duz = dqZ[IU];
    real_t dvz = dqZ[IV];
    real_t dwz = dqZ[IW];
      
    // source terms (with transverse derivatives)
    real_t sr0 = (-u*drx-dux*r)*dtdx + (-v*dry-dvy*r)*dtdy + (-w*drz-dwz*r)*dtdz;
    real_t su0 = (-u*dux-dpx/r)*dtdx + (-v*duy      )*dtdy + (-w*duz      )*dtdz; 
    real_t sv0 = (-u*dvx      )*dtdx + (-v*dvy-dpy/r)*dtdy + (-w*dvz      )*dtdz;
    real_t sw0 = (-u*dwx      )*dtdx + (-v*dwy      )*dtdy + (-w*dwz-dpz/r)*dtdz; 
    real_t sp0 = (-u*dpx-dux*gamma0*p)*dtdx + (-v*dpy-dvy*gamma0*p)*dtdy + (-w*dpz-dwz*gamma0*p)*dtdz;
       
    // Right state at left interface
    qp_x[ID] = r - HALF_F*drx + sr0*HALF_F;
    qp_x[IP] = p - HALF_F*dpx + sp0*HALF_F;
    qp_x[IU] = u - HALF_F*dux + su0*HALF_F;
    qp_x[IV] = v - HALF_F*dvx + sv0*HALF_F;
    qp_x[IW] = w - HALF_F*dwx + sw0*HALF_F;
    qp_x[ID] = fmax(smallr, qp_x[ID]);
      
    // Left state at right interface
    qm_x[ID] = r + HALF_F*drx + sr0*HALF_F;
    qm_x[IP] = p + HALF_F*dpx + sp0*HALF_F;
    qm_x[IU] = u + HALF_F*dux + su0*HALF_F;
    qm_x[IV] = v + HALF_F*dvx + sv0*HALF_F;
    qm_x[IW] = w + HALF_F*dwx + sw0*HALF_F;
    qm_x[ID] = fmax(smallr, qm_x[ID]);
      
    // Top state at bottom interface
    qp_y[ID] = r - HALF_F*dry + sr0*HALF_F;
    qp_y[IP] = p - HALF_F*dpy + sp0*HALF_F;
    qp_y[IU] = u - HALF_F*duy + su0*HALF_F;
    qp_y[IV] = v - HALF_F*dvy + sv0*HALF_F;
    qp_y[IW] = w - HALF_F*dwy + sw0*HALF_F;
    qp_y[ID] = fmax(smallr, qp_y[ID]);
      
    // Bottom state at top interface
    qm_y[ID] = r + HALF_F*dry + sr0*HALF_F;
    qm_y[IP] = p + HALF_F*dpy + sp0*HALF_F;
    qm_y[IU] = u + HALF_F*duy + su0*HALF_F;
    qm_y[IV] = v + HALF_F*dvy + sv0*HALF_F;
    qm_y[IW] = w + HALF_F*dwy + sw0*HALF_F;
    qm_y[ID] = fmax(smallr, qm_y[ID]);

    // Back state at bottom interface
    qp_z[ID] = r - HALF_F*drz + sr0*HALF_F;
    qp_z[IP] = p - HALF_F*dpz + sp0*HALF_F;
    qp_z[IU] = u - HALF_F*duz + su0*HALF_F;
    qp_z[IV] = v - HALF_F*dvz + sv0*HALF_F;
    qp_z[IW] = w - HALF_F*dwz + sw0*HALF_F;
    qp_z[ID] = fmax(smallr, qp_z[ID]);
      
    // Front state at top interface
    qm_z[ID] = r + HALF_F*drz + sr0*HALF_F;
    qm_z[IP] = p + HALF_F*dpz + sp0*HALF_F;
    qm_z[IU] = u + HALF_F*duz + su0*HALF_F;
    qm_z[IV] = v + HALF_F*dvz + sv0*HALF_F;
    qm_z[IW] = w + HALF_F*dwz + sw0*HALF_F;
    qm_z[ID] = fmax(smallr, qm_z[ID]);

  } // trace_unsplit_3d



  /**
   * Trace computations for unsplit Godunov scheme (3d).
   *
   * \param[in] q          : Primitive variables state.
   * \param[in] dqX        : slope along X
   * \param[in] dqY        : slope along Y
   * \param[in] dqZ        : slope along Z
   * \param[in] c          : local sound speed.
   * \param[in] dtdx       : dt over dx
   * \param[in] dtdy       : dt over dy
   * \param[in] dtdz       : dt over dz
   * \param[in] faceId     : which face will be reconstructed
   * \param[out] qface     : q reconstructed state at cell interface
   */
  KOKKOS_INLINE_FUNCTION
  void trace_unsplit_3d_along_dir(const HydroState& q, 
				  const HydroState& dqX,
				  const HydroState& dqY,
				  const HydroState& dqZ,
				  real_t dtdx, 
				  real_t dtdy, 
				  real_t dtdz, 
				  int    faceId,
				  HydroState& qface) const
  {
  
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;

    // Cell centered values
    real_t r =  q[ID];
    real_t p =  q[IP];
    real_t u =  q[IU];
    real_t v =  q[IV];
    real_t w =  q[IW];
  
    // TVD slopes in all directions
    real_t drx = dqX[ID];
    real_t dpx = dqX[IP];
    real_t dux = dqX[IU];
    real_t dvx = dqX[IV];
    real_t dwx = dqX[IW];
  
    real_t dry = dqY[ID];
    real_t dpy = dqY[IP];
    real_t duy = dqY[IU];
    real_t dvy = dqY[IV];
    real_t dwy = dqY[IW];
  
    real_t drz = dqZ[ID];
    real_t dpz = dqZ[IP];
    real_t duz = dqZ[IU];
    real_t dvz = dqZ[IV];
    real_t dwz = dqZ[IW];
  
    // source terms (with transverse derivatives)
    real_t sr0 = -u*drx-v*dry-w*drz - (dux+dvy+dwz)*r;
    real_t sp0 = -u*dpx-v*dpy-w*dpz - (dux+dvy+dwz)*gamma0*p;
    real_t su0 = -u*dux-v*duy-w*duz - (dpx        )/r;
    real_t sv0 = -u*dvx-v*dvy-w*dvz - (dpy        )/r;
    real_t sw0 = -u*dwx-v*dwy-w*dwz - (dpz        )/r;

    if (faceId == FACE_XMIN) {
      // Right state at left interface
      qface[ID] = r - HALF_F*drx + sr0*dtdx*HALF_F;
      qface[IP] = p - HALF_F*dpx + sp0*dtdx*HALF_F;
      qface[IU] = u - HALF_F*dux + su0*dtdx*HALF_F;
      qface[IV] = v - HALF_F*dvx + sv0*dtdx*HALF_F;
      qface[IW] = w - HALF_F*dwx + sw0*dtdx*HALF_F;
      qface[ID] = fmax(smallr, qface[ID]);
    }

    if (faceId == FACE_XMAX) {
      // Left state at right interface
      qface[ID] = r + HALF_F*drx + sr0*dtdx*HALF_F;
      qface[IP] = p + HALF_F*dpx + sp0*dtdx*HALF_F;
      qface[IU] = u + HALF_F*dux + su0*dtdx*HALF_F;
      qface[IV] = v + HALF_F*dvx + sv0*dtdx*HALF_F;
      qface[IW] = w + HALF_F*dwx + sw0*dtdx*HALF_F;
      qface[ID] = fmax(smallr, qface[ID]);
    }
  
    if (faceId == FACE_YMIN) {
      // Top state at bottom interface
      qface[ID] = r - HALF_F*dry + sr0*dtdy*HALF_F;
      qface[IP] = p - HALF_F*dpy + sp0*dtdy*HALF_F;
      qface[IU] = u - HALF_F*duy + su0*dtdy*HALF_F;
      qface[IV] = v - HALF_F*dvy + sv0*dtdy*HALF_F;
      qface[IW] = w - HALF_F*dwy + sw0*dtdy*HALF_F;
      qface[ID] = fmax(smallr, qface[ID]);
    }

    if (faceId == FACE_YMAX) {
      // Bottom state at top interface
      qface[ID] = r + HALF_F*dry + sr0*dtdy*HALF_F;
      qface[IP] = p + HALF_F*dpy + sp0*dtdy*HALF_F;
      qface[IU] = u + HALF_F*duy + su0*dtdy*HALF_F;
      qface[IV] = v + HALF_F*dvy + sv0*dtdy*HALF_F;
      qface[IW] = w + HALF_F*dwy + sw0*dtdy*HALF_F;
      qface[ID] = fmax(smallr, qface[ID]);
    }

    if (faceId == FACE_ZMIN) {
      // Top state at bottom interface
      qface[ID] = r - HALF_F*drz + sr0*dtdz*HALF_F;
      qface[IP] = p - HALF_F*dpz + sp0*dtdz*HALF_F;
      qface[IU] = u - HALF_F*duz + su0*dtdz*HALF_F;
      qface[IV] = v - HALF_F*dvz + sv0*dtdz*HALF_F;
      qface[IW] = w - HALF_F*dwz + sw0*dtdz*HALF_F;
      qface[ID] = fmax(smallr, qface[ID]);
    }

    if (faceId == FACE_ZMAX) {
      // Top state at bottom interface
      qface[ID] = r + HALF_F*drz + sr0*dtdz*HALF_F;
      qface[IP] = p + HALF_F*dpz + sp0*dtdz*HALF_F;
      qface[IU] = u + HALF_F*duz + su0*dtdz*HALF_F;
      qface[IV] = v + HALF_F*dvz + sv0*dtdz*HALF_F;
      qface[IW] = w + HALF_F*dwz + sw0*dtdz*HALF_F;
      qface[ID] = fmax(smallr, qface[ID]);
    }

  } // trace_unsplit_3d_along_dir

  /**
   * Compute primitive variables slopes (dqX,dqY,dqZ) for one component
   * from q and its neighbors.
   * This routine is only used in the 3D UNSPLIT integration and 
   * slope_type = 0,1 and 2.
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
    dcen = HALF_F * (qPlusX - qMinusX);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqX = dsgn * fmin( dlim, FABS(dcen) );
  
    // slopes in second coordinate direction
    dlft = slope_type*(q      - qMinusY);
    drgt = slope_type*(qPlusY - q      );
    dcen = HALF_F * (qPlusY - qMinusY);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqY = dsgn * fmin( dlim, FABS(dcen) );

    // slopes in third coordinate direction
    dlft = slope_type*(q      - qMinusZ);
    drgt = slope_type*(qPlusZ - q      );
    dcen = HALF_F * (qPlusZ - qMinusZ);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqZ = dsgn * fmin( dlim, FABS(dcen) );

  } // slope_unsplit_hydro_3d_scalar



  /**
   * Compute primitive variables slope (vector dq) from q and its neighbors.
   * This routine is only used in the 3D UNSPLIT integration and slope_type = 0,1 and 2.
   * 
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in]  q       : current primitive variable state
   * \param[in]  qPlusX  : state in the next neighbor cell along XDIR
   * \param[in]  qMinusX : state in the previous neighbor cell along XDIR
   * \param[in]  qPlusY  : state in the next neighbor cell along YDIR
   * \param[in]  qMinusY : state in the previous neighbor cell along YDIR
   * \param[in]  qPlusZ  : state in the next neighbor cell along ZDIR
   * \param[in]  qMinusZ : state in the previous neighbor cell along ZDIR
   * \param[out] dqX     : reference to an array returning the X slopes
   * \param[out] dqY     : reference to an array returning the Y slopes
   * \param[out] dqZ     : reference to an array returning the Z slopes
   *
   */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_hydro_3d(const HydroState& q, 
			      const HydroState& qPlusX, 
			      const HydroState& qMinusX,
			      const HydroState& qPlusY,
			      const HydroState& qMinusY,
			      const HydroState& qPlusZ,
			      const HydroState& qMinusZ,
			      HydroState& dqX,
			      HydroState& dqY,
			      HydroState& dqZ) const
  {
  
    real_t slope_type = params.settings.slope_type;

    if (slope_type==0) {

      dqX[ID] = ZERO_F;
      dqX[IP] = ZERO_F;
      dqX[IU] = ZERO_F;
      dqX[IV] = ZERO_F;
      dqX[IW] = ZERO_F;

      dqY[ID] = ZERO_F;
      dqY[IP] = ZERO_F;
      dqY[IU] = ZERO_F;
      dqY[IV] = ZERO_F;
      dqY[IW] = ZERO_F;

      dqZ[ID] = ZERO_F;
      dqZ[IP] = ZERO_F;
      dqZ[IU] = ZERO_F;
      dqZ[IV] = ZERO_F;
      dqZ[IW] = ZERO_F;

      return;
    }

    if (slope_type==1 || slope_type==2) {  // minmod or average

      slope_unsplit_hydro_3d_scalar( q[ID], qPlusX[ID], qMinusX[ID], qPlusY[ID], qMinusY[ID], qPlusZ[ID], qMinusZ[ID],
				     &(dqX[ID]), &(dqY[ID]), &(dqZ[ID]));
      slope_unsplit_hydro_3d_scalar( q[IP], qPlusX[IP], qMinusX[IP], qPlusY[IP], qMinusY[IP], qPlusZ[IP], qMinusZ[IP],
				     &(dqX[IP]), &(dqY[IP]), &(dqZ[IP]));
      slope_unsplit_hydro_3d_scalar( q[IU], qPlusX[IU], qMinusX[IU], qPlusY[IU], qMinusY[IU], qPlusZ[IU], qMinusZ[IU],
				     &(dqX[IU]), &(dqY[IU]), &(dqZ[IV]));
      slope_unsplit_hydro_3d_scalar( q[IV], qPlusX[IV], qMinusX[IV], qPlusY[IV], qMinusY[IV], qPlusZ[IV], qMinusZ[IV],
				     &(dqX[IV]), &(dqY[IV]), &(dqZ[IV]));
      slope_unsplit_hydro_3d_scalar( q[IW], qPlusX[IW], qMinusX[IW], qPlusY[IW], qMinusY[IW], qPlusZ[IW], qMinusZ[IW],
				     &(dqX[IW]), &(dqY[IW]), &(dqZ[IW]));

    } // end slope_type == 1 or 2
  
  } // slope_unsplit_hydro_3d

}; // class HydroBaseFunctor3D

} // namespace muscl

} // namespace euler_kokkos

#endif // HYDRO_BASE_FUNCTOR_3D_H_
