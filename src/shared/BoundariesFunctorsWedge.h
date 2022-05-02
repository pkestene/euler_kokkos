#ifndef BOUNDARIES_FUNCTORS_WEDGE_H_
#define BOUNDARIES_FUNCTORS_WEDGE_H_

#include "shared/HydroParams.h"    // for HydroParams
#include "shared/kokkos_shared.h"  // for Data arrays
#include "shared/problems/WedgeParams.h"    // for Wedge border condition

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Functors to update ghost cells (Hydro 2D) for the test case wedge,
 * also called Double Mach reflection.
 *
 * See http://amroc.sourceforge.net/examples/euler/2d/html/ramp_n.htm
 *
 * This border condition is time-dependent.
 */
template <FaceIdType faceId>
class MakeBoundariesFunctor2D_wedge {

public:

  MakeBoundariesFunctor2D_wedge(HydroParams params,
				WedgeParams wparams,
				DataArray2d Udata) :
    params(params), wparams(wparams), Udata(Udata) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int nx = params.nx;
    const int ny = params.ny;

    const int ghostWidth = params.ghostWidth;
    const int nbvar = params.nbvar;

    const int imin = params.imin;
    const int imax = params.imax;

    const int jmin = params.jmin;
    const int jmax = params.jmax;

    const real_t dx = params.dx;
    const real_t dy = params.dy;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;

    const real_t rho1   = wparams.rho1;
    const real_t rho_u1 = wparams.rho_u1;
    const real_t rho_v1 = wparams.rho_v1;
    const real_t e_tot1 = wparams.e_tot1;

    int i,j;
    int i0, j0;

    if (faceId == FACE_XMIN) {

      // boundary xmin (inflow)

      j = index / ghostWidth;
      i = index - j*ghostWidth;

      if(j >= jmin && j <= jmax    &&
	 i >= 0    && i <ghostWidth) {

	Udata(i,j,ID) = rho1;
	Udata(i,j,IP) = e_tot1;
	Udata(i,j,IU) = rho_u1;
	Udata(i,j,IV) = rho_v1;

      }

    } // end FACE_XMIN

    if (faceId == FACE_XMAX) {

      // boundary xmax (outflow)
      j = index / ghostWidth;
      i = index - j*ghostWidth;
      i += (nx+ghostWidth);

      if(j >= jmin          && j <= jmax             &&
	 i >= nx+ghostWidth && i <= nx+2*ghostWidth-1) {

	i0=nx+ghostWidth-1;
	for ( int iVar=0; iVar<nbvar; iVar++ )
	  Udata(i,j,iVar) = Udata(i0,j,iVar);

      }

    } // end FACE_XMAX

    if (faceId == FACE_YMIN) {

      // boundary ymin
      // if (x <  x_f) inflow
      // else          reflective

      i = index / ghostWidth;
      j = index - i*ghostWidth;

      real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
      //real_t y = ymin + dy/2 + (j-ghostWidth)*dy;

      if(i >= imin && i <= imax    &&
	 j >= 0    && j <ghostWidth) {

	if (x < wparams.x_f) { // inflow

	  Udata(i,j,ID) = rho1;
	  Udata(i,j,IP) = e_tot1;
	  Udata(i,j,IU) = rho_u1;
	  Udata(i,j,IV) = rho_v1;

	} else { // reflective

	  real_t sign=1.0;
	  for ( int iVar=0; iVar<nbvar; iVar++ ) {
	    j0=2*ghostWidth-1-j;
	    if (iVar==IV) sign=-ONE_F;
	    Udata(i,j,iVar) = Udata(i,j0,iVar)*sign;
	  }

	}

      } // end if i,j

    } // end FACE_YMIN

    if (faceId == FACE_YMAX) {

      // boundary ymax
      // if (x <  x_f + y/slope_f + delta_x) inflow
      // else                                outflow

      i = index / ghostWidth;
      j = index - i*ghostWidth;
      j += (ny+ghostWidth);

      real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
      real_t y = ymin + dy/2 + (j-ghostWidth)*dy;

      if(i >= imin          && i <= imax              &&
	 j >= ny+ghostWidth && j <= ny+2*ghostWidth-1) {

	if (x < wparams.x_f + y/wparams.slope_f + wparams.delta_x) { // inflow

	  Udata(i,j,ID) = rho1;
	  Udata(i,j,IP) = e_tot1;
	  Udata(i,j,IU) = rho_u1;
	  Udata(i,j,IV) = rho_v1;

	} else { // outflow

	  j0=ny+ghostWidth-1;
	  for ( int iVar=0; iVar<nbvar; iVar++ )
	    Udata(i,j,iVar) = Udata(i,j0,iVar);
	}

      } // end if i,j

    } // end FACE_YMAX

  } // end operator ()

  HydroParams params;
  WedgeParams wparams;
  DataArray2d Udata;

}; // MakeBoundariesFunctor2D_wedge

#endif // BOUNDARIES_FUNCTORS_WEDGE_H_
