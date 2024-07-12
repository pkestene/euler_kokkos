#ifndef MHD_BASE_FUNCTOR_2D_H_
#define MHD_BASE_FUNCTOR_2D_H_

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

namespace euler_kokkos
{
namespace muscl
{

/**
 * Base class to derive actual kokkos functor.
 * params is passed by copy.
 */
class MHDBaseFunctor2D
{

public:
  using HydroState = MHDState;
  using DataArray = DataArray2d;

  MHDBaseFunctor2D(HydroParams params)
    : params(params){};
  virtual ~MHDBaseFunctor2D(){};

  HydroParams params;
  const int   nbvar = params.nbvar;

  // utility routines used in various computational kernels

  KOKKOS_INLINE_FUNCTION
  void
  swapValues(real_t * a, real_t * b) const
  {

    real_t tmp = *a;

    *a = *b;
    *b = tmp;

  } // swapValues


  /**
   * Copy data(index) into q.
   */
  KOKKOS_INLINE_FUNCTION
  void
  get_state(DataArray const & data, int i, int j, MHDState & q) const
  {

    q[ID] = data(i, j, ID);
    q[IP] = data(i, j, IP);
    q[IU] = data(i, j, IU);
    q[IV] = data(i, j, IV);
    q[IW] = data(i, j, IW);
    q[IA] = data(i, j, IA);
    q[IB] = data(i, j, IB);
    q[IC] = data(i, j, IC);

  } // get_state

  /**
   * Copy q into data(i,j).
   */
  KOKKOS_INLINE_FUNCTION
  void
  set_state(DataArray data, int i, int j, const MHDState & q) const
  {

    data(i, j, ID) = q[ID];
    data(i, j, IP) = q[IP];
    data(i, j, IU) = q[IU];
    data(i, j, IV) = q[IV];
    data(i, j, IW) = q[IW];
    data(i, j, IA) = q[IA];
    data(i, j, IB) = q[IB];
    data(i, j, IC) = q[IC];

  } // set_state

  /**
   *
   */
  KOKKOS_INLINE_FUNCTION
  void
  get_magField(const DataArray & data, int i, int j, BField & b) const
  {

    b[IBFX] = data(i, j, IA);
    b[IBFY] = data(i, j, IB);
    b[IBFZ] = data(i, j, IC);

  } // get_magField

  /**
   * Compute slopes of face-centered magnetic field along normal direction
   *
   * \param[in] data is Conservative data array (which is the one containing face-centered magnetic
   * field components)
   * \param[in] i x-coordinate
   * \param[in] j y-coordinate
   *
   * \return db magnetic slopes
   */
  KOKKOS_INLINE_FUNCTION
  BField
  compute_normal_mag_field_slopes(const DataArray & data, int i, int j) const
  {

    BField db;

    // clang-format off
    db[IX] = data(i + 1, j    , IA) - data(i, j, IA);
    db[IY] = data(i    , j + 1, IB) - data(i, j, IB);
    db[IZ] = 0.0;
    // clang-format on

    return db;
  } // compute_normal_mag_field_slopes

  /**
   * Compute limited slope of face centered magnetic field component.
   *
   * \param[in] udata is a conservative variable array
   */
  template <Direction dir>
  KOKKOS_INLINE_FUNCTION real_t
  compute_limited_slope(DataArray udata, int i, int j, int component) const
  {
    KOKKOS_ASSERT(((component == IA) or (component == IB) or (component == IC)) &&
                  "Wrong component index for a magnetic field.");

    constexpr auto delta_i = dir == DIR_X ? 1 : 0;
    constexpr auto delta_j = dir == DIR_Y ? 1 : 0;

    // clang-format off
    const auto b       = udata(i          , j          , component);
    const auto b_plus  = udata(i + delta_i, j + delta_j, component);
    const auto b_minus = udata(i - delta_i, j - delta_j, component);
    // clang-format on

    const real_t dlft = params.settings.slope_type * (b - b_minus);
    const real_t drgt = params.settings.slope_type * (b_plus - b);
    const real_t dcen = HALF_F * (b_plus - b_minus);
    const real_t dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    const real_t slop = fmin(fabs(dlft), fabs(drgt));
    real_t       dlim = slop;
    if ((dlft * drgt) <= ZERO_F)
      dlim = ZERO_F;
    return dsgn * fmin(dlim, fabs(dcen));

  } // compute_limited_slope

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
  void
  eos(real_t rho, real_t eint, real_t * p, real_t * c) const
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
   * @param[out] q  primitive    variables array (allocated in calling routine, size is constant
   *NBVAR)
   */
  KOKKOS_INLINE_FUNCTION
  void
  constoprim_mhd(const MHDState & u,
                 const real_t     magFieldNeighbors[3],
                 real_t &         c,
                 MHDState &       q) const
  {
    real_t smallr = params.settings.smallr;

    // compute density
    q[ID] = fmax(u[ID], smallr);

    // compute velocities
    q[IU] = u[IU] / q[ID];
    q[IV] = u[IV] / q[ID];
    q[IW] = u[IW] / q[ID];

    // compute cell-centered magnetic field
    q[IA] = 0.5 * (u[IA] + magFieldNeighbors[0]);
    q[IB] = 0.5 * (u[IB] + magFieldNeighbors[1]);
    q[IC] = 0.5 * (u[IC] + magFieldNeighbors[2]);

    // compute specific kinetic energy and magnetic energy
    real_t eken = 0.5 * (q[IU] * q[IU] + q[IV] * q[IV] + q[IW] * q[IW]);
    real_t emag = 0.5 * (q[IA] * q[IA] + q[IB] * q[IB] + q[IC] * q[IC]);

    // compute pressure

    if (params.settings.cIso > 0)
    { // isothermal

      q[IP] = q[ID] * (params.settings.cIso) * (params.settings.cIso);
      c = params.settings.cIso;
    }
    else
    {

      real_t eint = (u[IP] - emag) / q[ID] - eken;

      q[IP] = fmax((params.settings.gamma0 - 1.0) * q[ID] * eint, q[ID] * params.settings.smallp);

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
   * This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1 and 2.
   *
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in]  q       : current primitive variable
   * \param[in]  qPlusX  : value in the next neighbor cell along XDIR
   * \param[in]  qMinusX : value in the previous neighbor cell along XDIR
   * \param[in]  qPlusY  : value in the next neighbor cell along YDIR
   * \param[in]  qMinusY : value in the previous neighbor cell along YDIR
   * \param[out] dqX     : reference to an array returning the X slopes
   * \param[out] dqY     : reference to an array returning the Y slopes
   *
   */
  KOKKOS_INLINE_FUNCTION
  void
  slope_unsplit_hydro_2d_scalar(real_t   q,
                                real_t   qPlusX,
                                real_t   qMinusX,
                                real_t   qPlusY,
                                real_t   qMinusY,
                                real_t * dqX,
                                real_t * dqY) const
  {
    real_t slope_type = params.settings.slope_type;

    real_t dlft, drgt, dcen, dsgn, slop, dlim;

    // slopes in first coordinate direction
    dlft = slope_type * (q - qMinusX);
    drgt = slope_type * (qPlusX - q);
    dcen = 0.5 * (qPlusX - qMinusX);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin(fabs(dlft), fabs(drgt));
    dlim = slop;
    if ((dlft * drgt) <= ZERO_F)
      dlim = ZERO_F;
    *dqX = dsgn * fmin(dlim, fabs(dcen));

    // slopes in second coordinate direction
    dlft = slope_type * (q - qMinusY);
    drgt = slope_type * (qPlusY - q);
    dcen = 0.5 * (qPlusY - qMinusY);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin(fabs(dlft), fabs(drgt));
    dlim = slop;
    if ((dlft * drgt) <= ZERO_F)
      dlim = ZERO_F;
    *dqY = dsgn * fmin(dlim, fabs(dcen));

  } // slope_unsplit_hydro_2d_scalar

  /**
   * Compute primitive variables slopes (dqX,dqY) for one component from q and its neighbors.
   * This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1 and 2.
   *
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in]  q      : current primitive variable
   * \param[in]  qPlus  : value in the next neighbor cell
   * \param[in]  qMinus : value in the previous neighbor cell
   *
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  slope_unsplit_hydro_2d_scalar(real_t q, real_t qPlus, real_t qMinus) const
  {
    real_t slope_type = params.settings.slope_type;

    real_t dlft, drgt, dcen, dsgn, slop, dlim;

    // slopes in first coordinate direction
    dlft = slope_type * (q - qMinus);
    drgt = slope_type * (qPlus - q);
    dcen = 0.5 * (qPlus - qMinus);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin(fabs(dlft), fabs(drgt));
    dlim = slop;
    if ((dlft * drgt) <= ZERO_F)
      dlim = ZERO_F;
    return dsgn * fmin(dlim, fabs(dcen));

  } // slope_unsplit_hydro_2d_scalar

  /**
   * Compute primitive variables slope (vector dq) from q and its neighbors.
   * This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1,2 and 3.
   *
   * Note that slope_type is a global variable, located in symbol memory when
   * using the GPU version.
   *
   * Loosely adapted from RAMSES/hydro/umuscl.f90: subroutine uslope
   * Interface is changed to become cellwise.
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in]  qNb     : array to primitive variable vector state in the neighborhood
   * \param[out] dq      : reference to an array returning the X and Y slopes
   *
   *
   */
  KOKKOS_INLINE_FUNCTION
  void
  slope_unsplit_hydro_2d(const MHDState (&qNb)[3][3], MHDState (&dq)[2]) const
  {
    real_t slope_type = params.settings.slope_type;

    // index of current cell in the neighborhood
    enum
    {
      CENTER = 1
    };

    // aliases to input qState neighbors
    // clang-format off
    const MHDState & q       = qNb[CENTER    ][CENTER    ];
    const MHDState & qPlusX  = qNb[CENTER + 1][CENTER    ];
    const MHDState & qMinusX = qNb[CENTER - 1][CENTER    ];
    const MHDState & qPlusY  = qNb[CENTER    ][CENTER + 1];
    const MHDState & qMinusY = qNb[CENTER    ][CENTER - 1];
    // clang-format on

    MHDState & dqX = dq[IX];
    MHDState & dqY = dq[IY];

    if (slope_type == 1 or slope_type == 2)
    { // minmod or average

      slope_unsplit_hydro_2d_scalar(
        q[ID], qPlusX[ID], qMinusX[ID], qPlusY[ID], qMinusY[ID], &(dqX[ID]), &(dqY[ID]));
      slope_unsplit_hydro_2d_scalar(
        q[IP], qPlusX[IP], qMinusX[IP], qPlusY[IP], qMinusY[IP], &(dqX[IP]), &(dqY[IP]));
      slope_unsplit_hydro_2d_scalar(
        q[IU], qPlusX[IU], qMinusX[IU], qPlusY[IU], qMinusY[IU], &(dqX[IU]), &(dqY[IU]));
      slope_unsplit_hydro_2d_scalar(
        q[IV], qPlusX[IV], qMinusX[IV], qPlusY[IV], qMinusY[IV], &(dqX[IV]), &(dqY[IV]));
      slope_unsplit_hydro_2d_scalar(
        q[IW], qPlusX[IW], qMinusX[IW], qPlusY[IW], qMinusY[IW], &(dqX[IW]), &(dqY[IW]));
      slope_unsplit_hydro_2d_scalar(
        q[IA], qPlusX[IA], qMinusX[IA], qPlusY[IA], qMinusY[IA], &(dqX[IA]), &(dqY[IA]));
      slope_unsplit_hydro_2d_scalar(
        q[IB], qPlusX[IB], qMinusX[IB], qPlusY[IB], qMinusY[IB], &(dqX[IB]), &(dqY[IB]));
      slope_unsplit_hydro_2d_scalar(
        q[IC], qPlusX[IC], qMinusX[IC], qPlusY[IC], qMinusY[IC], &(dqX[IC]), &(dqY[IC]));
    }

  } // slope_unsplit_hydro_2d

  /**
   * Compute primitive variables slopes (vector dq) from q and its neighbors using cell-centered
   * values.
   * This routine is only used in the 2D UNSPLIT integration and slope_type = 1 or 2.
   *
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in]  q     : primitive variable state in center cell
   * \param[in]  qp    : primitive variable state in center cell plus one
   * \param[in]  qm    : primitive variable state in center cell minus one
   * \param[out] dq    : reference to a slope state
   *
   */
  KOKKOS_INLINE_FUNCTION void
  slope_unsplit_hydro_2d(MHDState const & q,
                         MHDState const & qp,
                         MHDState const & qm,
                         MHDState &       dq) const
  {
    real_t slope_type = params.settings.slope_type;

    // index of current cell in the neighborhood
    enum
    {
      CENTER = 1
    };


    if (slope_type == 1 or slope_type == 2)
    { // minmod or average

      dq[ID] = slope_unsplit_hydro_2d_scalar(q[ID], qp[ID], qm[ID]);
      dq[IP] = slope_unsplit_hydro_2d_scalar(q[IP], qp[IP], qm[IP]);
      dq[IU] = slope_unsplit_hydro_2d_scalar(q[IU], qp[IU], qm[IU]);
      dq[IV] = slope_unsplit_hydro_2d_scalar(q[IV], qp[IV], qm[IV]);
      dq[IW] = slope_unsplit_hydro_2d_scalar(q[IW], qp[IW], qm[IW]);
      dq[IA] = slope_unsplit_hydro_2d_scalar(q[IA], qp[IA], qm[IA]);
      dq[IB] = slope_unsplit_hydro_2d_scalar(q[IB], qp[IB], qm[IB]);
      dq[IC] = slope_unsplit_hydro_2d_scalar(q[IC], qp[IC], qm[IC]);
    }

  } // slope_unsplit_hydro_2d

  /**
   * slope_unsplit_mhd_2d computes only magnetic field slopes in 2D; hydro
   * slopes are always computed in slope_unsplit_hydro_2d.
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
   * and neighboring cells. There are 6 values (3 values for bf_x along
   * y and 3 for bf_y along x).
   *
   * \param[out] dbf : reference to an array returning magnetic field slopes
   */
  KOKKOS_INLINE_FUNCTION
  void
  slope_unsplit_mhd_2d(const real_t (&bfNeighbors)[6], real_t (&dbf)[2][3]) const
  {
    // layout for face centered magnetic field

    // clang-format off
    const real_t & bfx        = bfNeighbors[0];
    const real_t & bfx_yplus  = bfNeighbors[1];
    const real_t & bfx_yminus = bfNeighbors[2];
    const real_t & bfy        = bfNeighbors[3];
    const real_t & bfy_xplus  = bfNeighbors[4];
    const real_t & bfy_xminus = bfNeighbors[5];
    // clang-format on

    real_t(&dbfX)[3] = dbf[IX];
    real_t(&dbfY)[3] = dbf[IY];

    // default values for magnetic field slopes
    for (int nVar = 0; nVar < 3; ++nVar)
    {
      dbfX[nVar] = ZERO_F;
      dbfY[nVar] = ZERO_F;
    }

    /*
     * face-centered magnetic field slopes
     */
    // 1D transverse TVD slopes for face-centered magnetic fields

    {
      // Bx along direction Y
      real_t dlft, drgt, dcen, dsgn, slop, dlim;
      dlft = params.settings.slope_type * (bfx - bfx_yminus);
      drgt = params.settings.slope_type * (bfx_yplus - bfx);
      dcen = HALF_F * (bfx_yplus - bfx_yminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = fmin(fabs(dlft), fabs(drgt));
      dlim = slop;
      if ((dlft * drgt) <= ZERO_F)
        dlim = ZERO_F;
      dbfY[IX] = dsgn * fmin(dlim, fabs(dcen));

      // By along direction X
      dlft = params.settings.slope_type * (bfy - bfy_xminus);
      drgt = params.settings.slope_type * (bfy_xplus - bfy);
      dcen = HALF_F * (bfy_xplus - bfy_xminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = fmin(fabs(dlft), fabs(drgt));
      dlim = slop;
      if ((dlft * drgt) <= ZERO_F)
        dlim = ZERO_F;
      dbfX[IY] = dsgn * fmin(dlim, fabs(dcen));
    }

  } // slope_unsplit_mhd_2d

  /**
   * Compute electric field E = U ^ B at mid-edge
   *
   * \param[in] conservative data array
   * \param[in] primitive data array
   * \param[in] x-coordinate
   * \param[in] y-coordinate
   *
   * \return electric field at mid-edge
   */
  KOKKOS_INLINE_FUNCTION real_t
  compute_electric_field_2d(DataArray const & udata, DataArray const & qdata, int i, int j) const
  {
    // clang-format off
    const real_t u = ONE_FOURTH_F * (qdata(i - 1, j - 1, IU) +
                                     qdata(i - 1, j    , IU) +
                                     qdata(i    , j - 1, IU) +
                                     qdata(i    , j    , IU));

    const real_t v = ONE_FOURTH_F * (qdata(i - 1, j - 1, IV) +
                                     qdata(i - 1, j    , IV) +
                                     qdata(i    , j - 1, IV) +
                                     qdata(i    , j    , IV));


    const real_t A = HALF_F * (udata(i,j-1,IA) + udata(i,j,IA));

    const real_t B = HALF_F * (udata(i-1,j,IB) + udata(i,j,IB));
    // clang-format on

    return u * B - v * A;
  }

  /**
   * Trace computations for unsplit Godunov scheme.
   *
   * \param[in] q          : Primitive variables state.
   * \param[in] dqX        : slope along X
   * \param[in] dqY        : slope along Y
   * \param[in] c          : local sound speed.
   * \param[in] dtdx       : dt over dx
   * \param[in] dtdy       : dt over dy
   * \param[in] faceId     : which face will be reconstructed
   * \param[out] qface     : q reconstructed state at cell interface
   */
  KOKKOS_INLINE_FUNCTION
  void
  trace_unsplit_2d_along_dir(const MHDState & q,
                             const MHDState & dqX,
                             const MHDState & dqY,
                             real_t           dtdx,
                             real_t           dtdy,
                             int              faceId,
                             MHDState &       qface) const
  {

    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;

    // Cell centered values
    real_t r = q[ID];
    real_t p = q[IP];
    real_t u = q[IU];
    real_t v = q[IV];

    // TVD slopes in all directions
    real_t drx = dqX[ID];
    real_t dpx = dqX[IP];
    real_t dux = dqX[IU];
    real_t dvx = dqX[IV];

    real_t dry = dqY[ID];
    real_t dpy = dqY[IP];
    real_t duy = dqY[IU];
    real_t dvy = dqY[IV];

    // source terms (with transverse derivatives)
    real_t sr0 = -u * drx - v * dry - (dux + dvy) * r;
    real_t sp0 = -u * dpx - v * dpy - (dux + dvy) * gamma0 * p;
    real_t su0 = -u * dux - v * duy - (dpx) / r;
    real_t sv0 = -u * dvx - v * dvy - (dpy) / r;

    if (faceId == FACE_XMIN)
    {
      // Right state at left interface
      qface[ID] = r - 0.5 * drx + sr0 * dtdx * 0.5;
      qface[IP] = p - 0.5 * dpx + sp0 * dtdx * 0.5;
      qface[IU] = u - 0.5 * dux + su0 * dtdx * 0.5;
      qface[IV] = v - 0.5 * dvx + sv0 * dtdx * 0.5;
      qface[ID] = fmax(smallr, qface[ID]);
    }

    if (faceId == FACE_XMAX)
    {
      // Left state at right interface
      qface[ID] = r + 0.5 * drx + sr0 * dtdx * 0.5;
      qface[IP] = p + 0.5 * dpx + sp0 * dtdx * 0.5;
      qface[IU] = u + 0.5 * dux + su0 * dtdx * 0.5;
      qface[IV] = v + 0.5 * dvx + sv0 * dtdx * 0.5;
      qface[ID] = fmax(smallr, qface[ID]);
    }

    if (faceId == FACE_YMIN)
    {
      // Top state at bottom interface
      qface[ID] = r - 0.5 * dry + sr0 * dtdy * 0.5;
      qface[IP] = p - 0.5 * dpy + sp0 * dtdy * 0.5;
      qface[IU] = u - 0.5 * duy + su0 * dtdy * 0.5;
      qface[IV] = v - 0.5 * dvy + sv0 * dtdy * 0.5;
      qface[ID] = fmax(smallr, qface[ID]);
    }

    if (faceId == FACE_YMAX)
    {
      // Bottom state at top interface
      qface[ID] = r + 0.5 * dry + sr0 * dtdy * 0.5;
      qface[IP] = p + 0.5 * dpy + sp0 * dtdy * 0.5;
      qface[IU] = u + 0.5 * duy + su0 * dtdy * 0.5;
      qface[IV] = v + 0.5 * dvy + sv0 * dtdy * 0.5;
      qface[ID] = fmax(smallr, qface[ID]);
    }

  } // trace_unsplit_2d_along_dir


  /**
   * This another implementation of trace computations for 2D data; it
   * is used when unsplitVersion = 1
   *
   * Note that :
   * - hydro slopes computations are done outside this routine
   *
   * \param[in]  q  primitive variable state vector
   * \param[in]  dq primitive variable slopes
   * \param[in]  dtdx dt divided by dx
   * \param[in]  dtdy dt divided by dy
   * \param[out] qm
   * \param[out] qp
   *
   */
  KOKKOS_INLINE_FUNCTION
  void
  trace_unsplit_hydro_2d(const MHDState & q,
                         const MHDState & dqX,
                         const MHDState & dqY,
                         real_t           dtdx,
                         real_t           dtdy,
                         MHDState &       qm_x,
                         MHDState &       qm_y,
                         MHDState &       qp_x,
                         MHDState &       qp_y) const
  {

    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;
    real_t smallp = params.settings.smallp;

    // Cell centered values
    real_t r = q[ID];
    real_t p = q[IP];
    real_t u = q[IU];
    real_t v = q[IV];

    // Cell centered TVD slopes in X direction
    real_t drx = dqX[ID];
    drx *= 0.5;
    real_t dpx = dqX[IP];
    dpx *= 0.5;
    real_t dux = dqX[IU];
    dux *= 0.5;
    real_t dvx = dqX[IV];
    dvx *= 0.5;

    // Cell centered TVD slopes in Y direction
    real_t dry = dqY[ID];
    dry *= 0.5;
    real_t dpy = dqY[IP];
    dpy *= 0.5;
    real_t duy = dqY[IU];
    duy *= 0.5;
    real_t dvy = dqY[IV];
    dvy *= 0.5;

    // Source terms (including transverse derivatives)
    real_t sr0, su0, sv0, sp0;

    /*only true for cartesian grid */
    {
      sr0 = (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
      su0 = (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy;
      sv0 = (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy;
      sp0 = (-u * dpx - dux * gamma0 * p) * dtdx + (-v * dpy - dvy * gamma0 * p) * dtdy;
    } // end cartesian

    // Update in time the  primitive variables
    r = r + sr0;
    u = u + su0;
    v = v + sv0;
    p = p + sp0;

    // Face averaged right state at left interface
    qp_x[ID] = r - drx;
    qp_x[IU] = u - dux;
    qp_x[IV] = v - dvx;
    qp_x[IP] = p - dpx;
    qp_x[ID] = fmax(smallr, qp_x[ID]);
    qp_x[IP] = fmax(smallp * qp_x[ID], qp_x[IP]);

    // Face averaged left state at right interface
    qm_x[ID] = r + drx;
    qm_x[IU] = u + dux;
    qm_x[IV] = v + dvx;
    qm_x[IP] = p + dpx;
    qm_x[ID] = fmax(smallr, qm_x[ID]);
    qm_x[IP] = fmax(smallp * qm_x[ID], qm_x[IP]);

    // Face averaged top state at bottom interface
    qp_y[ID] = r - dry;
    qp_y[IU] = u - duy;
    qp_y[IV] = v - dvy;
    qp_y[IP] = p - dpy;
    qp_y[ID] = fmax(smallr, qp_y[ID]);
    qp_y[IP] = fmax(smallp * qp_y[ID], qp_y[IP]);

    // Face averaged bottom state at top interface
    qm_y[ID] = r + dry;
    qm_y[IU] = u + duy;
    qm_y[IV] = v + dvy;
    qm_y[IP] = p + dpy;
    qm_y[ID] = fmax(smallr, qm_y[ID]);
    qm_y[IP] = fmax(smallp * qm_y[ID], qm_y[IP]);

  } // trace_unsplit_hydro_2d

  /**
   * 2D Trace computations for unsplit Godunov scheme.
   *
   * \note Note that this routine uses global variables iorder, scheme and
   * slope_type.
   *
   * \note Note that is routine is loosely adapted from trace2d found in
   * Dumses and in Ramses sources (sub-dir mhd, file umuscl.f90) to be now a one cell
   * computation.
   *
   * \param[in]  qNb        state in neighbor cells (3-by-3 neighborhood indexed as qNb[i][j], for
   * i,j=0,1,2); current center cell is at index (i=j=1).
   *
   * \param[in]  bfNb       face centered
   * magnetic field in neighbor cells (4-by-4 neighborhood indexed as bfNb[i][j]
   * for i,j=0,1,2,3); current cell is located at index (i=j=1)
   *
   * \param[in]  c          local sound speed.
   *
   * \param[in]  dtdx       dt over dx
   *
   * \param[in]  dtdy       dt over dy
   *
   * \param[in]  xPos       x location of current cell (needed for shear computation)
   *
   * \param[out] qm         qm state (one per dimension)
   * \param[out] qp         qp state (one per dimension)
   * \param[out] qEdge      q state on cell edges (qRT, qRB, qLT, qLB)
   */
  KOKKOS_INLINE_FUNCTION
  void
  trace_unsplit_mhd_2d(const MHDState (&qNb)[3][3],
                       const BField (&bfNb)[4][4],
                       real_t c,
                       real_t dtdx,
                       real_t dtdy,
                       real_t xPos,
                       MHDState (&qm)[2],
                       MHDState (&qp)[2],
                       MHDState (&qEdge)[4]) const
  {
    (void)c;

    // neighborhood sizes
    enum
    {
      Q_SIZE = 3,
      BF_SIZE = 4
    };

    // index of current cell in the neighborhood
    enum
    {
      CENTER = 1
    };

    // alias for q on cell edge (as defined in DUMSES trace2d routine)
    MHDState & qRT = qEdge[0];
    MHDState & qRB = qEdge[1];
    MHDState & qLT = qEdge[2];
    MHDState & qLB = qEdge[3];

    real_t smallR = params.settings.smallr;
    real_t smallp = params.settings.smallp;
    // real_t &smallP = params.settings.smallpp;
    real_t gamma = params.settings.gamma0;
    // real_t &Omega0 = params.settings.Omega0;

    const MHDState & q = qNb[CENTER][CENTER]; // current cell (neighborhood center)

    // compute u,v,A,B,Ez (electric field)
    real_t Ez[2][2];
    for (int di = 0; di < 2; di++)
    {
      for (int dj = 0; dj < 2; dj++)
      {

        int centerX = CENTER + di;
        int centerY = CENTER + dj;
        // clang-format off
        real_t u = 0.25f * (qNb[centerX - 1][centerY - 1][IU] + qNb[centerX - 1][centerY][IU] +
                            qNb[centerX    ][centerY - 1][IU] + qNb[centerX    ][centerY][IU]);

        real_t v = 0.25f * (qNb[centerX - 1][centerY - 1][IV] + qNb[centerX - 1][centerY][IV] +
                            qNb[centerX    ][centerY - 1][IV] + qNb[centerX    ][centerY][IV]);

        real_t A = 0.5f * (bfNb[centerX    ][centerY - 1][IBFX] + bfNb[centerX][centerY][IBFX]);

        real_t B = 0.5f * (bfNb[centerX - 1][centerY    ][IBFY] + bfNb[centerX][centerY][IBFY]);
        // clang-format on

        Ez[di][dj] = u * B - v * A;
      }
    }

    // Electric field
    real_t & ELL = Ez[0][0];
    real_t & ELR = Ez[0][1];
    real_t & ERL = Ez[1][0];
    real_t & ERR = Ez[1][1];

    // Cell centered values
    real_t r = q[ID];
    real_t p = q[IP];
    real_t u = q[IU];
    real_t v = q[IV];
    real_t w = q[IW];
    real_t A = q[IA];
    real_t B = q[IB];
    real_t C = q[IC];

    // Face centered variables
    // clang-format off
    real_t AL = bfNb[CENTER    ][CENTER    ][IBFX];
    real_t AR = bfNb[CENTER + 1][CENTER    ][IBFX];
    real_t BL = bfNb[CENTER    ][CENTER    ][IBFY];
    real_t BR = bfNb[CENTER    ][CENTER + 1][IBFY];
    // clang-format on

    // TODO LATER : compute xL, xR and xC using ::gParam
    // this is only needed when doing cylindrical or spherical coordinates

    /*
     * compute hydro slopes
     */
    MHDState dq[2];

    slope_unsplit_hydro_2d(qNb, dq);

    // slight modification compared to DUMSES (we re-used dq itself,
    // instead of re-declaring new variables, better for the GPU
    // register count

    // Cell centered TVD slopes in X direction
    real_t drx = dq[IX][ID];
    drx *= 0.5;
    real_t dpx = dq[IX][IP];
    dpx *= 0.5;
    real_t dux = dq[IX][IU];
    dux *= 0.5;
    real_t dvx = dq[IX][IV];
    dvx *= 0.5;
    real_t dwx = dq[IX][IW];
    dwx *= 0.5;
    real_t dCx = dq[IX][IC];
    dCx *= 0.5;
    real_t dBx = dq[IX][IB];
    dBx *= 0.5;

    // Cell centered TVD slopes in Y direction
    real_t dry = dq[IY][ID];
    dry *= 0.5;
    real_t dpy = dq[IY][IP];
    dpy *= 0.5;
    real_t duy = dq[IY][IU];
    duy *= 0.5;
    real_t dvy = dq[IY][IV];
    dvy *= 0.5;
    real_t dwy = dq[IY][IW];
    dwy *= 0.5;
    real_t dCy = dq[IY][IC];
    dCy *= 0.5;
    real_t dAy = dq[IY][IA];
    dAy *= 0.5;

    /*
     * compute dbf slopes needed for Face centered TVD slopes in transverse direction
     */
    real_t bfNeighbors[6];
    real_t dbf[2][3];
    real_t(&dbfX)[3] = dbf[IX];
    real_t(&dbfY)[3] = dbf[IY];

    // clang-format off
    bfNeighbors[0] = bfNb[CENTER    ][CENTER    ][IBFX];
    bfNeighbors[1] = bfNb[CENTER    ][CENTER + 1][IBFX];
    bfNeighbors[2] = bfNb[CENTER    ][CENTER - 1][IBFX];
    bfNeighbors[3] = bfNb[CENTER    ][CENTER    ][IBFY];
    bfNeighbors[4] = bfNb[CENTER + 1][CENTER    ][IBFY];
    bfNeighbors[5] = bfNb[CENTER - 1][CENTER    ][IBFY];
    // clang-format on

    slope_unsplit_mhd_2d(bfNeighbors, dbf);

    // Face centered TVD slopes in transverse direction
    real_t dALy = 0.5 * dbfY[IX];
    real_t dBLx = 0.5 * dbfX[IY];

    // change neighbors to i+1, j and recompute dbf
    // clang-format off
    bfNeighbors[0] = bfNb[CENTER + 1][CENTER    ][IBFX];
    bfNeighbors[1] = bfNb[CENTER + 1][CENTER + 1][IBFX];
    bfNeighbors[2] = bfNb[CENTER + 1][CENTER - 1][IBFX];
    bfNeighbors[3] = bfNb[CENTER + 1][CENTER    ][IBFY];
    bfNeighbors[4] = bfNb[CENTER + 2][CENTER    ][IBFY];
    bfNeighbors[5] = bfNb[CENTER    ][CENTER    ][IBFY];
    // clang-format on

    slope_unsplit_mhd_2d(bfNeighbors, dbf);

    real_t dARy = 0.5 * dbfY[IX];

    // change neighbors to i, j+1 and recompute dbf
    // clang-format off
    bfNeighbors[0] = bfNb[CENTER    ][CENTER + 1][IBFX];
    bfNeighbors[1] = bfNb[CENTER    ][CENTER + 2][IBFX];
    bfNeighbors[2] = bfNb[CENTER    ][CENTER    ][IBFX];
    bfNeighbors[3] = bfNb[CENTER    ][CENTER + 1][IBFY];
    bfNeighbors[4] = bfNb[CENTER + 1][CENTER + 1][IBFY];
    bfNeighbors[5] = bfNb[CENTER - 1][CENTER + 1][IBFY];
    // clang-format on

    slope_unsplit_mhd_2d(bfNeighbors, dbf);

    real_t dBRx = 0.5 * dbfX[IY];

    // Cell centered slopes in normal direction
    real_t dAx = 0.5 * (AR - AL);
    real_t dBy = 0.5 * (BR - BL);

    // Source terms (including transverse derivatives)
    real_t sr0, su0, sv0, sw0, sp0, sA0, sB0, sC0;
    real_t sAL0, sAR0, sBL0, sBR0;

    if (true /*cartesian*/)
    {

      sr0 = (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
      su0 =
        (-u * dux - dpx / r - B * dBx / r - C * dCx / r) * dtdx + (-v * duy + B * dAy / r) * dtdy;
      sv0 =
        (-u * dvx + A * dBx / r) * dtdx + (-v * dvy - dpy / r - A * dAy / r - C * dCy / r) * dtdy;
      sw0 = (-u * dwx + A * dCx / r) * dtdx + (-v * dwy + B * dCy / r) * dtdy;
      sp0 = (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy;
      sA0 = (u * dBy + B * duy - v * dAy - A * dvy) * dtdy;
      sB0 = (-u * dBx - B * dux + v * dAx + A * dvx) * dtdx;
      sC0 = (w * dAx + A * dwx - u * dCx - C * dux) * dtdx +
            (-v * dCy - C * dvy + w * dBy + B * dwy) * dtdy;
      // if (Omega0 > ZERO_F) {
      // 	real_t shear = -1.5 * Omega0 * xPos;
      // 	sC0 += (shear * dAx - 1.5 * Omega0 * A) * dtdx;
      // 	sC0 +=  shear * dBy                     * dtdy;
      // }

      // Face centered B-field
      sAL0 = +(ELR - ELL) * 0.5 * dtdy;
      sAR0 = +(ERR - ERL) * 0.5 * dtdy;
      sBL0 = -(ERL - ELL) * 0.5 * dtdx;
      sBR0 = -(ERR - ELR) * 0.5 * dtdx;

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

    // Right state at left interface
    qp[0][ID] = r - drx;
    qp[0][IU] = u - dux;
    qp[0][IV] = v - dvx;
    qp[0][IW] = w - dwx;
    qp[0][IP] = p - dpx;
    qp[0][IA] = AL;
    qp[0][IB] = B - dBx;
    qp[0][IC] = C - dCx;
    qp[0][ID] = fmax(smallR, qp[0][ID]);
    qp[0][IP] = fmax(smallp * qp[0][ID], qp[0][IP]);

    // Left state at right interface
    qm[0][ID] = r + drx;
    qm[0][IU] = u + dux;
    qm[0][IV] = v + dvx;
    qm[0][IW] = w + dwx;
    qm[0][IP] = p + dpx;
    qm[0][IA] = AR;
    qm[0][IB] = B + dBx;
    qm[0][IC] = C + dCx;
    qm[0][ID] = fmax(smallR, qm[0][ID]);
    qm[0][IP] = fmax(smallp * qm[0][ID], qm[0][IP]);

    // Top state at bottom interface
    qp[1][ID] = r - dry;
    qp[1][IU] = u - duy;
    qp[1][IV] = v - dvy;
    qp[1][IW] = w - dwy;
    qp[1][IP] = p - dpy;
    qp[1][IA] = A - dAy;
    qp[1][IB] = BL;
    qp[1][IC] = C - dCy;
    qp[1][ID] = fmax(smallR, qp[1][ID]);
    qp[1][IP] = fmax(smallp * qp[1][ID], qp[1][IP]);

    // Bottom state at top interface
    qm[1][ID] = r + dry;
    qm[1][IU] = u + duy;
    qm[1][IV] = v + dvy;
    qm[1][IW] = w + dwy;
    qm[1][IP] = p + dpy;
    qm[1][IA] = A + dAy;
    qm[1][IB] = BR;
    qm[1][IC] = C + dCy;
    qm[1][ID] = fmax(smallR, qm[1][ID]);
    qm[1][IP] = fmax(smallp * qm[1][ID], qm[1][IP]);


    // Right-top state (RT->LL)
    qRT[ID] = r + (+drx + dry);
    qRT[IU] = u + (+dux + duy);
    qRT[IV] = v + (+dvx + dvy);
    qRT[IW] = w + (+dwx + dwy);
    qRT[IP] = p + (+dpx + dpy);
    qRT[IA] = AR + (+dARy);
    qRT[IB] = BR + (+dBRx);
    qRT[IC] = C + (+dCx + dCy);
    qRT[ID] = fmax(smallR, qRT[ID]);
    qRT[IP] = fmax(smallp * qRT[ID], qRT[IP]);

    // Right-Bottom state (RB->LR)
    qRB[ID] = r + (+drx - dry);
    qRB[IU] = u + (+dux - duy);
    qRB[IV] = v + (+dvx - dvy);
    qRB[IW] = w + (+dwx - dwy);
    qRB[IP] = p + (+dpx - dpy);
    qRB[IA] = AR + (-dARy);
    qRB[IB] = BL + (+dBLx);
    qRB[IC] = C + (+dCx - dCy);
    qRB[ID] = fmax(smallR, qRB[ID]);
    qRB[IP] = fmax(smallp * qRB[ID], qRB[IP]);

    // Left-Bottom state (LB->RR)
    qLB[ID] = r + (-drx - dry);
    qLB[IU] = u + (-dux - duy);
    qLB[IV] = v + (-dvx - dvy);
    qLB[IW] = w + (-dwx - dwy);
    qLB[IP] = p + (-dpx - dpy);
    qLB[IA] = AL + (-dALy);
    qLB[IB] = BL + (-dBLx);
    qLB[IC] = C + (-dCx - dCy);
    qLB[ID] = fmax(smallR, qLB[ID]);
    qLB[IP] = fmax(smallp * qLB[ID], qLB[IP]);

    // Left-Top state (LT->RL)
    qLT[ID] = r + (-drx + dry);
    qLT[IU] = u + (-dux + duy);
    qLT[IV] = v + (-dvx + dvy);
    qLT[IW] = w + (-dwx + dwy);
    qLT[IP] = p + (-dpx + dpy);
    qLT[IA] = AL + (+dALy);
    qLT[IB] = BR + (-dBRx);
    qLT[IC] = C + (-dCx + dCy);
    qLT[ID] = fmax(smallR, qLT[ID]);
    qLT[IP] = fmax(smallp * qLT[ID], qLT[IP]);

  } // trace_unsplit_mhd_2d

  /**
   * Compute primitive variables slope (vector dq) from q and its neighbors.
   * This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1 and 2.
   *
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in]  q       : current primitive variable state
   * \param[in]  qPlusX  : state in the next neighbor cell along XDIR
   * \param[in]  qMinusX : state in the previous neighbor cell along XDIR
   * \param[in]  qPlusY  : state in the next neighbor cell along YDIR
   * \param[in]  qMinusY : state in the previous neighbor cell along YDIR
   * \param[out] dqX     : reference to an array returning the X slopes
   * \param[out] dqY     : reference to an array returning the Y slopes
   *
   */
  KOKKOS_INLINE_FUNCTION
  void
  slope_unsplit_hydro_2d(const MHDState & q,
                         const MHDState & qPlusX,
                         const MHDState & qMinusX,
                         const MHDState & qPlusY,
                         const MHDState & qMinusY,
                         MHDState &       dqX,
                         MHDState &       dqY) const
  {

    real_t slope_type = params.settings.slope_type;

    if (slope_type == 0)
    {

      dqX[ID] = ZERO_F;
      dqX[IP] = ZERO_F;
      dqX[IU] = ZERO_F;
      dqX[IV] = ZERO_F;

      dqY[ID] = ZERO_F;
      dqY[IP] = ZERO_F;
      dqY[IU] = ZERO_F;
      dqY[IV] = ZERO_F;

      return;
    }

    if (slope_type == 1 || slope_type == 2)
    { // minmod or average

      slope_unsplit_hydro_2d_scalar(
        q[ID], qPlusX[ID], qMinusX[ID], qPlusY[ID], qMinusY[ID], &(dqX[ID]), &(dqY[ID]));
      slope_unsplit_hydro_2d_scalar(
        q[IP], qPlusX[IP], qMinusX[IP], qPlusY[IP], qMinusY[IP], &(dqX[IP]), &(dqY[IP]));
      slope_unsplit_hydro_2d_scalar(
        q[IU], qPlusX[IU], qMinusX[IU], qPlusY[IU], qMinusY[IU], &(dqX[IU]), &(dqY[IU]));
      slope_unsplit_hydro_2d_scalar(
        q[IV], qPlusX[IV], qMinusX[IV], qPlusY[IV], qMinusY[IV], &(dqX[IV]), &(dqY[IV]));

    } // end slope_type == 1 or 2

  } // slope_unsplit_hydro_2d

}; // class MHDBaseFunctor2D

} // namespace muscl

} // namespace euler_kokkos

#endif // MHD_BASE_FUNCTOR_2D_H_
