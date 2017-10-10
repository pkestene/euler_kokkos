#ifndef SHARED_ENUMS_H_
#define SHARED_ENUMS_H_

//! dimension of the problem
enum DimensionType {
  TWO_D = 2, 
  THREE_D = 3,
  DIM2 = 2,
  DIM3 = 3
};

//! hydro field indexes
enum VarIndex {
  ID=0,   /*!< ID Density field index */
  IP=1,   /*!< IP Pressure/Energy field index */
  IE=1,   /*!< IE Energy/Pressure field index */
  IU=2,   /*!< X velocity / momentum index */
  IV=3,   /*!< Y velocity / momentum index */ 
  IW=4,   /*!< Z velocity / momentum index */ 
  IA=5,   /*!< X magnetic field index */ 
  IB=6,   /*!< Y magnetic field index */ 
  IC=7,   /*!< Z magnetic field index */ 
  IBX=5,  /*!< X magnetic field index */ 
  IBY=6,  /*!< Y magnetic field index */ 
  IBZ=7,   /*!< Z magnetic field index */
  IBFX = 0,
  IBFY = 1,
  IBFZ = 2
};

// //! enum class to enumerate field location in a velocity/temperature gradient array in 2d at solution points
// enum class VarIndexGrad2d {
//   IGU = 0, /* x-component of velocity gradient */
//   IGV = 1, /* y-component of velocity gradient */
//   IGW = -1, /* UNUSED */
//   IGT = 2 /* temperature gradient component */
// };

// //! enum class to enumerate field location in a velocity/temperature gradient array in 3d at solution points
// enum class VarIndexGrad3d {
//   IGU = 0, /* x-component of velocity gradient */
//   IGV = 1, /* y-component of velocity gradient */
//   IGW = 2, /* y-component of velocity gradient */
//   IGT = 3  /* temperature gradient component */
// };

//! enum class to enumerate field location in a velocity / velocity tensor gradient array in 2d at flux points
enum class VarIndexGrad2d {
  IGU  = 0, /* x-component of velocity */
  IGV  = 1, /* y-component of velocity */
  IGW  =-1, /* UNUSED */
  IGUX = 2, /* partial U over partial x */
  IGUY = 3, /* partial U over partial y */
  IGUZ =-1, /* UNUSED */
  IGVX = 4, /* partial V over partial x */
  IGVY = 5, /* partial V over partial y */
  IGVZ =-1, /* UNUSED */
  IGWX =-1, /* UNUSED */
  IGWY =-1, /* UNUSED */
  IGWZ =-1, /* UNUSED */
  IGT  = 6  /* temperature gradient component */
};

//! enum class to enumerate field location in a velocity / velocity tensor gradient array in 3d at flux points
enum class VarIndexGrad3d {
  IGU  = 0, /* x-component of velocity */
  IGV  = 1, /* y-component of velocity */
  IGW  = 2, /* z-component of velocity */
  IGUX = 3, /* partial U over partial x */
  IGUY = 4, /* partial U over partial y */
  IGUZ = 5, /* partial U over partial z */
  IGVX = 6, /* partial V over partial x */
  IGVY = 7, /* partial V over partial y */
  IGVZ = 8, /* partial V over partial z */
  IGWX = 9, /* partial W over partial x */
  IGWY =10, /* partial W over partial y */
  IGWZ =11, /* partial W over partial z */
  IGT  =12  /* temperature gradient component */
};

//! velocity gradient tensor components in 2d
enum class gradientV_IDS_2d {
  U_X =  0,
  U_Y =  1,
  U_Z = -1,

  V_X =  2,
  V_Y =  3,
  V_Z = -1,

  W_X = -1,
  W_Y = -1,
  W_Z = -1,
}; // enum class gradientV_IDS_2d

//! velocity gradient tensor components in 3d
enum class gradientV_IDS_3d {
  U_X = 0,
  U_Y = 1,
  U_Z = 2,
    
  V_X = 3,
  V_Y = 4,
  V_Z = 5,
    
  W_X = 6,
  W_Y = 7,
  W_Z = 8
}; // enum class gradientV_IDS_3d

//! face index
enum FaceIdType {
  FACE_XMIN=0,
  FACE_XMAX=1,
  FACE_YMIN=2,
  FACE_YMAX=3,
  FACE_ZMIN=4,
  FACE_ZMAX=5,
  FACE_MIN =0,
  FACE_MAX =1
};

//! Riemann solver type for hydro fluxes
enum RiemannSolverType {
  RIEMANN_APPROX, /*!< quasi-exact Riemann solver (hydro-only) */ 
  RIEMANN_LLF,    /*!< LLF Local Lax-Friedrich */
  RIEMANN_HLL,    /*!< HLL hydro and MHD Riemann solver */
  RIEMANN_HLLC,   /*!< HLLC hydro-only Riemann solver */
  RIEMANN_HLLD    /*!< HLLD MHD-only Riemann solver */
};

//! type of boundary condition (note that BC_COPY is only used in the
//! MPI version for inside boundary)
enum BoundaryConditionType {
  BC_UNDEFINED, 
  BC_DIRICHLET,   /*!< reflecting border condition */
  BC_NEUMANN,     /*!< absorbing border condition */
  BC_PERIODIC,    /*!< periodic border condition */
  BC_COPY         /*!< only used in MPI parallelized version */
};

//! enum component index
enum ComponentIndex3D {
  IX = 0,
  IY = 1,
  IZ = 2
};

//! direction used in directional splitting scheme
enum Direction {
  XDIR=1, 
  YDIR=2,
  ZDIR=3,
  DIR_X = 0,
  DIR_Y = 1,
  DIR_Z = 2
};

//! location of the outside boundary
enum BoundaryLocation {
  XMIN = 0, 
  XMAX = 1, 
  YMIN = 2, 
  YMAX = 3,
  ZMIN = 4,
  ZMAX = 5
};

//! enum edge index (use in MHD - EMF computations)
enum EdgeIndex {
  IRT = 0, /*!< RT (Right - Top   ) */
  IRB = 1, /*!< RB (Right - Bottom) */
  ILT = 2, /*!< LT (Left  - Top   ) */
  ILB = 3  /*!< LB (Left  - Bottom) */
};

enum EdgeIndex2 {
  ILL = 0,
  IRL = 1,
  ILR = 2,
  IRR = 3
};

//! enum used in MHD - EMF computations
enum EmfDir {
  EMFX = 0,
  EMFY = 1,
  EMFZ = 2
};

//! EMF indexes (EMFZ is first because in 2D, we only need EMFZ)
enum EmfIndex {
  I_EMFZ=0,
  I_EMFY=1,
  I_EMFX=2
};

//! implementation version
enum ImplementationVersion {
  IMPL_VERSION_0,
  IMPL_VERSION_1,
  IMPL_VERSION_2
};

//! problem type
enum ProblemType {
  PROBLEM_IMPLODE,
  PROBLEM_BLAST,
  PROBLEM_ORSZAG_TANG
};

#endif // SHARED_ENUMS_H_
