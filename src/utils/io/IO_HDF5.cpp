#include "IO_HDF5.h"

#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"

#include <fstream>

namespace euler_kokkos { namespace io {

// =======================================================
// =======================================================
void writeXdmfForHdf5Wrapper(HydroParams& params,
			     ConfigMap& configMap,
			     const std::map<int, std::string>& variables_names,
			     int totalNumberOfSteps,
			     bool singleStep)
{

  // domain (no-MPI) or sub-domain sizes (MPI)
  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;

#ifdef USE_MPI
  // sub-domain decomposition sizes
  const int mx = params.mx;
  const int my = params.my;
  const int mz = params.mz;
#endif
  
  const int ghostWidth = params.ghostWidth;

  const int dimType = params.dimType;

  const bool ghostIncluded = configMap.getBool("output","ghostIncluded",false);

  const bool mhdEnabled = params.mhdEnabled;

#ifdef USE_MPI
  // global sizes
  int nxg = mx*nx;
  int nyg = my*ny;
  int nzg = mz*nz;
#else  
  // data size actually written on disk
  int nxg = nx;
  int nyg = ny;
  int nzg = nz;
#endif // USE_MPI

  if (ghostIncluded) {
    nxg += (2*ghostWidth);
    nyg += (2*ghostWidth);
    nzg += (2*ghostWidth);
  }

#ifdef USE_MPI
  /*
   * The follwing only makes sense in MPI: is allghostIncluded is true,
   * every sub-domain dumps its own local ghosts (might be usefull for debug,
   * at least it was useful in ramsesGPU for the shearing box border condition
   * debug).
   */
  bool allghostIncluded = configMap.getBool("output","allghostIncluded",false);
  if (allghostIncluded) {
    nxg = mx*(nx+2*ghostWidth);
    nyg = my*(ny+2*ghostWidth);
    nzg = mz*(nz+2*ghostWidth);
  }

  /*
   * Let MPIIO underneath hdf5 re-assemble the pieces and provides a single 
   * nice file. Thanks parallel HDF5 !
   */
  bool reassembleInFile = configMap.getBool("output", "reassembleInFile", true);
  if (!reassembleInFile) {
    if (dimType==TWO_D) {
      if (allghostIncluded or ghostIncluded) {
	nxg = (nx+2*ghostWidth);
	nyg = (ny+2*ghostWidth)*mx*my;
      } else {
	nxg = nx;
	nyg = ny*mx*my;
      }
    } else {
      if (allghostIncluded or ghostIncluded) {
	nxg = nx+2*ghostWidth;
	nyg = ny+2*ghostWidth;
	nzg = (nz+2*ghostWidth)*mx*my*mz;
      } else {
	nxg = nx;
	nyg = ny;
	nzg = nz*mx*my*mz;
      }
    }
  }
#endif // USE_MPI
  
  // get data type as a string for Xdmf
  std::string dataTypeName;
  if (sizeof(real_t) == sizeof(float))
    dataTypeName = "Float";
  else
    dataTypeName = "Double";

  /*
   * 1. open XDMF and write header lines
   */
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
  std::string xdmfFilename = outputPrefix+".xmf";
  if (singleStep) { // add iStep to file name
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << totalNumberOfSteps;
    xdmfFilename = outputPrefix+"_"+outNum.str()+".xmf";
  }
  std::fstream xdmfFile;
  xdmfFile.open(xdmfFilename.c_str(), std::ios_base::out);

  xdmfFile << "<?xml version=\"1.0\" ?>"                       << std::endl;
  xdmfFile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>"         << std::endl;
  xdmfFile << "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">" << std::endl;
  xdmfFile << "  <Domain>"                                     << std::endl;
  xdmfFile << "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">" << std::endl;

  // for each time step write a <grid> </grid> item
  int startStep=0;
  int stopStep =totalNumberOfSteps;
  int deltaStep=1;
  if (params.nOutput == -1)
    deltaStep=1;
  
  if (singleStep) {
    startStep = totalNumberOfSteps;
    stopStep  = totalNumberOfSteps+1;
    deltaStep = 1;
  }

  for (int iStep=startStep; iStep<=stopStep; iStep+=deltaStep) {
 
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << iStep;

    // take care that the following filename must be exactly the same as in routine outputHdf5 !!!
    std::string baseName         = outputPrefix+"_"+outNum.str();
    std::string hdf5Filename     = outputPrefix+"_"+outNum.str()+".h5";
    std::string hdf5FilenameFull = outputDir+"/"+outputPrefix+"_"+outNum.str()+".h5";

    xdmfFile << "    <Grid Name=\"" << baseName << "\" GridType=\"Uniform\">" << std::endl;
    xdmfFile << "    <Time Value=\"" << iStep << "\" />"                      << std::endl;

    // topology CoRectMesh
    if (dimType == TWO_D) 
      xdmfFile << "      <Topology TopologyType=\"2DCoRectMesh\" NumberOfElements=\"" << nyg << " " << nxg << "\"/>" << std::endl;
    else
      xdmfFile << "      <Topology TopologyType=\"3DCoRectMesh\" NumberOfElements=\"" << nzg << " " << nyg << " " << nxg << "\"/>" << std::endl;
      
    // geometry
    if (dimType == TWO_D) {
      xdmfFile << "    <Geometry Type=\"ORIGIN_DXDY\">"        << std::endl;
      xdmfFile << "    <DataStructure"                         << std::endl;
      xdmfFile << "       Name=\"Origin\""                     << std::endl;
      xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
      xdmfFile << "       Dimensions=\"2\""                    << std::endl;
      xdmfFile << "       Format=\"XML\">"                     << std::endl;
      xdmfFile << "       0 0"                                 << std::endl;
      xdmfFile << "    </DataStructure>"                       << std::endl;
      xdmfFile << "    <DataStructure"                         << std::endl;
      xdmfFile << "       Name=\"Spacing\""                    << std::endl;
      xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
      xdmfFile << "       Dimensions=\"2\""                    << std::endl;
      xdmfFile << "       Format=\"XML\">"                     << std::endl;
      xdmfFile << "       1 1"                                 << std::endl;
      xdmfFile << "    </DataStructure>"                       << std::endl;
      xdmfFile << "    </Geometry>"                            << std::endl;
    } else {
      xdmfFile << "    <Geometry Type=\"ORIGIN_DXDYDZ\">"      << std::endl;
      xdmfFile << "    <DataStructure"                         << std::endl;
      xdmfFile << "       Name=\"Origin\""                     << std::endl;
      xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
      xdmfFile << "       Dimensions=\"3\""                    << std::endl;
      xdmfFile << "       Format=\"XML\">"                     << std::endl;
      xdmfFile << "       0 0 0"                               << std::endl;
      xdmfFile << "    </DataStructure>"                       << std::endl;
      xdmfFile << "    <DataStructure"                         << std::endl;
      xdmfFile << "       Name=\"Spacing\""                    << std::endl;
      xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
      xdmfFile << "       Dimensions=\"3\""                    << std::endl;
      xdmfFile << "       Format=\"XML\">"                     << std::endl;
      xdmfFile << "       1 1 1"                               << std::endl;
      xdmfFile << "    </DataStructure>"                       << std::endl;
      xdmfFile << "    </Geometry>"                            << std::endl;
    }
      
    // density
    xdmfFile << "      <Attribute Center=\"Node\" Name=\""<< variables_names.at(ID) << "\">" << std::endl;
    xdmfFile << "        <DataStructure"                             << std::endl;
    xdmfFile << "           DataType=\"" << dataTypeName <<  "\""    << std::endl;
    if (dimType == TWO_D)
      xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
    else
      xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
    xdmfFile << "           Format=\"HDF\">"                         << std::endl;
    xdmfFile << "           "<<hdf5Filename<<":/" << variables_names.at(ID) <<""             << std::endl;
    xdmfFile << "        </DataStructure>"                           << std::endl;
    xdmfFile << "      </Attribute>"                                 << std::endl;
      
    // energy
    xdmfFile << "      <Attribute Center=\"Node\" Name=\"" << variables_names.at(IE) << "\">" << std::endl;
    xdmfFile << "        <DataStructure"                              << std::endl;
    xdmfFile << "           DataType=\"" << dataTypeName <<  "\""     << std::endl;
    if (dimType == TWO_D)
      xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
    else
      xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
    xdmfFile << "           Format=\"HDF\">"                          << std::endl;
    xdmfFile << "           "<<hdf5Filename<<":/"<<  variables_names.at(IE) <<""             << std::endl;
    xdmfFile << "        </DataStructure>"                            << std::endl;
    xdmfFile << "      </Attribute>"                                  << std::endl;
      
    // momentum X
    xdmfFile << "      <Attribute Center=\"Node\" Name=\"" << variables_names.at(IU) << "\">" << std::endl;
    xdmfFile << "        <DataStructure"                                << std::endl;
    xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
    if (dimType == TWO_D)
      xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
    else
      xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
    xdmfFile << "           Format=\"HDF\">"                            << std::endl;
    xdmfFile << "           "<<hdf5Filename<<":/"<<  variables_names.at(IU) <<""             << std::endl;
    xdmfFile << "        </DataStructure>"                              << std::endl;
    xdmfFile << "      </Attribute>"                                    << std::endl;
      
    // momentum Y
    xdmfFile << "      <Attribute Center=\"Node\" Name=\"" << variables_names.at(IV) << "\">" << std::endl;
    xdmfFile << "        <DataStructure" << std::endl;
    xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
    if (dimType == TWO_D)
      xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
    else
      xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
    xdmfFile << "           Format=\"HDF\">"                            << std::endl;
    xdmfFile << "           "<<hdf5Filename<<":/"<<  variables_names.at(IV) <<""             << std::endl;
    xdmfFile << "        </DataStructure>"                              << std::endl;
    xdmfFile << "      </Attribute>"                                    << std::endl;
      
    // momentum Z
    if (dimType == THREE_D and !mhdEnabled) {
      xdmfFile << "      <Attribute Center=\"Node\" Name=\"" << variables_names.at(IW) << "\">" << std::endl;
      xdmfFile << "        <DataStructure"                                << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
      xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                            << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/"<< variables_names.at(IW) <<""             << std::endl;
      xdmfFile << "        </DataStructure>"                              << std::endl;
      xdmfFile << "      </Attribute>"                                    << std::endl;
    }
      
    if (mhdEnabled) {
      // momentum Z
      xdmfFile << "      <Attribute Center=\"Node\" Name=\"" << variables_names.at(IW) <<"\">" << std::endl;
      xdmfFile << "        <DataStructure" << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
      if (dimType == TWO_D)
	xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
      else
	xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                            << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/"<<  variables_names.at(IW) <<""             << std::endl;
      xdmfFile << "        </DataStructure>"                              << std::endl;
      xdmfFile << "      </Attribute>"                                    << std::endl;

      // magnetic field X
      xdmfFile << "      <Attribute Center=\"Node\" Name=\"" << variables_names.at(IA) << "\">" << std::endl;
      xdmfFile << "        <DataStructure" << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
      if (dimType == TWO_D)
	xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
      else
	xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                            << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/"<<  variables_names.at(IA) <<""             << std::endl;
      xdmfFile << "        </DataStructure>"                              << std::endl;
      xdmfFile << "      </Attribute>"                                    << std::endl;
	
      // magnetic field Y
      xdmfFile << "      <Attribute Center=\"Node\" Name=\""<< variables_names.at(IB) <<"\">" << std::endl;
      xdmfFile << "        <DataStructure" << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
      if (dimType == TWO_D)
	xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
      else
	xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                            << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/"<< variables_names.at(IB) <<""             << std::endl;
      xdmfFile << "        </DataStructure>"                              << std::endl;
      xdmfFile << "      </Attribute>"                                    << std::endl;
	
      // magnetic field Z
      xdmfFile << "      <Attribute Center=\"Node\" Name=\"" << variables_names.at(IC) <<"\">" << std::endl;
      xdmfFile << "        <DataStructure" << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
      if (dimType == TWO_D)
	xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
      else
	xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                            << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/"<< variables_names.at(IC) <<""             << std::endl;
      xdmfFile << "        </DataStructure>"                              << std::endl;
      xdmfFile << "      </Attribute>"                                    << std::endl;
	
    } // end mhdEnabled

      // finalize grid file for the current time step
    xdmfFile << "   </Grid>" << std::endl;
      
  } // end for loop over time step

  // finalize Xdmf wrapper file
  xdmfFile << "   </Grid>" << std::endl;
  xdmfFile << " </Domain>" << std::endl;
  xdmfFile << "</Xdmf>"    << std::endl;

} // writeXdmfForHdf5Wrapper

} // namespace io

} // namespace euler_kokkos
