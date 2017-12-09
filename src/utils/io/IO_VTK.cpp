#include "IO_VTK.h"

#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"

#include <fstream>

namespace euler_kokkos { namespace io {

// =======================================================
// =======================================================
static bool isBigEndian()
{
  const int i = 1;
  return ( (*(char*)&i) == 0 );
}

// =======================================================
// =======================================================
void save_VTK_2D(DataArray2d             Udata,
		 DataArray2d::HostMirror Uhost,
		 HydroParams& params,
		 ConfigMap& configMap,
		 int nbvar,
		 const std::map<int, std::string>& variables_names,
		 int iStep,
		 std::string debug_name)
{
  const int nx = params.nx;
  const int ny = params.ny;

  const int imin = params.imin;
  const int imax = params.imax;

  const int jmin = params.jmin;
  const int jmax = params.jmax;

  const int ghostWidth = params.ghostWidth;

  const int isize = params.isize;
  const int jsize = params.jsize;
  const int nbCells = isize * jsize;
  
  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);
  
  // local variables
  int i,j,iVar;
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);

  // check scalar data type
  bool useDouble = false;

  if (sizeof(real_t) == sizeof(double)) {
    useDouble = true;
  }
  
  // write iStep in string stepNum
  std::ostringstream stepNum;
  stepNum.width(7);
  stepNum.fill('0');
  stepNum << iStep;
  
  // concatenate file prefix + file number + suffix
  std::string filename;
  if ( debug_name.empty() )
    filename = outputDir + "/" + outputPrefix + "_" + stepNum.str() + ".vti";
  else
    filename = outputDir + "/" + outputPrefix + "_" + debug_name + "_" + stepNum.str() + ".vti";
  
  // open file 
  std::fstream outFile;
  outFile.open(filename.c_str(), std::ios_base::out);
  
  // write header

  // if writing raw binary data (file does not respect XML standard)
  if (outputVtkAscii)
    outFile << "<?xml version=\"1.0\"?>\n";

  // write xml data header
  if (isBigEndian()) {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  } else {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  }

  // write mesh extent
  outFile << "  <ImageData WholeExtent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << 1  << " "
	  <<  "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
  outFile << "  <Piece Extent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << 1  << " "    
	  << "\">\n";
  
  outFile << "    <PointData>\n";
  outFile << "    </PointData>\n";

  if (outputVtkAscii) {

    outFile << "    <CellData>\n";

    // write data array (ascii), remove ghost cells
    for ( iVar=0; iVar<nbvar; iVar++) {
      outFile << "    <DataArray type=\"";
      if (useDouble)
	outFile << "Float64";
      else
	outFile << "Float32";
      outFile << "\" Name=\"" << variables_names.at(iVar) << "\" format=\"ascii\" >\n";
      
      for (int index=0; index<nbCells; ++index) {
	//index2coord(index,i,j,isize,jsize);
	
	// enforce the use of left layout (Ok for CUDA)
	// but for OpenMP, we will need to transpose
	j = index / isize;
	i = index - j*isize;
	
	if (j>=jmin+ghostWidth and j<=jmax-ghostWidth and
	    i>=imin+ghostWidth and i<=imax-ghostWidth) {
	  outFile << Uhost(i, j, iVar) << " ";
	}
      }
      outFile << "\n    </DataArray>\n";
    } // end for iVar

    outFile << "    </CellData>\n";
    
    // write footer
    outFile << "  </Piece>\n";
    outFile << "  </ImageData>\n";
    outFile << "</VTKFile>\n";

  } else { // write data in binary format

    outFile << "    <CellData>" << std::endl;

    for (int iVar=0; iVar<nbvar; iVar++) {
      if (useDouble) {
	outFile << "     <DataArray type=\"Float64\" Name=\"" ;
      } else {
	outFile << "     <DataArray type=\"Float32\" Name=\"" ;
      }
      outFile << variables_names.at(iVar)
	      << "\" format=\"appended\" offset=\""
	      << iVar*nx*ny*sizeof(real_t)+iVar*sizeof(unsigned int)
	      <<"\" />" << std::endl;
    }

    outFile << "    </CellData>" << std::endl;
    outFile << "  </Piece>" << std::endl;
    outFile << "  </ImageData>" << std::endl;
    
    outFile << "  <AppendedData encoding=\"raw\">" << std::endl;

    // write the leading undescore
    outFile << "_";
    // then write heavy data (column major format)
    {
      unsigned int nbOfWords = nx*ny*sizeof(real_t);
      for (int iVar=0; iVar<nbvar; iVar++) {
	outFile.write((char *)&nbOfWords,sizeof(unsigned int));
	for (int j=jmin+ghostWidth; j<=jmax-ghostWidth; j++)
	  for (int i=imin+ghostWidth; i<=imax-ghostWidth; i++) {
	    real_t tmp = Uhost(i, j, iVar);
	    outFile.write((char *)&tmp,sizeof(real_t));
	  }
      }
    }

    outFile << "  </AppendedData>" << std::endl;
    outFile << "</VTKFile>" << std::endl;

  } // end ascii/binary heavy data write

  
  outFile.close();
  
} // end save_VTK_2D

// =======================================================
// =======================================================
void save_VTK_3D(DataArray3d             Udata,
		 DataArray3d::HostMirror Uhost,
		 HydroParams& params,
		 ConfigMap& configMap,
		 int nbvar,
		 const std::map<int, std::string>& variables_names,
		 int iStep,
		 std::string debug_name)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;

  const int imin = params.imin;
  const int imax = params.imax;

  const int jmin = params.jmin;
  const int jmax = params.jmax;

  const int kmin = params.kmin;
  const int kmax = params.kmax;

  const int isize = params.isize;
  const int jsize = params.jsize;
  const int ksize = params.ksize;
  const int ijsize = isize * jsize;
  const int nbCells = isize * jsize * ksize;

  
  const int ghostWidth = params.ghostWidth;
  
  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);
  
  // local variables
  int i, j, k, iVar;
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    
  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);

  // check scalar data type
  bool useDouble = false;

  if (sizeof(real_t) == sizeof(double)) {
    useDouble = true;
  }
  
  // write iStep in string stepNum
  std::ostringstream stepNum;
  stepNum.width(7);
  stepNum.fill('0');
  stepNum << iStep;
  
  // concatenate file prefix + file number + suffix
  std::string filename;
  if ( debug_name.empty() )
    filename = outputDir + "/" + outputPrefix + "_" + stepNum.str() + ".vti";
  else
    filename = outputDir + "/" + outputPrefix + "_" + debug_name + "_" + stepNum.str() + ".vti";
  
  // open file 
  std::fstream outFile;
  outFile.open(filename.c_str(), std::ios_base::out);
  
  // write header

  // if writing raw binary data (file does not respect XML standard)
  if (outputVtkAscii)
    outFile << "<?xml version=\"1.0\"?>\n";
  
  // write xml data header
  if (isBigEndian()) {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  } else {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  }

  // write mesh extent
  outFile << "  <ImageData WholeExtent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << nz  << " "
	  <<  "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
  outFile << "  <Piece Extent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << nz << " "    
	  << "\">\n";
  
  outFile << "    <PointData>\n";
  outFile << "    </PointData>\n";

  if (outputVtkAscii) {
    
    outFile << "    <CellData>\n";
    
    // write data array (ascii), remove ghost cells
    for ( iVar=0; iVar<nbvar; iVar++) {
      outFile << "    <DataArray type=\"";
      if (useDouble)
	outFile << "Float64";
      else
	outFile << "Float32";
      outFile << "\" Name=\"" << variables_names.at(iVar) << "\" format=\"ascii\" >\n";
      
      for (int index=0; index<nbCells; ++index) {
	//index2coord(index,i,j,k,isize,jsize,ksize);
	
	// enforce the use of left layout (Ok for CUDA)
	// but for OpenMP, we will need to transpose
	k = index / ijsize;
	j = (index - k*ijsize) / isize;
	i = index - j*isize - k*ijsize;
	
	if (k>=kmin+ghostWidth and k<=kmax-ghostWidth and
	    j>=jmin+ghostWidth and j<=jmax-ghostWidth and
	    i>=imin+ghostWidth and i<=imax-ghostWidth) {
    	outFile << Uhost(i,j,k,iVar) << " ";
	}
      }
      outFile << "\n    </DataArray>\n";
    } // end for iVar

    outFile << "    </CellData>\n";

    // write footer
    outFile << "  </Piece>\n";
    outFile << "  </ImageData>\n";
    outFile << "</VTKFile>\n";

  } else { // write data in binary format

    outFile << "    <CellData>" << std::endl;

    for (int iVar=0; iVar<nbvar; iVar++) {
      if (useDouble) {
	outFile << "     <DataArray type=\"Float64\" Name=\"" ;
      } else {
	outFile << "     <DataArray type=\"Float32\" Name=\"" ;
      }
      outFile << variables_names.at(iVar)
	      << "\" format=\"appended\" offset=\""
	      << iVar*nx*ny*nz*sizeof(real_t)+iVar*sizeof(unsigned int)
	      <<"\" />" << std::endl;
    }

    outFile << "    </CellData>" << std::endl;
    outFile << "  </Piece>" << std::endl;
    outFile << "  </ImageData>" << std::endl;
    
    outFile << "  <AppendedData encoding=\"raw\">" << std::endl;

    // write the leading undescore
    outFile << "_";

    // then write heavy data (column major format)
    {
      unsigned int nbOfWords = nx*ny*nz*sizeof(real_t);
      for (int iVar=0; iVar<nbvar; iVar++) {
	outFile.write((char *)&nbOfWords,sizeof(unsigned int));
	 for (int k=kmin+ghostWidth; k<=kmax-ghostWidth; k++)
	   for (int j=jmin+ghostWidth; j<=jmax-ghostWidth; j++)
	     for (int i=imin+ghostWidth; i<=imax-ghostWidth; i++) {
	       real_t tmp = Uhost(i, j, k, iVar);
	       outFile.write((char *)&tmp,sizeof(real_t));
	     }
      }
    }

    outFile << "  </AppendedData>" << std::endl;
    outFile << "</VTKFile>" << std::endl;

  } // end ascii/binary heavy data write
  
  outFile.close();

} // end save_VTK_3D

#ifdef USE_MPI
// =======================================================
// =======================================================
void save_VTK_2D_mpi(DataArray2d             Udata,
		     DataArray2d::HostMirror Uhost,
		     HydroParams& params,
		     ConfigMap& configMap,
		     int nbvar,
		     const std::map<int, std::string>& variables_names,
		     int iStep,
		     std::string debug_name)
{
  
  const int nx = params.nx;
  const int ny = params.ny;

  const int imin = params.imin;
  const int imax = params.imax;

  const int jmin = params.jmin;
  const int jmax = params.jmax;
  
  const int ghostWidth = params.ghostWidth;

  const real_t dx = params.dx;
  const real_t dy = params.dy;
  const real_t dz = dx;
  
  const int isize = params.isize;
  const int jsize = params.jsize;
  const int nbCells = isize*jsize;

  int xmin=0, xmax=0, ymin=0, ymax=0;

  xmin=params.myMpiPos[0]*nx   ;
  xmax=params.myMpiPos[0]*nx+nx;
  ymin=params.myMpiPos[1]*ny   ;
  ymax=params.myMpiPos[1]*ny+ny;
  
  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);
  
  // local variables
  int i,j,iVar;
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);

  // check scalar data type
  bool useDouble = false;

  if (sizeof(real_t) == sizeof(double)) {
    useDouble = true;
  }
  
  // write iStep in string timeFormat
  std::ostringstream timeFormat;
  timeFormat.width(7);
  timeFormat.fill('0');
  timeFormat << iStep;

  // write MPI rank in string rankFormat
  std::ostringstream rankFormat;
  rankFormat.width(5);
  rankFormat.fill('0');
  rankFormat << params.myRank;

  // concatenate file prefix + file number + suffix
  std::string filename;
  if ( debug_name.empty() )
    filename = outputDir+"/"+outputPrefix+"_time"+timeFormat.str()+"_mpi"+rankFormat.str()+".vti";
  else
    filename = outputDir+"/"+outputPrefix+"_"+debug_name +"_time"+timeFormat.str()+"_mpi"+rankFormat.str()+".vti";

  // header file : parallel vti format
  std::string headerFilename   = outputDir+"/"+outputPrefix+"_time"+timeFormat.str()+".pvti";

  
  // open file 
  std::fstream outFile;
  outFile.open(filename.c_str(), std::ios_base::out);
  
  // write header
  if (params.myRank == 0) {
    write_pvti_header(headerFilename,
		      outputPrefix,
		      params,
		      nbvar,
		      variables_names,
		      iStep);
  }
      
  // if writing raw binary data (file does not respect XML standard)
  if (outputVtkAscii)
    outFile << "<?xml version=\"1.0\"?>\n";

  // write xml data header
  if (isBigEndian()) {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  } else {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  }

  // write mesh extent
  outFile << "  <ImageData WholeExtent=\""
	  << xmin << " " << xmax << " " 
	  << ymin << " " << ymax << " " 
	  << 0    << " " << 1    << ""
	  << "\" Origin=\"0 0 0\" Spacing=\"" << dx << " " << dy << " " << dz << "\">" << std::endl;
  outFile << "  <Piece Extent=\"" 
	  << xmin << " " << xmax << " " 
	  << ymin << " " << ymax << " " 
	  << 0    << " " << 1    << ""
	  << "\">" << std::endl;
  
  outFile << "    <PointData>\n";
  outFile << "    </PointData>\n";

  if (outputVtkAscii) {

    outFile << "    <CellData>\n";

    // write data array (ascii), remove ghost cells
    for ( iVar=0; iVar<nbvar; iVar++) {
      outFile << "    <DataArray type=\"";
      if (useDouble)
	outFile << "Float64";
      else
	outFile << "Float32";
      outFile << "\" Name=\"" << variables_names.at(iVar) << "\" format=\"ascii\" >\n";
      	
      for (int index=0; index<nbCells; ++index) {
	//index2coord(index,i,j,isize,jsize);
	
	// enforce the use of left layout (Ok for CUDA)
	// but for OpenMP, we will need to transpose
	j = index / isize;
	i = index - j*isize;
	
	if (j>=jmin+ghostWidth and j<=jmax-ghostWidth and
	    i>=imin+ghostWidth and i<=imax-ghostWidth) {
	  outFile << Uhost(i, j, iVar) << " ";
	}
      }
      
      outFile << "\n    </DataArray>\n";
      
    } // end for iVar
    
    outFile << "    </CellData>\n";
    
    // write footer
    outFile << "  </Piece>\n";
    outFile << "  </ImageData>\n";
    outFile << "</VTKFile>\n";

  } else { // write data in binary format

    outFile << "    <CellData>" << std::endl;

    for (int iVar=0; iVar<nbvar; iVar++) {
      if (useDouble) {
	outFile << "     <DataArray type=\"Float64\" Name=\"" ;
      } else {
	outFile << "     <DataArray type=\"Float32\" Name=\"" ;
      }
      outFile << variables_names.at(iVar)
	      << "\" format=\"appended\" offset=\""
	      << iVar*nx*ny*sizeof(real_t)+iVar*sizeof(unsigned int)
	      <<"\" />" << std::endl;
    }

    outFile << "    </CellData>" << std::endl;
    outFile << "  </Piece>" << std::endl;
    outFile << "  </ImageData>" << std::endl;
    
    outFile << "  <AppendedData encoding=\"raw\">" << std::endl;

    // write the leading undescore
    outFile << "_";
    // then write heavy data (column major format)
    {
      unsigned int nbOfWords = nx*ny*sizeof(real_t);
      for (int iVar=0; iVar<nbvar; iVar++) {
	outFile.write((char *)&nbOfWords,sizeof(unsigned int));
	for (int j=jmin+ghostWidth; j<=jmax-ghostWidth; j++)
	  for (int i=imin+ghostWidth; i<=imax-ghostWidth; i++) {
	    real_t tmp = Uhost(i, j, iVar);
	    outFile.write((char *)&tmp,sizeof(real_t));
	  }
      }
    }

    outFile << "  </AppendedData>" << std::endl;
    outFile << "</VTKFile>" << std::endl;

  } // end ascii/binary heavy data write
  
  outFile.close();
  
} // save_VTK_2D_mpi

// =======================================================
// =======================================================
/**
 * \param[in] Udata device data to save
 * \param[in,out] Uhost host data temporary array before saving to file
 */
void save_VTK_3D_mpi(DataArray3d             Udata,
		     DataArray3d::HostMirror Uhost,
		     HydroParams& params,
		     ConfigMap& configMap,
		     int nbvar,
		     const std::map<int, std::string>& variables_names,
		     int iStep,
		     std::string debug_name)
{
  
  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;

  const int imin = params.imin;
  const int imax = params.imax;

  const int jmin = params.jmin;
  const int jmax = params.jmax;

  const int kmin = params.kmin;
  const int kmax = params.kmax;

  const int ghostWidth = params.ghostWidth;

  const real_t dx = params.dx;
  const real_t dy = params.dy;
  const real_t dz = params.dz;

  const int isize = params.isize;
  const int jsize = params.jsize;
  const int ksize = params.ksize;
  const int ijsize = isize*jsize;
  const int nbCells = isize*jsize*ksize;

  int xmin=0, xmax=0, ymin=0, ymax=0, zmin=0, zmax=0;
  xmin=params.myMpiPos[0]*nx   ;
  xmax=params.myMpiPos[0]*nx+nx;
  ymin=params.myMpiPos[1]*ny   ;
  ymax=params.myMpiPos[1]*ny+ny;
  zmin=params.myMpiPos[2]*nz   ;
  zmax=params.myMpiPos[2]*nz+nz;

  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);
  
  // local variables
  int i,j,k,iVar;
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);

  // check scalar data type
  bool useDouble = false;

  if (sizeof(real_t) == sizeof(double)) {
    useDouble = true;
  }
  
  // write iStep in string timeFormat
  std::ostringstream timeFormat;
  timeFormat.width(7);
  timeFormat.fill('0');
  timeFormat << iStep;

  // write MPI rank in string rankFormat
  std::ostringstream rankFormat;
  rankFormat.width(5);
  rankFormat.fill('0');
  rankFormat << params.myRank;

  // concatenate file prefix + file number + suffix
  std::string filename;
  if ( debug_name.empty() )
    filename = outputDir+"/"+outputPrefix+"_time"+timeFormat.str()+"_mpi"+rankFormat.str()+".vti";
  else
    filename = outputDir+"/"+outputPrefix+"_"+debug_name +"_time"+timeFormat.str()+"_mpi"+rankFormat.str()+".vti";

  // header file : parallel vti format
  std::string headerFilename   = outputDir+"/"+outputPrefix+"_time"+timeFormat.str()+".pvti";

  
  // open file 
  std::fstream outFile;
  outFile.open(filename.c_str(), std::ios_base::out);
  
  // write header
  if (params.myRank == 0) {
    write_pvti_header(headerFilename,
		      outputPrefix,
		      params,
		      nbvar,
		      variables_names,
		      iStep);
  }
      
  // if writing raw binary data (file does not respect XML standard)
  if (outputVtkAscii)
    outFile << "<?xml version=\"1.0\"?>\n";

  // write xml data header
  if (isBigEndian()) {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  } else {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  }

  // write mesh extent
  outFile << "  <ImageData WholeExtent=\""
	  << xmin << " " << xmax << " " 
	  << ymin << " " << ymax << " " 
	  << zmin << " " << zmax << ""
	  << "\" Origin=\"0 0 0\" Spacing=\"" << dx << " " << dy << " " << dz << "\">" << std::endl;
  outFile << "  <Piece Extent=\"" 
	  << xmin << " " << xmax << " " 
	  << ymin << " " << ymax << " " 
	  << zmin << " " << zmax << ""
	  << "\">" << std::endl;
  
  outFile << "    <PointData>\n";
  outFile << "    </PointData>\n";

  if (outputVtkAscii) {

    outFile << "    <CellData>\n";

    // write data array (ascii), remove ghost cells
    for ( iVar=0; iVar<nbvar; iVar++) {
      outFile << "    <DataArray type=\"";
      if (useDouble)
	outFile << "Float64";
      else
	outFile << "Float32";
      outFile << "\" Name=\"" << variables_names.at(iVar) << "\" format=\"ascii\" >\n";
      
      for (int index=0; index<nbCells; ++index) {
	//index2coord(index,i,j,k,isize,jsize,ksize);
	
	// enforce the use of left layout (Ok for CUDA)
	// but for OpenMP, we will need to transpose
	k = index / ijsize;
	j = (index - k*ijsize) / isize;
	i = index - j*isize - k*ijsize;
	
	if (k>=kmin+ghostWidth and k<=kmax-ghostWidth and
	    j>=jmin+ghostWidth and j<=jmax-ghostWidth and
	    i>=imin+ghostWidth and i<=imax-ghostWidth) {
	  outFile << Uhost(i,j,k,iVar) << " ";
	}
      }
      
      outFile << "\n    </DataArray>\n";

    } // end for iVar
    
    outFile << "    </CellData>\n";
    
    // write footer
    outFile << "  </Piece>\n";
    outFile << "  </ImageData>\n";
    outFile << "</VTKFile>\n";

  } else { // write data in binary format

    outFile << "    <CellData>" << std::endl;

    for (int iVar=0; iVar<nbvar; iVar++) {
      if (useDouble) {
	outFile << "     <DataArray type=\"Float64\" Name=\"" ;
      } else {
	outFile << "     <DataArray type=\"Float32\" Name=\"" ;
      }
      outFile << variables_names.at(iVar)
	      << "\" format=\"appended\" offset=\""
	      << iVar*nx*ny*nz*sizeof(real_t)+iVar*sizeof(unsigned int)
	      <<"\" />" << std::endl;
    }

    outFile << "    </CellData>" << std::endl;
    outFile << "  </Piece>" << std::endl;
    outFile << "  </ImageData>" << std::endl;
    
    outFile << "  <AppendedData encoding=\"raw\">" << std::endl;

    // write the leading undescore
    outFile << "_";
    // then write heavy data (column major format)
    {
      unsigned int nbOfWords = nx*ny*nz*sizeof(real_t);
      for (int iVar=0; iVar<nbvar; iVar++) {
	outFile.write((char *)&nbOfWords,sizeof(unsigned int));
	for (int k=kmin+ghostWidth; k<=kmax-ghostWidth; k++) {
	  for (int j=jmin+ghostWidth; j<=jmax-ghostWidth; j++) {
	    for (int i=imin+ghostWidth; i<=imax-ghostWidth; i++) {
	      real_t tmp = Uhost(i, j, k, iVar);
	      outFile.write((char *)&tmp,sizeof(real_t));
	    } // for i
	  } // for j
	} // for k
      } // for iVar
    }
    
    outFile << "  </AppendedData>" << std::endl;
    outFile << "</VTKFile>" << std::endl;
    
  } // end ascii/binary heavy data write
  
  outFile.close();
  
} // save_VTK_3D_mpi

/*
 * write pvti header in a separate file.
 */
// =======================================================
// =======================================================
void write_pvti_header(std::string headerFilename,
		       std::string outputPrefix,
		       HydroParams& params,
		       int nbvar,
		       const std::map<int, std::string>& varNames,
		       int iStep)
{
  // file handler
  std::fstream outHeader;
  
  // dummy string here, when using the full VTK API, data can be compressed
  // here, no compression used
  std::string compressor("");
  
  // check scalar data type
  bool useDouble = false;
  
  if (sizeof(real_t) == sizeof(double)) {
    useDouble = true;
  }
  
  const int dimType = params.dimType;
  const int nProcs = params.nProcs;
  
  // write iStep in string timeFormat
  std::ostringstream timeFormat;
  timeFormat.width(7);
  timeFormat.fill('0');
  timeFormat << iStep;
  
  // local sub-domain sizes
  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;

  // sizes of MPI Cartesian topology
  const int mx = params.mx;
  const int my = params.my;
  const int mz = params.mz;

  const real_t dx = params.dx;
  const real_t dy = params.dy;
  const real_t dz = (dimType == THREE_D) ? params.dz : params.dx;

  // open pvti header file
  outHeader.open (headerFilename.c_str(), std::ios_base::out);
  
  outHeader << "<?xml version=\"1.0\"?>" << std::endl;
  if (isBigEndian())
    outHeader << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"BigEndian\"" << compressor << ">" << std::endl;
  else
    outHeader << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"LittleEndian\"" << compressor << ">" << std::endl;
  outHeader << "  <PImageData WholeExtent=\"";
  outHeader << 0 << " " << mx*nx << " ";
  outHeader << 0 << " " << my*ny << " ";
  outHeader << 0 << " " << mz*nz << "\" GhostLevel=\"0\" "
	    << "Origin=\"0 0 0\" "
	    << "Spacing=\"" << dx << " " << dy << " " << dz << "\">"
	    << std::endl;
  outHeader << "    <PCellData Scalars=\"Scalars_\">" << std::endl;
  for (int iVar=0; iVar<nbvar; iVar++) {
    if (useDouble) 
      outHeader << "      <PDataArray type=\"Float64\" Name=\""<< varNames.at(iVar)<<"\"/>" << std::endl;
    else
      outHeader << "      <PDataArray type=\"Float32\" Name=\""<< varNames.at(iVar)<<"\"/>" << std::endl;	  
  }
  outHeader << "    </PCellData>" << std::endl;
  
  // one piece per MPI process
  if (dimType == TWO_D) {
    for (int iPiece=0; iPiece<nProcs; ++iPiece) {
      std::ostringstream pieceFormat;
      pieceFormat.width(5);
      pieceFormat.fill('0');
      pieceFormat << iPiece;
      std::string pieceFilename   = outputPrefix+"_time"+timeFormat.str()+"_mpi"+pieceFormat.str()+".vti";
      // get MPI coords corresponding to MPI rank iPiece
      int coords[2];
      params.communicator->getCoords(iPiece,2,coords);
      outHeader << "    <Piece Extent=\"";
      
      // pieces in first line of column are different (due to the special
      // pvti file format with overlapping by 1 cell)
      if (coords[0] == 0)
	outHeader << 0 << " " << nx << " ";
      else
	outHeader << coords[0]*nx << " " << coords[0]*nx+nx << " ";
      if (coords[1] == 0)
	outHeader << 0 << " " << ny << " ";
      else
	outHeader << coords[1]*ny << " " << coords[1]*ny+ny << " ";
      outHeader << 0 << " " << 1 << "\" Source=\"";
      outHeader << pieceFilename << "\"/>" << std::endl;
    } 
  } else { // THREE_D
    for (int iPiece=0; iPiece<nProcs; ++iPiece) {
      std::ostringstream pieceFormat;
      pieceFormat.width(5);
      pieceFormat.fill('0');
      pieceFormat << iPiece;
      std::string pieceFilename   = outputPrefix+"_time"+timeFormat.str()+"_mpi"+pieceFormat.str()+".vti";
      // get MPI coords corresponding to MPI rank iPiece
      int coords[3];
      params.communicator->getCoords(iPiece,3,coords);
      outHeader << " <Piece Extent=\"";
      
      if (coords[0] == 0)
	outHeader << 0 << " " << nx << " ";
      else
	outHeader << coords[0]*nx << " " << coords[0]*nx+nx << " ";
      
      if (coords[1] == 0)
	outHeader << 0 << " " << ny << " ";
      else
	outHeader << coords[1]*ny << " " << coords[1]*ny+ny << " ";
      
      if (coords[2] == 0)
	outHeader << 0 << " " << nz << " ";
      else
	outHeader << coords[2]*nz << " " << coords[2]*nz+nz << " ";
      
      outHeader << "\" Source=\"";
      outHeader << pieceFilename << "\"/>" << std::endl;
    } 
  }
  outHeader << "</PImageData>" << std::endl;
  outHeader << "</VTKFile>" << std::endl;
  
  // close header file
  outHeader.close();
  
  // end writing pvti header
  
} // write_pvti_header
#endif // USE_MPI

} // namespace io

} // namespace euler_kokkos
