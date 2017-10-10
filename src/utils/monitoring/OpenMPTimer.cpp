/**
 * \file OpenMPTimer.cpp
 * \brief a simpe Timer class implementation.
 * 
 * \author Pierre Kestener
 * \date 29 Oct 2010
 *
 */

#include "OpenMPTimer.h"

#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////
// OpenMPTimer class methods body
////////////////////////////////////////////////////////////////////////////////

// =======================================================
// =======================================================
OpenMPTimer::OpenMPTimer() {
  start_time = 0.0;
  total_time = 0.0;
  start();
} // OpenMPTimer::OpenMPTimer

// =======================================================
// =======================================================
OpenMPTimer::OpenMPTimer(double t) 
{
    
  start_time = 0;
  total_time = t;
    
} // OpenMPTimer::OpenMPTimer

  // =======================================================
  // =======================================================
OpenMPTimer::OpenMPTimer(OpenMPTimer const& aTimer) : start_time(aTimer.start_time), total_time(aTimer.total_time)
{
} // OpenMPTimer::OpenMPTimer

  // =======================================================
  // =======================================================
OpenMPTimer::~OpenMPTimer()
{
} // OpenMPTimer::~OpenMPTimer

  // =======================================================
  // =======================================================
void OpenMPTimer::start() 
{

  start_time = omp_get_wtime();
  
} // OpenMPTimer::start
  
  // =======================================================
  // =======================================================
void OpenMPTimer::stop()
{
  double now = omp_get_wtime();
  
  total_time += (now-start_time);

} // OpenMPTimer::stop

  // =======================================================
  // =======================================================
double OpenMPTimer::elapsed() const
{

  return total_time;

} // OpenMPTimer::elapsed
