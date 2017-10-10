/**
 * \file OpenMPTimer.h
 * \brief A simple timer class.
 *
 * \author Pierre Kestener
 * \date 29 Oct 2010
 *
 */
#ifndef OPENMP_TIMER_H_
#define OPENMP_TIMER_H_

#include <omp.h>

/**
 * \brief a simple Timer class.
 * If MPI is enabled, should we use MPI_WTime instead of gettimeofday (?!?)
 */
class OpenMPTimer
{
public:
  /** default constructor, timing starts rightaway */
  OpenMPTimer();
    
  OpenMPTimer(double t);
  OpenMPTimer(OpenMPTimer const& aTimer);
  virtual ~OpenMPTimer();

  /** start time measure */
  virtual void start();
    
  /** stop time measure and add result to total_time */
  virtual void stop();

  /** return elapsed time in seconds (as stored in total_time) */
  virtual double elapsed() const;

protected:
  double    start_time;

  /** store total accumulated timings */
  double    total_time;

}; // class OpenMPTimer


#endif // OPENMP_TIMER_H_
