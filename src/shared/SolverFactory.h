#ifndef SOLVER_FACTORY_H_
#define SOLVER_FACTORY_H_

#include <string>
#include <map>
#include <cstdlib>

#include "SolverBase.h"
struct HydroParams;
class ConfigMap;

namespace euler_kokkos {

/**
 * An abstract base class to define a common interface for concrete solvers.
 *
 * The main purpose is to return a concrete Solver object.
 * The idea here it to define a map between a name and the actual solver.
 *
 * Each derived class will have to define from this class.
 *
 */
class SolverFactory {

private:
  // make constructor private -- this class is singleton
  SolverFactory();
  SolverFactory(const SolverFactory&) = delete; // non construction-copyable
  SolverFactory& operator=(const SolverFactory &) {return *this;} // non-copyable

  /**
   * typedef to the solver creation function pointer.
   * This function pointer will actually be populated with a concrete solver
   * method named "create" which takes in input a HydroParams pointer
   * (necessary to call the concrete solver constructor).
   */
  using SolverCreateFn = SolverBase* (*)(HydroParams& params,
					 ConfigMap& configMap);

  /**
   * Map to associate a label with a pair of solver creation function, and
   * UserDataManager creation function.
   * Each concrete solver / UserDataManger class must provide a (static)
   * creation method named create.
   */
  using SolverCreateMap = std::map<std::string, SolverCreateFn>;
  SolverCreateMap m_solverCreateMap;

public:
  ~SolverFactory() { m_solverCreateMap.clear(); }

  static SolverFactory& Instance()
  {
    static SolverFactory instance;
    return instance;
  }

  /**
   * Routine to insert an solver function into the map.
   * Note that this register function can be used to serve
   * at least two different purposes:
   * - in the concrete factory: register existing callback's
   * - in some client code, register a callback from a plugin code, at runtime.
   */
  void registerSolver(const std::string& key, SolverCreateFn cfn) {
    m_solverCreateMap[key] = cfn;
  };

  /**
   * \brief Retrieve one of the possible solvers by name.
   *
   * Allowed default names are defined in the concrete factory.
   */
  SolverBase* create (const std::string &solver_name,
		      HydroParams& params,
		      ConfigMap& configMap) {

    // find the solver name in the register map
    SolverCreateMap::iterator it = m_solverCreateMap.find(solver_name);

    // if found, just create and return the Solver object
    if ( it != m_solverCreateMap.end() ) {

      // create solver
      SolverBase *solver = it->second(params, configMap);

      // additionnal initialization (each solver might override this method)
      solver->init_io();

      return solver;
    }

    // if not found, return null pointer
    // it is the responsability of the client code to deal with
    // the possibility to have a nullptr callback (does nothing).
    printf("############ WARNING: ############\n");
    printf("%s: is not recognized as a valid application name key.\n",solver_name.c_str());
    printf("Valid solver names are:\n");
    for (auto it=m_solverCreateMap.begin(); it!=m_solverCreateMap.end(); ++it)
      printf("%s\n",it->first.c_str());
    printf("############ WARNING: ############\n");

    printf("Solver application name not found\n");
    std::abort();

    return nullptr;
  }; // create

}; // class SolverFactory

} // namespace euler_kokkos

#endif // SOLVER_FACTORY_H_
