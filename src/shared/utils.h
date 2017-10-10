#ifndef EULER_KOKKOS_UTILS_H
#define EULER_KOKKOS_UTILS_H

#include <math.h>
#include <iosfwd>

// make the compiler ignore an unused variable
#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

// make the compiler ignore an unused function
#ifdef __GNUC__
#define UNUSED_FUNCTION __attribute__ ((unused))
#else
#define UNUSED_FUNCTION
#endif

#define THRESHOLD 1e-12

#define ISFUZZYNULL(a) (std::abs(a) < THRESHOLD)
#define FUZZYCOMPARE(a, b) \
    ((ISFUZZYNULL(a) && ISFUZZYNULL(b)) || \
     (std::abs((a) - (b)) * 1000000000000. <= std::fmin(std::abs(a), std::abs(b))))
#define FUZZYLIMITS(x, a, b) \
    (((x) > ((a) - THRESHOLD)) && ((x) < ((b) + THRESHOLD)))

void        print_current_date(std::ostream& stream);
std::string get_current_date();

#endif // EULER_KOKKOS_UTILS_H
