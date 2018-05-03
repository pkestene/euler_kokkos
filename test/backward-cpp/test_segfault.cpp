/**
 * A minimalist example to use library backward-cpp to pretty print 
 * error / stack trace...
 *
 * for as standalone build:
 * g++ -DBACKWARD_HAS_DW=1 -g -I../../external/backward-cpp test_segfault.cpp ../../external/backward-cpp/backward.cpp -ldw
 *
 */

#include <iostream>

#include "backward.hpp"

void badass_function() {
        char* ptr = (char*)42;
        *ptr = 42;
}

int main(int argc, char* argv[])
{

  std::cout << "Create a segfault on purpose...\n";
  badass_function();

  return 0;
}
