// Example that shows simple usage of the INIReader class

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream> // string stream

#include "config/ConfigMap.h"

// =====================================================================
// =====================================================================
// =====================================================================
int test1() {
  
  std::cout << "===============================================\n";
  std::cout << "Test1\n";
  
  // make test.ini file
  std::fstream iniFile;
  iniFile.open ("./test.ini", std::ios_base::out);
  iniFile << "; Test config file for ini_test.c" << std::endl;
  
  iniFile << "[Protocol]             ; Protocol configuration" << std::endl;
  iniFile << "Version=6              ; IPv6" << std::endl;
  
  iniFile << "[User]" << std::endl;
  iniFile << "Name = Bob Smith       ; Spaces around '=' are stripped" << std::endl;
  iniFile << "Email = bob@smith.com  ; And comments (like this) ignored" << std::endl;
  iniFile.close();

  // create a ConfigMap instance
  ConfigMap configMap("./test.ini");
  
  if (configMap.ParseError() < 0) {
    std::cout << "Can't load 'test.ini'\n";
    return -1;
  }
  std::cout << "Config loaded from 'test.ini': version="
	    << configMap.getInteger("protocol", "version", -1) << ", name="
	    << configMap.getString("user", "name", "UNKNOWN") << ", email="
	    << configMap.getString("user", "email", "UNKNOWN") << "\n";
  
  ConfigMap configMap2 = configMap;
  std::cout << std::endl;
  std::cout << "Config copied from configMap: version="
	    << configMap.getInteger("protocol", "version", -1) << ", name="
	    << configMap.getString("user", "name", "UNKNOWN") << ", email="
	    << configMap.getString("user", "email", "UNKNOWN") << "\n";

  return 0;
  
} // test1

// =====================================================================
// =====================================================================
// =====================================================================
int test2() {

  std::cout << "===============================================\n";
  std::cout << "Test2\n";

  // make test.ini data in a string stream
  std::stringstream iniFile;
  iniFile << "; Test config file for ini_test.c" << std::endl;
  
  iniFile << "[Protocol]             ; Protocol configuration" << std::endl;
  iniFile << "Version=6              ; IPv6" << std::endl;
  
  iniFile << "[User]" << std::endl;
  iniFile << "Name = Bob Smith       ; Spaces around '=' are stripped" << std::endl;
  iniFile << "Email = bob@smith.com  ; And comments (like this) ignored" << std::endl;

  std::string s = iniFile.str();

  std::cout << "s string content\n";
  std::cout << s << std::endl;
  std::cout << "s size : " << s.length() << std::endl;

  int buffer_size = s.length() + 1;
  char* buffer = new char[s.length()+1];
  strcpy(buffer,s.c_str());

  std::cout << "buffer size : " << buffer_size << std::endl;
  
  // create a ConfigMap instance
  ConfigMap configMap(buffer,buffer_size);

  delete [] buffer;
  
  if (configMap.ParseError() < 0) {
    std::cout << "Can't load buffer\n";
    return -1;
  }
  std::cout << "ConfigMap loaded from buffer: version="
	    << configMap.getInteger("protocol", "version", -1) << ", name="
	    << configMap.getString("user", "name", "UNKNOWN") << ", email="
	    << configMap.getString("user", "email", "UNKNOWN") << "\n";
  
  ConfigMap configMap2 = configMap;
  std::cout << std::endl;
  std::cout << "configMap2 copied from configMap: version="
	    << configMap.getInteger("protocol", "version", -1) << ", name="
	    << configMap.getString("user", "name", "UNKNOWN") << ", email="
	    << configMap.getString("user", "email", "UNKNOWN") << "\n";

  return 0;
  
} // test2

// =====================================================================
// =====================================================================
// =====================================================================
int main(int argc, char* argv[])
{

  int status = 0;
  
  status = test1();
  std::cout << "\n\n";
  status = test2();

  return status;
}
