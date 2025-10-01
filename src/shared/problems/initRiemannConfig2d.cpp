/**
 * \file initRiemannConfig2d.cpp
 */

#include "initRiemannConfig2d.h"
namespace euler_kokkos
{

// =====================================================================
// =====================================================================
KOKKOS_FUNCTION
RiemannConfig<2>::HydroStates_t
getRiemannConfig2d(int numConfig)
{
  RiemannConfig<2>::HydroStates_t  Us;
  RiemannConfig<2>::HydroState_t & U0 = Us[0];
  RiemannConfig<2>::HydroState_t & U1 = Us[1];
  RiemannConfig<2>::HydroState_t & U2 = Us[2];
  RiemannConfig<2>::HydroState_t & U3 = Us[3];

  switch (numConfig)
  {

    case 0:
      // Config 1
      U0[ID] = 1.0;
      U0[IU] = 0.0;
      U0[IV] = 0.0;
      U0[IP] = 1.0;

      U1[ID] = 0.5197;
      U1[IU] = -0.7259;
      U1[IV] = 0.0;
      U1[IP] = 0.4;

      U2[ID] = 0.1072;
      U2[IU] = -0.7259;
      U2[IV] = -1.4045;
      U2[IP] = 0.0439;

      U3[ID] = 0.2579;
      U3[IU] = 0.0;
      U3[IV] = -1.4045;
      U3[IP] = 0.15;

      break;
    case 1:
      // Config 2
      U0[ID] = 1.0;
      U0[IU] = 0.0;
      U0[IV] = 0.0;
      U0[IP] = 1.0;

      U1[ID] = 0.5197;
      U1[IU] = -0.7259;
      U1[IV] = 0.0;
      U1[IP] = 0.4;

      U2[ID] = 1.0;
      U2[IU] = -0.7259;
      U2[IV] = -0.7259;
      U2[IP] = 1.0;

      U3[ID] = 0.5197;
      U3[IU] = 0.0;
      U3[IV] = -0.7259;
      U3[IP] = 0.4;

      break;
    case 2:
      // Config 3
      U0[ID] = 1.5;
      U0[IU] = 0.0;
      U0[IV] = 0.0;
      U0[IP] = 1.5;

      U1[ID] = 0.5323;
      U1[IU] = 1.206;
      U1[IV] = 0.0;
      U1[IP] = 0.3;

      U2[ID] = 0.138;
      U2[IU] = 1.206;
      U2[IV] = 1.206;
      U2[IP] = 0.029;

      U3[ID] = 0.5323;
      U3[IU] = 0.0;
      U3[IV] = 1.206;
      U3[IP] = 0.3;

      break;
    case 3:
      // Config 4
      U0[ID] = 1.1;
      U0[IU] = 0.0;
      U0[IV] = 0.0;
      U0[IP] = 1.1;

      U1[ID] = 0.5065;
      U1[IU] = 0.8939;
      U1[IV] = 0.0;
      U1[IP] = 0.35;

      U2[ID] = 1.1;
      U2[IU] = 0.8939;
      U2[IV] = 0.8939;
      U2[IP] = 1.1;

      U3[ID] = 0.5065;
      U3[IU] = 0.0;
      U3[IV] = 0.8939;
      U3[IP] = 0.35;

      break;
    case 4:
      // Config 5
      U0[ID] = 1.0;
      U0[IU] = -0.75;
      U0[IV] = -0.5;
      U0[IP] = 1.0;

      U1[ID] = 2.0;
      U1[IU] = -0.75;
      U1[IV] = 0.5;
      U1[IP] = 1.0;

      U2[ID] = 1.0;
      U2[IU] = 0.75;
      U2[IV] = 0.5;
      U2[IP] = 1.0;

      U3[ID] = 3.0;
      U3[IU] = 0.75;
      U3[IV] = -0.5;
      U3[IP] = 1.0;

      break;
    case 5:
      // Config 6
      U0[ID] = 1.0;
      U0[IU] = 0.75;
      U0[IV] = -0.5;
      U0[IP] = 1.0;

      U1[ID] = 2.0;
      U1[IU] = 0.75;
      U1[IV] = 0.5;
      U1[IP] = 0.5;

      U2[ID] = 1.0;
      U2[IU] = -0.75;
      U2[IV] = 0.5;
      U2[IP] = 1.0;

      U3[ID] = 3.0;
      U3[IU] = -0.75;
      U3[IV] = -0.5;
      U3[IP] = 1.0;

      break;
    case 6:
      // Config 7
      U0[ID] = 1.0;
      U0[IU] = 0.1;
      U0[IV] = 0.1;
      U0[IP] = 1.0;

      U1[ID] = 0.5197;
      U1[IU] = -0.6259;
      U1[IV] = 0.1;
      U1[IP] = 0.4;

      U2[ID] = 0.8;
      U2[IU] = 0.1;
      U2[IV] = 0.1;
      U2[IP] = 0.4;

      U3[ID] = 0.5197;
      U3[IU] = 0.1;
      U3[IV] = -0.6259;
      U3[IP] = 0.4;

      break;
    case 7:
      // Config 8
      U0[ID] = 0.5197;
      U0[IU] = 0.1;
      U0[IV] = 0.1;
      U0[IP] = 0.4;

      U1[ID] = 1.0;
      U1[IU] = -0.6259;
      U1[IV] = 0.1;
      U1[IP] = 1.0;

      U2[ID] = 0.8;
      U2[IU] = 0.1;
      U2[IV] = 0.1;
      U2[IP] = 1.0;

      U3[ID] = 1.0;
      U3[IU] = 0.1;
      U3[IV] = -0.6259;
      U3[IP] = 1.0;

      break;
    case 8:
      // Config 9
      U0[ID] = 1.0;
      U0[IU] = 0.0;
      U0[IV] = 0.3;
      U0[IP] = 1.0;

      U1[ID] = 2.0;
      U1[IU] = 0.0;
      U1[IV] = -0.3;
      U1[IP] = 1.0;

      U2[ID] = 1.039;
      U2[IU] = 0.0;
      U2[IV] = -0.8133;
      U2[IP] = 0.4;

      U3[ID] = 0.5197;
      U3[IU] = 0.0;
      U3[IV] = -0.4259;
      U3[IP] = 0.4;

      break;
    case 9:
      // Config 10
      U0[ID] = 1.0;
      U0[IU] = 0.0;
      U0[IV] = 0.4297;
      U0[IP] = 1.0;

      U1[ID] = 0.5;
      U1[IU] = 0.0;
      U1[IV] = 0.6076;
      U1[IP] = 1.0;

      U2[ID] = 0.2281;
      U2[IU] = 0.0;
      U2[IV] = -0.6076;
      U2[IP] = 0.3333;

      U3[ID] = 0.4562;
      U3[IU] = 0.0;
      U3[IV] = -0.4259;
      U3[IP] = 0.3333;

      break;
    case 10:
      // Config 11
      U0[ID] = 1.0;
      U0[IU] = 0.1;
      U0[IV] = 0.0;
      U0[IP] = 1.0;

      U1[ID] = 0.5313;
      U1[IU] = 0.8276;
      U1[IV] = 0.0;
      U1[IP] = 0.4;

      U2[ID] = 0.8;
      U2[IU] = 0.1;
      U2[IV] = 0.0;
      U2[IP] = 0.4;

      U3[ID] = 0.5313;
      U3[IU] = 0.1;
      U3[IV] = 0.7276;
      U3[IP] = 0.4;

      break;
    case 11:
      // Config 12
      U0[ID] = 0.5313;
      U0[IU] = 0.0;
      U0[IV] = 0.0;
      U0[IP] = 0.4;

      U1[ID] = 1.0;
      U1[IU] = 0.7276;
      U1[IV] = 0.0;
      U1[IP] = 1.0;

      U2[ID] = 0.8;
      U2[IU] = 0.0;
      U2[IV] = 0.0;
      U2[IP] = 1.0;

      U3[ID] = 1.0;
      U3[IU] = 0.0;
      U3[IV] = 0.7276;
      U3[IP] = 1.0;

      break;
    case 12:
      // Config 13
      U0[ID] = 1.0;
      U0[IU] = 0.0;
      U0[IV] = -0.3;
      U0[IP] = 1.0;

      U1[ID] = 2.0;
      U1[IU] = 0.0;
      U1[IV] = 0.3;
      U1[IP] = 1.0;

      U2[ID] = 1.0625;
      U2[IU] = 0.0;
      U2[IV] = 0.8145;
      U2[IP] = 0.4;

      U3[ID] = 0.5313;
      U3[IU] = 0.0;
      U3[IV] = 0.4276;
      U3[IP] = 0.4;

      break;
    case 13:
      // Config 14
      U0[ID] = 2.0;
      U0[IU] = 0.0;
      U0[IV] = -0.5606;
      U0[IP] = 8.0;

      U1[ID] = 1.0;
      U1[IU] = 0.0;
      U1[IV] = -1.2172;
      U1[IP] = 8.0;

      U2[ID] = 0.4736;
      U2[IU] = 0.0;
      U2[IV] = 1.2172;
      U2[IP] = 2.6667;

      U3[ID] = 0.9474;
      U3[IU] = 0.0;
      U3[IV] = 1.1606;
      U3[IP] = 2.6667;

      break;
    case 14:
      // Config 15
      U0[ID] = 1.0;
      U0[IU] = 0.1;
      U0[IV] = -0.3;
      U0[IP] = 1.0;

      U1[ID] = 0.5197;
      U1[IU] = -0.6259;
      U1[IV] = -0.3;
      U1[IP] = 0.4;

      U2[ID] = 0.8;
      U2[IU] = 0.1;
      U2[IV] = -0.3;
      U2[IP] = 0.4;

      U3[ID] = 0.5313;
      U3[IU] = 0.1;
      U3[IV] = 0.4276;
      U3[IP] = 0.4;

      break;
    case 15:
      // Config 16
      U0[ID] = 0.5313;
      U0[IU] = 0.1;
      U0[IV] = 0.1;
      U0[IP] = 0.4;

      U1[ID] = 1.0222;
      U1[IU] = -0.6179;
      U1[IV] = 0.1;
      U1[IP] = 1.0;

      U2[ID] = 0.8;
      U2[IU] = 0.1;
      U2[IV] = 0.1;
      U2[IP] = 1.0;

      U3[ID] = 1.0;
      U3[IU] = 0.1;
      U3[IV] = 0.8276;
      U3[IP] = 1.0;

      break;
    case 16:
      // Config 17
      U0[ID] = 1.0;
      U0[IU] = 0.0;
      U0[IV] = -0.4;
      U0[IP] = 1.0;

      U1[ID] = 2.0;
      U1[IU] = 0.0;
      U1[IV] = -0.3;
      U1[IP] = 1.0;

      U2[ID] = 1.0625;
      U2[IU] = 0.0;
      U2[IV] = 0.2145;
      U2[IP] = 0.4;

      U3[ID] = 0.5197;
      U3[IU] = 0.0;
      U3[IV] = -1.1259;
      U3[IP] = 0.4;

      break;
    case 17:
      // Config 18
      U0[ID] = 1.0;
      U0[IU] = 0.0;
      U0[IV] = 1.0;
      U0[IP] = 1.0;

      U1[ID] = 2.0;
      U1[IU] = 0.0;
      U1[IV] = -0.3;
      U1[IP] = 1.0;

      U2[ID] = 1.0625;
      U2[IU] = 0.0;
      U2[IV] = 0.2145;
      U2[IP] = 0.4;

      U3[ID] = 0.5197;
      U3[IU] = 0.0;
      U3[IV] = 0.2741;
      U3[IP] = 0.4;

      break;
    case 18:
      // Config 19
      U0[ID] = 1.0;
      U0[IU] = 0.0;
      U0[IV] = 0.3;
      U0[IP] = 1.0;

      U1[ID] = 2.0;
      U1[IU] = 0.0;
      U1[IV] = -0.3;
      U1[IP] = 1.0;

      U2[ID] = 1.0625;
      U2[IU] = 0.0;
      U2[IV] = 0.2145;
      U2[IP] = 0.4;

      U3[ID] = 0.5197;
      U3[IU] = 0.0;
      U3[IV] = -0.4259;
      U3[IP] = 0.4;

    default:
      // Config 1
      U0[ID] = 1.0;
      U0[IU] = 0.0;
      U0[IV] = 0.0;
      U0[IP] = 1.0;

      U1[ID] = 0.5197;
      U1[IU] = -0.7259;
      U1[IV] = 0.0;
      U1[IP] = 0.4;

      U2[ID] = 0.1072;
      U2[IU] = -0.7259;
      U2[IV] = -1.4045;
      U2[IP] = 0.0439;

      U3[ID] = 0.2579;
      U3[IU] = 0.0;
      U3[IV] = -1.4045;
      U3[IP] = 0.15;
  }

  return Us;
};

} // namespace euler_kokkos
