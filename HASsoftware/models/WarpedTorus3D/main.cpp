/*
       Squishy sampler project, the RATTLE Accellerated Sampler approach.
       See main.cpp for more information.
       
       The correctness checks below are adapted from Jonathan Goodman's
       implementation of the Foliation sampler
*/
//
//  3D Warped Torus model
//
//  main.cpp
//
#include <cstdio>
#include <iostream>
#include <fstream>                // write the output to a file
#include <blaze/Math.h>
#include <random>                 // contains the random number generator
#include <cmath>                  // defines exp(..), M_PI
#include <chrono>
#include "HAS.hpp"
#include "model.hpp"
using namespace std;
using namespace blaze;



int main(int argc, char** argv){
   
   const int d = 3;       // dimension of ambient space
   const int m = 2;       // number of constraint functions

   
   //  Set up the specific model problem -- a surface defined by intersection of m spheres
   
   DynamicVector<double, columnVector> ck(d);         // Center of sphere k
   DynamicMatrix<double, columnMajor>  c(d,m);        // Column k is the center of sphere k
   DynamicVector<double, columnVector> s(d);          // dimensional radius numbers for the ellipsoid

   double r;                                          // r = radius of sphere
   
   ck[0] = 0.;                   // first center: c_0 = ( 0, 0, 1)
   ck[1] = 0.;
   ck[2] = 1.;
   column( c, 0UL) = ck;         // 0UL means 0, as an unsigned long int
   
   ck[0] = 0.;                   // second center: c_0 = ( 0,-1, 0, )
   ck[1] = -1.;
   ck[2] =  0.;
   column( c, 1UL) = ck;
   
   r = sqrt(2.);                // The distance between centers is sqrt(2)
   s[0] = sqrt(2.);
   s[1] = sqrt(3.);
   s[2] = sqrt(5.);
   
   // copy the parameters into the instance model
   
   
   Model M( d, m, r, s, c);   // instance of model " 3D Warped Torus "


   
   cout << "--------------------------------------" << endl;
   cout << "\n  Running model: " << M.ModelName() << "\n" << endl;
   
   
//        Inputs for the SoftSampler
   
   
   DynamicVector<double, columnVector> q( d); // starting point for sampler
   
   
   //q[0] = 1.20105824111462;                   // give value to starting point, on 3D Simple Torus model constraint ..
   //q[1] = -0.669497937230318;                 // // .. r1 = sqrt(2), r2 = sqrt(2)
   //q[2] = 0.669497937230317;
   
   q[0] = 0.3780902941558512;                 // give value to starting point, on 3D Warped Torus model constraint ..
   q[1] = -0.7586573478136572;                // .. r = sqrt(2), sx = sqrt(2), sy = sqrt(3), sz = sqrt(5)
   q[2] = 2.132027719657734;
   
   //q[0] =  0.7780902941558512;                 // give value to starting point, OFF the 3D Warped Torus model constraint ..
   //q[1] = -0.7586573478136572;
   //q[2] =  2.132027719657734;



   size_t T      = 1000000;     // number of MCMC steps
   double neps   = 1.e-10;       // convergence tolerance for Newton projection
   double rrc    = 1.e-8;        // closeness criterion for the reverse check
   int itm       = 6;            // maximum number of Newtons iterations
   
   int debug_off = 0;                // if = 1, then prints data on Metropolis ratio for Off move
   int debug_on  = 0;                // if = 1, then prints data on Metropolis ratio for On  move
   
   int integrate = 1;                // if = 1, then it does the integration to find marginal density for x[0]
   
   
// --------------------------------------------KEY PARAMETERS FOR SAMPLER---------------------------------------------------------
   
   
   double beta   = 1000.0;                      // squish parameter, beta = 1 / 2*eps^2
   double eps    = 1.0 / sqrt(2.0*beta);        // squish parameter
   
   int Nsoft = 1;          // number of Soft moves for MCMC step
   int Nrattle = 1;        // number of RATTLE integrator time steps for each MCMC step
   
   double ks  = 0.7;       // factor for Soft proposal size
   double ss  = ks*eps;    // scale for Soft proposal
   
   double dt  = 1.0;       // time step size in RATTLE integrator
   
// -------------------------------------------------------------------------------------------------------------------------------
   
   size_t size_factor = Nsoft + Nrattle;  // multiplies T below
   size_t T_chain = size_factor * T;      // length of chain
   vector<double> chain(d*T_chain);       // chain[k+d*l] is the value of q[k] where q is l-th sample
   unsigned seed = 17U;     // seed for the random number generator -- 17 is the most random number?
   mt19937 RG(seed);        // Mersenne twister random number generator
   SamplerStats stats;
   
   
   auto start = chrono::steady_clock::now();
   HASampler(chain, &stats, T, eps, dt, Nsoft, Nrattle, q, M, ss, neps, rrc, itm, RG);
   auto end = chrono::steady_clock::now();
   
   
   double Ts = stats.SoftSample;    // number of Soft samples
   double Tr = stats.HardSample;    // number of Rattle samples
   int    Ns = Ts;                  // number of good samples = number of soft samples
   double As = 0.;                  // Pr of Acceptance of Soft sample
   double Ar = 0.;                  // Pr of Acceptance of Rattle sample
   
   if (stats.SoftSample > 0) {  // Set Pr of Acceptance of Soft sample
      As    = (double) stats.SoftSampleAccepted / (double) stats.SoftSample;
   }
   if (stats.HardSample > 0) {  // Set Pr of Acceptance of Rattle sample
      Ar    = (double) stats.HardSampleAccepted / (double) stats.HardSample;
   }

   
   

   cout << " beta = " << beta << endl;
   cout << " eps = " << eps << endl;
   cout << " Elapsed time : " << chrono::duration_cast<chrono::seconds>(end - start).count() << " sec" << endl;
   cout << " " << endl;
   cout << " RATTLE move POSITION projection failures    : " << stats.HardRejectionFailedProjection_qn << endl;
   cout << " Rattle move POSITION reverse check failures : " << stats.HardRejectionReverseCheck_q << endl;
   cout << " Rattle move MOMENTUM reverse check failures : " << stats.HardRejectionReverseCheck_p << endl;
   cout << " Rattle move reverse check failures : " << stats.HardRejectionReverseCheck << endl;
   cout << " "              << endl;
   cout << " RATTLE Metropolis rejection        : " << stats.HardRejectionMetropolis << endl;
   cout << " "              << endl;
   cout << " Number of Soft   samples = " << Ts    << endl;
   cout << " Number of Rattle samples = " << Tr    << endl;
   cout << " " << endl;
   cout << " Soft   sample Acceptance Pr = " << As   << endl;
   //cout << " Rattle sample Acceptance Pr = " << Ar   << endl;

   cout << " " << endl;
   cout << " T  = " << T_chain  << endl;
   cout << " " << endl;
   cout << " Ns = " << Ns << endl;
   cout << " " << endl;
   cout << "--------------------------------------" << endl;

   
//  setup for data analysis:
   
   int bin;
   char OutputString[200];
   int  StringLength;
   
//   NO ANGLE CHECK
   
   
//   Histogram of the x coordinates q[0]=x, q[1]=y, q[2]=z
   
   int nx   = 100;             // number of x1 values for the PDF and number of x1 bins
   vector<double> Ratio(nx);   // contains the random variables Ri_N

   if ( integrate == 1 ) {
      int ni   = 500;     // number of integration points in each direction
      double L = -3.0;
      double R =  3.0;
      double x1    = .5;
      vector<double> fl(nx);  // approximate (un-normalized) true pdf for x[0]=x, compute by integrating out x,z variables
      vector<int>    Nxb(nx); // vector counting number of samples in each bin
      for ( bin = 0; bin < nx; bin++) Nxb[bin] = 0;
      double dx = ( R - L )/( (double) nx);
      for ( int i=0; i<nx; i++){
         x1 = L + dx*i + .5*dx;
         fl[i]= M.yzIntegrate( x1, L, R, eps, ni);
      }
               
      int outliers = 0;       //  number of samples outside the histogram range
      for ( unsigned int iter = 0; iter < Ns; iter++){
         x1 = chain[ d*iter ];  // same as before, but with k=0 as we need q[0] of iter-th good sample
         bin = (int) ( ( x1 - L )/ dx);
         if ( ( bin >= 0 ) && ( bin < nx ) ){
            Nxb[bin]++;
         }
         else{
            outliers++;
         }
      }        // end of analysis loop
     
      double Z;
      cout << " " << endl;
      cout << "   bin    center      count      pdf          1/Z" << endl;
      for ( bin = 0; bin < nx; bin++){
         if ( Nxb[bin] > 0 ) {
            x1 = L + dx*bin + .5*dx;
            Z = ( (double) Nxb[bin]) / ( (double) Ns*dx*fl[bin]);
            Ratio[bin] = Z;
            StringLength = snprintf( OutputString, sizeof(OutputString)," %4d   %8.3f   %8d   %9.3e    %9.3e", bin, x1, Nxb[bin], fl[bin], Z);
            cout << OutputString << endl;
         }
      }
      cout << " " << endl;
      cout << " Number of outliers : " << outliers << endl;
      cout << " " << endl;
   }
   

   ofstream OutputFile ( "ChainOutput.py");
   OutputFile << "# data output file from an MCMC code\n" << endl;
   OutputFile << "import numpy as np" << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"eps = %10.5e", eps);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"ks = %10.5e", ks);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"ss = %10.5e", ss);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"Nsoft = %10d", Nsoft);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"Nrattle = %10d", Nrattle);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"T = %10d", (int)T_chain);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"Ts = %10d", (int)Ts);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"Tr = %10d", (int)Ts);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"As = %6.3f", As);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"Ar = %6.3f", Ar);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"Ns = %10d", Ns);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"d = %10d", d);
   OutputFile << OutputString << endl;
   OutputFile << "ModelName = \"" << M.ModelName() << "\"" << endl;
   OutputFile.close();
   


// Write chain in binary format (in a file called "chain.bin"), much faster for Python to read when chain is very long
   size_t size1 = Ns*d;
   ofstream ostream1("chain.bin", ios::out | ios::binary);
   ostream1.write((char*)&chain[0], size1 * sizeof(double));
   ostream1.close();


   
}  // end of main
