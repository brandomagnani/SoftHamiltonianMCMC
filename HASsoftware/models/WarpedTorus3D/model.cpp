//
//  Model.cpp
//
//  model name: 3D warped torus
//
//  Adapted from Jonathan Goodman's foliation sampler code.
//
/*   This model defines a warped circle in 3D as the intersection
     of two surfaces.  One is a round sphere defined by
          
            | q - c_0 | = r_0
    
     The other surface is an ellipsoid, which is a distorted sphere,
     defined by
     
          sum_k ( q_k - c_{1,k} )^2/s_k^2 = 1
     
*/
#include <iostream>
#include <blaze/Math.h>
#include "model.hpp"
using namespace std;
using namespace blaze;

// DEFAULT Constructor
Model::Model(){}

// PARAMETRIZED Constructor, copy given dimensions into instance variables
Model::Model( int                         d0,    /* dimension of the ambient space */
              int                         m0,    /* number of constraints          */
              double                      r00,   /* radus of sphere                              */
       DynamicVector<double>              s0,    /* dimensional radius numbers for the ellipsoid */
       DynamicMatrix<double, columnMajor> c0){   /* column 0 = cdnter of round sphere            */
                                                 /* column 1 = center of ellipsoid               */
   // building the model object
   d = d0;
   m = m0;
   r = r00;
   s = s0;
   c = c0;
   
   ssq.resize(d); // contains the square of entries of s0, the "radius" parameters of the ellipsoid
   for ( int i = 0; i < d; i++) {
      ssq[i] = s0[i]*s0[i];
   }
}

// COPY Constructor
Model::Model(const Model& M0){
   d   = M0.d;
   m   = M0.m;
   r   = M0.r;
   s   = M0.s;
   c   = M0.c;
   ssq = M0.ssq;
}

DynamicVector<double, columnVector>
Model::xi(DynamicVector<double, columnVector> q){
   
   DynamicVector<double, columnVector> disp(d);   // vector from q to c_j (center of sphere j)
   DynamicVector<double, columnVector> xi(m);      // constraint function values
   
   //        The sphere
     
   disp = q - column(c,0);
   xi[0] = trans(disp)*disp - r*r;
   
   //       The ellipsoid

   double exi = 0.;              // xi[1] = exi = xi "for the ellipsoid"
   disp = q - column(c,1);
   for ( int j =0; j < d; j++){
      exi += disp[j]*disp[j]/ssq[j];
   }
   xi[1] = exi - 1.;
   
   return xi;
}

DynamicMatrix<double, columnMajor>
Model::gxi(DynamicVector<double, columnVector> q){
   
   DynamicVector<double, columnVector>disp(d);   // displacement from a center
   DynamicMatrix<double, columnMajor> gxi(d,m);  // gradient, (dxm) matrix
   
   //        The sphere, first column
     
   disp = q - column(c,0);
   column(gxi, 0)  = 2.*disp;
   
   //       The ellipsoid, second column

   disp = q - column(c,1);
   for ( int j =0; j < d; j++){
      gxi(j,1) = 2.0*disp[j]/ssq[j];
   }
   
   return gxi;
}

// Returns gxi(q) augmented to a square matrix (in case d > m), last n=d-m columns are just zeros.
// Needed to have a complete QR decomposition of gxi(q), with Q (dxd) matrix
DynamicMatrix<double, columnMajor>
Model::Agxi(DynamicMatrix<double, columnMajor> gxi){
   
   DynamicMatrix<double, columnMajor> Agxi(d,d);   // Augmented gradient, (dxd) matrix
   Agxi = 0.;  // initialize to zero
   
   for (int j = 0; j < m; j++){
      column(Agxi, j)  = column(gxi, j);
   }

   return Agxi;  // last n=d-m columns are zeros
}

string Model::ModelName(){                  /* Return the name of the model */
   
   return(" 3D Warped Torus ");
}

//  Compute the (un-normalized) probability density for q1 by integrating over
//  the other two variables.
 
double Model::yzIntegrate( double x, double L, double R, double eps, int n){

//  Use the rectangle rule in the y and z directions with n midpoints in each direction

   double dy  = (R-L)/( (double) n);
   double dz  = dy;
   double y, z;
   double sum = 0.;
   
   DynamicVector<double, columnVector> qv( 3);    // point in 3D
   DynamicVector<double, columnVector> xiv( 2);    // values of the constraint functions
   
   qv[0] = x;
   for ( int j = 0; j < n; j++){
      y = L + j*dy + .5*dy;
      qv[1] = y;
      for ( int k = 0; k < n; k++) {
         z = L + k*dz + .5*dz;
         qv[2] = z;
         xiv = xi(qv);
         sum += exp( - 0.5*(trans(xiv)*xiv)/(eps*eps) );
      }
   }
   return dy*dz*sum;
}
