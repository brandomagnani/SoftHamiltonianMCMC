//   model.cpp
//
//   model name: 3D simple torus

//   Adapted from Jonathan Goodman's foliation sampler code

/*   This model has a surface in d dimensions defined by m constraints
     of the form
          
            | q - c_k | = r_k
            
     Geomertically, the surface is the intersection of m spheres
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
       DynamicVector<double>              r0,    /* radii of spheres, distance constraints */
       DynamicMatrix<double, columnMajor> c0){   /* centers of spheres                     */
   d = d0;
   m = m0;
   r = r0;
   c = c0;
}

// COPY Constructor
Model::Model(const Model& M0){
   d = M0.d;
   m = M0.m;
   r = M0.r;
   c = M0.c;
}

DynamicVector<double, columnVector>
Model::xi(DynamicVector<double, columnVector> q){
   DynamicVector<double, columnVector> disp(d);   // vector from q to c_j (center of sphere j)
   DynamicVector<double, columnVector> xi(m);      // constraint function values
   for (int j = 0; j < m; j++){
      
      disp = q - column(c,j);
      xi[j] = trans(disp)*disp - r[j]*r[j];
    }
   return xi;
}

DynamicMatrix<double, columnMajor>
Model::gxi(DynamicVector<double, columnVector> q){
   DynamicVector<double, columnVector>disp(d);   // displacement from a center
   DynamicMatrix<double, columnMajor> gxi(d,m);   // gradient, (dxm) matrix
   for (int j = 0; j < m; j++){
      disp = q - column(c,j);
      column(gxi, j)  = 2.*disp;
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
   
   return(" 3D Simple Torus ");
}

//  Compute the (un-normalized) probability density for x1 by integrating over
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
