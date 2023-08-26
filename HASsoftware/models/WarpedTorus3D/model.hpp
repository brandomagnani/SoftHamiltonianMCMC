//
//  Model.hpp
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

#ifndef Model_hpp
#define Model_hpp

#include <iostream>
#include <blaze/Math.h>
using namespace std;
using namespace blaze;

class Model{

   public:
      double                              r;    // radius of sphere
      DynamicVector<double>               s;    // dimensional radius numbers for the ellipsoid
      DynamicMatrix<double, columnMajor>  c;    // column k = center of sphere k
      DynamicVector<double, columnVector> ssq;  // squares of the "radius" parameters of the ellipsoid
      int d;     // dimension of the ambient space
      int m;     // number of constraint functions
      int n;     // dimension of the hard constraint manifold = d-m
      
      DynamicVector<double, columnVector>            /* return the values ...           */
      xi( DynamicVector<double, columnVector> q);     /* ... of the constraint functions */
      
      DynamicMatrix<double, columnMajor>             /* column k is the gradient ...*/
      gxi( DynamicVector<double, columnVector> q);    /* ... of xi_k(q)               */
   
      // Returns gxi(q) augmented to a square matrix (in case d > m), just appends columns of zeros.
      DynamicMatrix<double, columnMajor>
      Agxi(DynamicMatrix<double, columnMajor> gxi);
   
      // DEFAULT Constructor
      Model();
   
      // PARAMETRIZED Constructor
      Model( int d0, int m0,               /* ambient dimension and constraint number */
             double                r00,    /* radus of sphere   */
             DynamicVector<double> s0,     /* dimensional radius numbers for the ellipsoid  */
             DynamicMatrix<double, columnMajor> c0); // column k = cdnter of sphere k
   
      // COPY Constructor
      Model(const Model& M0);
   
      string ModelName();                  /* Return the name of the model */
      
      double yzIntegrate( double x,        /* Integrate e^{-beta*U(x,y,z)} over y and z */
                          double L,        /* Integrate from y = L to y = R    */
                          double R,        /* Integrate z over the same range  */
                          double eps,      /* The temperature parameter*/
                          int n);          /* the number of integration points in each dir */

};

#endif /* Model.hpp */
