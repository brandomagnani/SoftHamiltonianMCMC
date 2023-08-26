//
//  Model.hpp
//
//  model name: 3D simple torus
//
//  Adapted from Jonathan Goodman's foliation sampler code
//
/*   This model has a surface in d dimensions defined by m constraints
     of the form
          
            | q - c_k | = r_k
            
     Geomertically, the surface is the intersection of m spheres
*/

#ifndef Model_hpp
#define Model_hpp

#include <iostream>
#include <blaze/Math.h>
using namespace std;
using namespace blaze;

class Model{

   public:
      DynamicVector<double> r;              // radii of constraint spheres
      DynamicMatrix<double, columnMajor> c; // column k = center of sphere k
      int d;     // dimension of the ambient space
      int m;     // number of constraint functions
      int n;     // dimension of the hard constraint manifold = d-m
      
      DynamicVector<double, columnVector>            /* return the values ...           */
      xi( DynamicVector<double, columnVector> q);     /* ... of the constraint functions */
      
      DynamicMatrix<double, columnMajor>             /* column k is the gradient ...*/
      gxi( DynamicVector<double, columnVector> q);    /* ... of xi_k(x)               */
   
      // Returns gxi(q) augmented to a square matrix (in case d > m), just appends columns of zeros.
      DynamicMatrix<double, columnMajor>
      Agxi(DynamicMatrix<double, columnMajor> gxi);
   
      // DEFAULT Constructor
      Model();
   
      // PARAMETRIZED Constructor
      Model( int d0, int m0,               /* ambient dimension and constraint number */
             DynamicVector<double> r0,     /* radii of constraint spheres  */
             DynamicMatrix<double, columnMajor> c0); // column k = cdnter of sphere k
   
      // COPY Constructor
      Model(const Model& M0);
   
      string ModelName();                  /* Return the name of the model */
      
      double yzIntegrate( double x,        /* Integrate e^{-beta*U(x,y,z)} over y and z */
                          double L,        /* Integrate from y = L to y = R    */
                          double R,        /* Integrate z over the same range  */
                          double eps,     /* The temperature parameter*/
                          int n);          /* the number of integration points in each dir */

};

#endif /* Model.hpp */
