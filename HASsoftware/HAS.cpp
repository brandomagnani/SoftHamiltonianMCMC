/*
       Squishy sampler project, the RATTLE Accellerated Sampler approach.
       See main.cpp for more information.
       

*/

#include <iostream>
#include <blaze/Math.h>   // numerical linear algebra and matrix/vector classes
#include <random>         // for the random number generators
#include <cmath>          // defines exp(..), M_PI
#include <string>
#include "model.hpp"
#include "HAS.hpp"
using namespace std;
using namespace blaze;


void HASampler(      vector<double>& chain,        /* Position Samples output from the MCMC run, pre-allocated, length = d*T */
                     struct SamplerStats *stats,     /* statistics about different kinds of rections                          */
                     size_t T,                       /* number of MCMC steps        */
                     double eps,                     /* squish parameter            */
                     double dt,                      /* time step size in RATTLE integrator                                    */
                     double Nsoft,                   /* number of Soft Moves: Gaussian Metropolis move to resample position q  */
                     double Nrattle,                 /* number of Rattle steps      */
                     DynamicVector<double>& q0,      /* starting point              */
                     Model M,                        /* evaluate q(x) and grad q(x) */
                     double ss,                      /* isotropic gaussian size for Soft move              */
                     double neps,                    /* convergence tolerance for Newton projection        */
                     double rrc,                     /* closeness criterion for the reverse check          */
                     int   itm,                      /* maximum number of Newton iterations per projection */
               mt19937 RG) {                   /* random generator engine, already instantiated      */
   
   DynamicVector<double, columnVector> xiDummy = M.xi(q0);// qStart is used only to learn m
   int d  = q0.size();      // infer dimensions, d = dimension of ambient space
   int m  = xiDummy.size();  // m = number of constraints
   int n  = d - m;           // n = dimension of constraint Surface (ie, dimension of its tangent space)
   
   DynamicVector<double, columnVector> Z(d);       // standard gaussian in ambient space R^d
   DynamicVector<double, columnVector> q(d);       // current position sample
   DynamicVector<double, columnVector> p(d);       // current momentum sample
   DynamicVector<double, columnVector> qn(d);      // proposed new position sample, also used for intermediate step in RATTLE integrator
   DynamicVector<double, columnVector> pn(d);      // proposed / new momentum sample, also used for intermediate step in RATTLE integrator
   DynamicVector<double, columnVector> qr(d);      // position for reverse step
   DynamicVector<double, columnVector> pr(d);      // momentum for reverse step
   
   DynamicVector<double, columnVector> z(m);          // level z = S_xi(q)
   DynamicVector<double, columnVector> zn(m);         // level zn = S_xi(qn)
   DynamicVector<double, columnVector> r(m);          // residual in Newton projection step
   DynamicMatrix<double, columnMajor> gtq2gq1(m,m);   // grad(xi)^t(q2)*grad(xi)(q1), used in Newton position projection
   DynamicMatrix<double, columnMajor> gtqrgq2(m,m);   // grad(xi)^t(qr)*grad(xi)(q2), used in Newton position projection
   DynamicMatrix<double, columnMajor> gtygy(m,m);     // grad(xi)^t(y)*grad(xi)(y), used in momentum projection
   
   DynamicVector<double, columnVector> a(m);       // coefficient in Newton's iteration
   DynamicVector<double, columnVector> da(m);      // increment of a in Newton's iteration
   DynamicMatrix<double, columnMajor> gxiqr(d,m);  // gradient matrix of constraint functions at qr; used for RATTLE integrator
   
   DynamicVector<double, columnVector> q1(d);      // intermediate position storage for RATTLE steps
   DynamicVector<double, columnVector> p1(d);      // intermediate momentum storage for RATTLE steps
   DynamicVector<double, columnVector> q2(d);      // intermediate position storage for RATTLE steps
   DynamicVector<double, columnVector> p2(d);      // intermediate momentum storage for RATTLE steps
   DynamicMatrix<double, columnMajor> gxiq1(d,m);  // gradient matrix of constraint functions at q1
   DynamicVector<double, columnVector> xiq1(m);    // constraint function values at q1
   DynamicMatrix<double, columnMajor> gxiq2(d,m);  // gradient matrix of constraint functions at q2
   DynamicVector<double, columnVector> xiq2(m);    // constraint function values at q2
   
   
   DynamicVector<double, columnVector> xiq(m);     // constraint function values at q
   DynamicVector<double, columnVector> xiqn(m);    // constraint function values at qn
   DynamicMatrix<double, columnMajor> gxiq(d,m);   // gradient matrix of constraint functions
   DynamicMatrix<double, columnMajor> gxiqn(d,m);   // gradient matrix of constraint functions
   DynamicMatrix<double, columnMajor> Tq(d,n);     // for basis of tangent space
   DynamicMatrix<double, columnMajor> Tqn(d,n);    // for basis of tangent space
   
   DynamicVector<double, columnVector> R(n);       // isotropic Gaussian of variance 1 sampled in q-tangent space of level surface S_xi(q)
   DynamicMatrix<double, columnMajor> Agxiq(d,d);  // augmented gxi(q) used for 'full' SVD decomposition
   DynamicMatrix<double, columnMajor> Agxiqn(d,d); // augmented gxi(qn) used for 'full' SVD decomposition where qn = position proposal
   DynamicMatrix<double, columnMajor>  U;          // U contains left singular vectors, size (d x d)
   DynamicVector<double, columnVector> s;          // vector contains m singular values, size m
   DynamicMatrix<double, columnMajor>  Vtr;        // V^t where V contains right singular vectors, size (m x m)
   
   double Uq;      // |xi(q)|^2
   double Uqn;     // |xi(qn)|^2
   double A;       //  Metropolis ratio
   double detq, detqn;    // detq = sqrt( det(gtg) ) = det(S) where S is the singular value matrix in reduced SVD of ( gxi * gxi^t )
   double p_sqn, pn_sqn;  // |p|^2, |pn|^2  used in Metropolis-Hastings check following the RATTLE step
   
   //(MODIFY DESCRIPTION)
   int l = -1;               // number of samples coming from Soft moves
   // "l" is the index used to is used to update chain
   
   int       softFlag;                           // a flag describing where in the SOFT move step you are
   const int Met_rej_soft               = 1;         // the proposed qn was rejected by Metropolis
   const int Met_acc_soft               = 2;         // the proposed qn was accepted by Metropolis
   const int accept_qn_soft             = 3;         // accepted the new point
   
   int       ssFlag;                             // a flag describing where in the Surface Sampler move you are
   const int starting_ss             = 1;            // possible values: have not started the y projection
   const int q2_proj_worked_ss       = 2;            // the Newton iteration found qn on the surface xi(qn)=xi(q)
   const int q2_proj_failed_ss       = 3;            // the qn projection Newton iteration ended without success
   const int Met_rej_ss              = 4;            // the proposed qn was rejected by Metropolis
   const int Met_acc_ss              = 5;            // the proposed 1n was accepted by Metropolis
   const int q_reverse_check_worked_ss = 6;     // the POSITION reverse check iteration found q
   const int q_reverse_check_failed_ss = 7;     // the POSITION reverse check iteration failed or found a point that isn't q
   const int p_reverse_check_failed_ss = 8;     // the MOMENTUM reverse check found a point that isn't p
   const int reverse_check_worked_ss   = 9;     // the overall reverse check worked
   const int reverse_check_failed_ss   = 10;    // either (1) POSITION reverse projection did not converge or (2) found point that is not (q,p)
   const int accept_qn_ss             = 11;          // accepted the new point
   
   normal_distribution<double>       SN(0.,1.);   // standard normal (mean zero, variance 1)
   uniform_real_distribution<double> SU(0.,1.);   // standard uniform [0,1]
   
   stats-> SoftSample                        = 0;
   stats-> SoftSampleAccepted                = 0;
   stats-> SoftRejectionMetropolis           = 0;
   
   stats-> HardSample                        = 0;
   stats-> HardSampleAccepted                = 0;
   stats-> HardRejectionFailedProjection_qn  = 0;     // Rattle move: failed projection from q + dt* p in T_q to qn
   stats-> HardRejectionMetropolis           = 0;
   stats-> HardRejectionReverseCheck_q       = 0;
   stats-> HardRejectionReverseCheck_p       = 0;
   stats-> HardRejectionReverseCheck         = 0;
   
   //    Setup for the MCMC iteration: get values at the starting point
   
   // Update these at the end of each move if proposal is accepted
   q   = q0;       // starting point
   xiq  = M.xi(q);   // constraint function evaluated at starting point
   gxiq = M.gxi(q);  // gradient of constraint function at starting point
   
   //    Start MCMC loop
   
   for (unsigned int iter = 0; iter < T; iter++){
      
      //---------------------------------------------Isotropic Gaussian Metropolis step in position space------------------------------------------
      
      for (unsigned int i = 0; i < Nsoft; i++){
         
         stats-> SoftSample++;     // one soft move
         
         // Draw proposal qn: isotropic gaussian ( std = ss ) in ambient space
         for (unsigned int k = 0; k < d; k++){  // Sample Isotropic Standard Gaussian
            Z[k] = SN(RG);
         }
         qn = q + ss*Z;      // Proposal: Isotropic gaussian with mean zero and covariance sm^2*Id
         
         // Do the metropolis detail balance check
         
         Uq    = sqrNorm( M.xi(q) );   // |xi(q)|^2
         xiqn  = M.xi(qn);             // evaluate xi(qn) (also used when processing accepted proposal)
         Uqn    = sqrNorm( xiqn );     // |xi(qn)|^2
         
         A = exp( 0.5*((Uq - Uqn) / (eps*eps)) );   // Metropolis ratio (gaussian proposal simplifies in the ratio, probabilities too)
         
         if ( SU(RG) > A ){      // Accept with probability A,
            softFlag = Met_rej_soft;    // rejected
            stats-> SoftRejectionMetropolis++;
         }
         else{
            softFlag = Met_acc_soft;    // accepted
         }
         
         if ( softFlag == Met_acc_soft ) {     //  process an accepted proposal
            q   = qn;
            gxiq = M.gxi(qn);     // update gradient
            xiq  = xiqn;          // update constraint function
            softFlag = accept_qn_soft;
            stats-> SoftSampleAccepted++;
         }
         else {                                       // process a rejected proposal
         }
         
         l++;   // we drew a Soft sample --> increment index for Schain[]
         for ( int k = 0; k < d; k++){    // add sample to Schain here
            chain[ k + d*l]  = q[k];    // value of q[k] where q is l-th position sample
         }
      } // end of Metropolis move
      
      
      //-----------------------------Re-sample momentum p in T_q = tangent space of level Surface S_xi(q) at point q---------------------------------
      
      
      for ( unsigned int k = 0; k < n; k++){   // Isotropic Gaussian, not tangent
         R[k] = SN(RG);
      }
      
      //       Compute Tq =  basis for tangent space at q. To do so, calculate Full SVD of gxiq = U * S * V^t. Then,
      //       .. Tq = last d-m columns of U in the Full SVD for gxiq
      
      Agxiq = M.Agxi(gxiq);        // add d-n column of zeros to gxiq to get full SVD, needed to get Tq = last d-n columns of U
      svd( Agxiq, U, s, Vtr);        // Computing the singular values and vectors of gxiq
      
      // multiply singular values to get detq = sqrt( det(gxiq^t gxiq) ) = det(S)
      detq = 1.;
      for ( unsigned long int i = 0; i < m; i++){
         detq *= s[i];    // detq = sqrt( det(gxiq^t gxiq) ) = det(S)
      }
      
      //    Build Tq = matrix for tangent space basis at q
      for ( unsigned long int i = m; i < d; i++){
         unsigned long int k = i-m;
         column(Tq,k) = column(U,i);
      }      // end of calculation for Tq
      
      p = Tq * R;   //   new momentum in the q-tangent space of level Surface S_xi(q)
      
      //---------------------------------------------------------------RATTLE steps-------------------------------------------------------------------
      
      ssFlag = starting_ss;
      q1 = q;             // save starting position, needed later for Metropolis check
      p1 = p;             // save starting momentum, needed later for Metropolis check
      gxiq1 = gxiq;
      xiq1  = xiq;
      
      stats-> HardSample++;     // one Rattle move
      
      for (unsigned int i = 0; i < Nrattle; i++){
         
         z = M.xi(q1);     // need the level z = S_xi(q1), can actually spare this computation and save z from the q Metropolis move above
         
         //    First, project q1 + dt*p1 onto the constraint surface S_z
         //    Newton loop to find q2 = q1 + dt*p1 + grad(xi)(q1)*a with xi(q2)=xi(q1)=z
         a = 0;                     // starting coefficient
         q2    = q1 + dt * p1;       // initial guess = move in the tangent direction
         gxiq2 = M.gxi(q2);           // because these are calculated at the end of this loop
         
         for ( int ni = 0; ni < itm; ni++){
            r     = z - M.xi(q2);              // equation residual
            gtq2gq1 = trans( gxiq2 )*gxiq1;   // Newton Jacobian matrix
            solve( dt * gtq2gq1, da, r);      // solve the linear system; Note the * dt on LHS
            q2  +=  dt * gxiq1*da;             // take the Newton step;    Note the * dt when updating
            a  += da;                         // need the coefficient to update momentum later
            gxiq2 = M.gxi(q2);                // constraint gradient at the new point (for later)
            if ( norm(r) <= neps ) {
               ssFlag = q2_proj_worked_ss;     // record that you found q2
               break;                      // stop the Newton iteration
            }   // end of loop if ( norm(r) ...
         }       // end of Newton solver loop

         if ( ssFlag == starting_ss ) {        // the Newton iteration failed, or the flag would be: q2_proj_worked
            ssFlag = q2_proj_failed_ss;         // done with this surface sampler step
            stats->HardRejectionFailedProjection_qn++;
         }
         
         
         if ( ssFlag == q2_proj_worked_ss ) {  // if q-projection was successful ...
            
            //  ... continue and project momentum p2 = p1 + grad(xi)(q1)*a onto T_q2
            p2    = p1 + gxiq1*a;
            gtygy = trans( gxiq2 )*gxiq2;
            r     = - trans( gxiq2 )*p2;
            solve( gtygy, a, r);
            
            // Set the new state to be the ones given by RATTLE with   ** Momentum Reversal **
            p2    = p2 + gxiq2*a;
            p2    = - p2;        // apply momentum reversal ...!!
            // q1 already set at the end of Newton iteration
            // end of RATTLE integrator single step
            
            
         // Now do the REVERSE CHECK in 2 steps :
            
         //    (1) Position Reverse check : does Newton converge to q from qn + tangent vector at qn?
            
            //    Project qn + dt*pn onto the constraint surface S_zn, check if the result is = q
            //    Newton loop to find qr = q2 + dt*p2 + grad(xi)(q2)*a with xi(qr) = xi(q2)= xi(1) = z
            a = 0;                     // starting coefficient
            qr    = q2 + dt * p2;       // initial guess = move in the tangent direction
            gxiqr = M.gxi(qr);           // because these are calculated at the end of this loop
            for ( int ni = 0; ni < itm; ni++){
               r     = z - M.xi(qr);            // equation residual (we project onto same level set as for starting point q)
               gtqrgq2 = trans( gxiqr )*gxiq2;  // Newton Jacobian matrix;
               solve( dt * gtqrgq2, da, r);     // solve the linear system; Note the * dt on LHS  ...!!
               qr +=  dt * gxiq2*da;             // take the Newton step;    Note the * dt when updating ...!!
               a  += da;                        // need the coefficient to update momentum later
               gxiqr = M.gxi(qr);               // constraint gradient at the new point (for later)
               if ( norm(r) <= neps ) {    // If Newton step converged ...
                  if ( norm( qr - q1 ) < rrc ) {      // ... did it converge to the right point?
                     ssFlag = q_reverse_check_worked_ss;
                  }
                  else {
                     ssFlag = q_reverse_check_failed_ss;   // converged to the wrong point -- a failure
                  }
                  break;                   // stop the Newton iteration, it converged
               }
            }       // end of Newton solver loop
            
            
         //    (2) Momentum Reverse check : if Reverse check (1) was successful, then does the projection of pr converge to pn ?
            
            if (ssFlag == q_reverse_check_worked_ss) {   // if position reverse check worked ...
               //  ... continue and project momentum pr = Proj ( p2 + grad(xi)(q2)*a )  onto T_y
               pr    = p2 + gxiq2*a;
               gtygy = trans( gxiqr )*gxiqr;
               r     = - trans( gxiqr )*pr;
               solve( gtygy, a, r);
               
               pr    = pr + gxiqr*a;
               pr    = - pr; // apply momentum reversal !!
               
               if ( norm( pr - p1 ) < rrc ) {      // ... did the reverse momentum projection pr converge to p ?
                  ssFlag = reverse_check_worked_ss;    // if so, then both position and momentum reverse checks worked
                  // so update state :
                  q1   = q2;
                  p1   = p2;
                  gxiq1 = gxiq2;     // update gradient
                  xiq1  = M.xi(q2);   // update constraint function
                  
               }  // so overall reverse check worked !!
               else {
                  ssFlag = p_reverse_check_failed_ss;   // position reverse check iteration converged to the wrong point -- a failure
                  stats->HardRejectionReverseCheck_p++;
               }
            } else {
               stats->HardRejectionReverseCheck_q++;  // if position reverse projection did not converge ..
            }                                         // .. that's another kind of reverse check failure
            
            if ( ssFlag != reverse_check_worked_ss ) {  // If either (qrev, prev) != (q, p) or position reverse projection did not converge ..
               ssFlag = reverse_check_failed_ss;        // .. then the overall reverse check failed
               stats->HardRejectionReverseCheck++;
            }
            
         } // end of reverse check
      
      }  // end of RATTLE step sequence
      
      // store the output state from the sequence of RATTLE steps, needed below for Metropolis check
      qn   = q1;
      pn   = p1;
      gxiqn = gxiq1;
      xiqn  = xiq1;
      
   // Do the Metropolis detailed balance check for (q,p) --> (qn, pn)
      if (true){
         
         //       Compute Tqn =  basis for tangent space at qn. To do so, calculate Full SVD of gxiy = U * S * V^t. Then,
         //       .. Tqn = last d-m columns of U in the Full SVD for gxiq
         
         Agxiqn = M.Agxi(gxiqn);          // add d-n column of zeros to gxiq to get full SVD, needed to get Tqn = last d-n columns of U
         svd( Agxiqn, U, s, Vtr);        // Computing the singular values and vectors of gxiqn
         
         // multiply singular values to get detqn = r(qn) = sqrt( det(gxiy^t gxiy) ) = det(S)
         detqn = 1.;
         for ( unsigned long int i = 0; i < m; i++){
            detqn *= s[i];    // detqn = sqrt( det(gxiy^t gxiy) ) = det(S)
         }
         
         //    Build Tqn = matrix for tangent space basis at qn
         for ( unsigned long int i = m; i < d; i++){
            unsigned long int k = i-m;
            column(Tqn,k) = column(U,i);
         }      // end of calculation for Tqn
         
         p_sqn  = sqrNorm(p);   // |p|^2  for M-H ratio
         pn_sqn = sqrNorm(pn);  // |pn|^2 for M-H ratio
         
         // NOTE: Here can add V(q) and V(qn) to M-H ratio, for now assume V=0
         
         A  = exp( .5*( p_sqn - pn_sqn ) ); //  part of the Metropolis ratio
         A *= ( detq / detqn );    // since r(q)/r(qn) = detq/detqn
         
         if ( SU(RG) > A ){      // Accept with probability A,
            ssFlag = Met_rej_ss;                       // rejected
            stats->HardRejectionMetropolis++;
         }
         else{
            ssFlag = Met_acc_ss;                       // accepted
         }    // Metropolis rejection step done
      }
      
      if ( ssFlag ==  Met_acc_ss) {     //  process an accepted proposal
         q   = qn;
         p   = pn;
         gxiq = gxiqn;     // update gradient
         xiq  = xiqn;   // update constraint function
         ssFlag = accept_qn_ss;
         stats-> HardSampleAccepted++;
      }
      else {                                       // process a rejected proposal
      }
         
   } // end of MCMC loop
   
} // end of sampler


























