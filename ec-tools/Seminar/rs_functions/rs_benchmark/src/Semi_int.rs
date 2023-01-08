// -------------------------------------
// Native implementation from python to rust
// -------------------------------------

use ndarray::prelude::*;
use std::f64::consts::PI;
use num_traits::pow;

// G1 Gruenwald Semi-Integration and Semi-Differentiation
// on base from Oldham: Electrochemical Science and Technology, 2012
#[allow(non_snake_case)]
pub fn G1(I: &Array1<f64>,t: &Array1<f64>) -> Array1<f64> {

    // (equidistant) time step
    let delta = t[1]-t[0];
    let sqrt_d = delta.sqrt();
    // No. of steps
    let N_max = I.len();
    
    // initialize with zeros
    let mut G1 = Array1::<f64>::zeros(N_max);
    let mut G1_i = I[0];
    //for N in range(0,N_max):
    for N in 1..N_max+1 {
        // value for n = N with w0 = 1
        G1_i = I[0];
        // reversed loop from N to 0
        for n in (1..N).rev() {
            G1_i = G1_i*(1.-(0.5)/(n as f64)) + I[N-n];
        }
        G1[N-1] = G1_i * sqrt_d;
   
    }
    // return results
    return G1
}

// R1 Riemann and Liouville Semi-Integration and Semi-Differentiation
// on base from Oldham: Electrochemical Science and Technology, 2012
#[allow(non_snake_case)]
pub fn R1(I: &Array1<f64>,t: &Array1<f64>) -> Array1<f64> {

    let myPi = PI as f64;

    // (equidistant) time step
    let delta = t[1]-t[0];

    let sqrt_dPi = (delta/myPi).sqrt();

    // No. of steps
    let N_max = I.len();
    
    // initialize with zeros
    let mut R1 = Array1::<f64>::zeros(N_max);
    let mut R1_i = 0.;

    for N in 1..N_max+1 {
        // value for n = N with w0 = 1
        R1_i = 0.;
        // reversed loop from N to 0
        for n in 1..N {
            R1_i += I[n-1]*(f64::powf((N as f64)-(n as f64)+1.,3./2.) - 2.*f64::powf((N as f64)-(n as f64),3./2.) + f64::powf((N as f64)-(n as f64)-1.,3./2.));
        }
        R1[N-1] = (4./3.)*sqrt_dPi*(I[N-1] + I[0]*(1.5*(N as f64).sqrt()-f64::powf(N as f64,3./2.) + f64::powf((N as f64)-1.,3./2.)) + R1_i);
    }
    // return results
    return R1
}

// Implementation of a algorithm for semi-integration.
// Fast Riemann-Liouville transformation (differintergration) - FRLT
// based on
// Pajkossy, T., Nyikos, L., 1984. Fast algorithm for differintegration. Journal of Electroanalytical Chemistry and Interfacial Electrochemistry 179, 65–69. https://doi.org/10.1016/S0022-0728(84)80275-2
#[allow(non_snake_case)]
pub fn prep_krnl(q: f64, delta_x: f64, N: usize, c1: f64, c2: f64) -> (Array1<f64>,Array1<f64>, Array1<f64>) {
    // Setup the integration kernel with the order q, the x interval delat_x, the length of the array N,
    // and the filter constants c1 and c2.
    let tau0 = delta_x*f64::powf(N as f64,0.5);
    let a0 = f64::sin(PI * q)/(PI*q*f64::powf(tau0,q));
    // total number of filters
    let n_filters = 2. * c1 * c2 + 1.;
    // dimension according to the number of filters
    // filter weights
    let mut w1 = Array1::<f64>::zeros(n_filters as usize);
    let mut w2 = Array1::<f64>::zeros(n_filters as usize);
    // auxiliary array
    let mut s = Array1::<f64>::zeros(n_filters as usize);
    
    for i in 0..(2. * c1 * c2) as usize {
        let j = (i as f64) - c1 * c2;
        let a_j = (a0/ c2) *  (j/c2).exp();
        let t_j = tau0 * (-j/(q*c2)).exp();
        w1[i] = t_j / (delta_x + t_j);
        w2[i] = a_j * (1. - w1[i] );
    }
    return (s,w1,w2)
}

#[allow(non_snake_case)]
pub fn semi_integration (y:&Array1<f64>, delta_x: f64) ->  Array1<f64> {
    // Return the semiintegral R of order q for y with the x interval delta_x and the filter constants
    // c1 and c2.

    // Semi-integrating two times with order q = -0.5 should give the same result as integrating once.
    // The relative error should not exceed 0.25 percent for 1000 and 0.5 percent per 10000 integration steps.
    let q = -0.5;
    let c1 = 8.;
    let c2 = 2.;
    let N = y.len();
    let mut R = Array1::<f64>::zeros(N);
    let (mut s,w1,w2) = prep_krnl(q, delta_x, N, c1, c2);
    for k in 1..N {
        for i in 0..s.len() {
            s[i] = s[i]*w1[i] + y[k]*w2[i];
            R[k] = R[k] + s[i]; 
        }
    }
    return R
}
