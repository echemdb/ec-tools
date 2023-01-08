// Based on Oldham: Electrochemical Science and Technology, 2012 (web resources: Web#1244, Web#1245) 
// Calculates nernstian steady-state voltammetry at a hemispherical microelectrode (eq. 12:19)
// native implementation from python to rust

use ndarray::prelude::*;
use std::f64::consts::PI;

// supress snake case warnings
#[allow(non_snake_case)]
pub fn create_pot(N_max: f64) -> (Array1<f64>, Array1<f64>) {

    let myPi = PI as f64;

    // start and end value
    let E_0 = -0.25;
    let E_final = 0.25;
    // ------------------------
    // Define constants:
    // ------------------------
    // Universal gas constant
    let R = 8.31446261815324; 
    // Faraday constant
    let F = 96485.33212331; 
    
    let u = 0.025;
    let T = 298.15;
    let D_R = 1e-9;
    let D_O = 1e-9;
    let c_bR = 1.;
    let r_hemi = 5e-6;
    let E_oi = 0.;
    //let v = 0.5;

    // (optional) calculate Numbers
    let N: Array1<f64> = Array1::range(0., N_max, 1.,);
    //println!("{}",N);
 
    // create time segments
    let delta_t = (E_final - E_0)/(u*N_max);
    let t: Array1<f64> = N*delta_t;
    
    // create time-dependent Potential
    let E: Array1<f64> = E_0 + u*&t;
    
    // calculate Current (in nA)
    let tmp1: Array1<f64> = -F*(&E-E_oi)/(R*T); 
    let exp_tmp = tmp1.mapv_into(|v|v.exp());

    let I: Array1<f64> = 1e9*(2.*myPi*F*D_R*D_O*c_bR*r_hemi)/(D_O + D_R*exp_tmp);
    return (t, I)
}