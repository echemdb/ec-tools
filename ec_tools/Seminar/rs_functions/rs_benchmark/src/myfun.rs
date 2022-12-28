use ndarray::prelude::*;
use std::f64::consts::PI;


pub fn run_test(n_max: f64)->(Array1<f64>, Array1<f64>) 
{
    // (optional) calculate Numbers
    let N: Array1<f64> = Array1::range(0., n_max, 1.,);
    let N_tmp:Array1<f64> = Array1::range(0., n_max, 1.,);
    // return results
    return (N,N_tmp)
}