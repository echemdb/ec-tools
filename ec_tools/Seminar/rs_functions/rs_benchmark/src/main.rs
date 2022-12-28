// -------------------------------------
// Benchmark with Rust code
// -------------------------------------

use csv;
use std::error::Error;
use std::time::{Duration, Instant};

use ndarray::prelude::*;
use std::f64::consts::PI;

#[allow(non_snake_case)]
mod hemispherical_electrode;
mod Semi_int;


fn main() {    
    
    // Define list of Element sizes
    let N_max: Array1<f64> = array![1000.,2500. ,5000. ,7500.,10000.,12500.,15000.,20000.];

    // Define file name of export csv
    let FILENAME = "rust_Benchmark_Results.csv";

    // ---------------------------------
    // Fast Riemann-Liouville transformation
    // ---------------------------------
    // Initialize 
    let mut t_FRLT: Array1<f64> = Array1::zeros(N_max.len());
    // Error calc NOT implemented yet
    //let mut e_abs_max: Array1<f64> = Array1::zeros(N_max.len());
    //let mut e_rel_max: Array1<f64> = Array1::zeros(N_max.len());

    // Print header
    println!("\n FRLT Algorithm\n");
    //println!(" N     | t (s)   | max abs err | max rel err");
    println!(" N     | t (s)");
    //let mut N
    for i in 0..N_max.len() {
        let N = N_max[i];

        // simple test case
        //let mut I: Array1<f64> = Array1::ones(N as usize);
        //let mut t: Array1<f64> = Array1::range(0., N as f64, 1.,);
        
        // more realistic test case
        let (t,I) = hemispherical_electrode::create_pot(N);

        let delta_x = t[1] - t[0]; 
         
        // timer start
        let t_start = Instant::now();
        let res_1 = Semi_int::semi_integration(&I, delta_x);
        let t_end = t_start.elapsed();
        // timer end
        t_FRLT[i] = t_end.as_secs_f64();
        
        // Results
        println!("{:5.2e} | {:.2e}",N,t_FRLT[i]);
    }   

    // ---------------------------------
    // Gruenwald ALG
    // ---------------------------------
    // Initialize 
    let mut t_G1: Array1<f64> = Array1::zeros(N_max.len());
    // Error calc NOT implemented yet
    //let mut e_abs_max: Array1<f64> = Array1::zeros(N_max.len());
    //let mut e_rel_max: Array1<f64> = Array1::zeros(N_max.len());

    // Print header
    println!("\n Gruenwald Algorithm\n");
    //println!(" N     | t (s)   | max abs err | max rel err");
    println!(" N     | t (s)");
    //let mut N
    for i in 0..N_max.len() {
        let N = N_max[i];

        // simple test case
        //let mut I: Array1<f64> = Array1::ones(N as usize);
        //let mut t: Array1<f64> = Array1::range(0., N as f64, 1.,);
        
        // more realistic test case
        let (t,I) = hemispherical_electrode::create_pot(N);

        // timer start
        let t_start = Instant::now();
        let res_2 = Semi_int::G1(&t,&I);
        let t_end = t_start.elapsed();
        // timer end
        t_G1[i] = t_end.as_secs_f64();
        
        // Results
        println!("{:5.2e} | {:.2e}",N,t_G1[i]);
    }

    // ---------------------------------
    // Riemann-Liouville ALG
    // ---------------------------------
    // Initialize 
    let mut t_R1: Array1<f64> = Array1::zeros(N_max.len());
    // Error calc NOT implemented yet
    //let mut e_abs_max: Array1<f64> = Array1::zeros(N_max.len());
    //let mut e_rel_max: Array1<f64> = Array1::zeros(N_max.len());

    // Print header
    println!("\n Riemann-Liouville Algorithm\n");
    //println!(" N     | t (s)   | max abs err | max rel err");
    println!(" N     | t (s)");
    //let mut N
    for i in 0..N_max.len() {
        let N = N_max[i];

        // simple test case
        //let mut I: Array1<f64> = Array1::ones(N as usize);
        //let mut t: Array1<f64> = Array1::range(0., N as f64, 1.,);
        
        // more realistic test case
        let (t,I) = hemispherical_electrode::create_pot(N);

        // timer start
        let t_start = Instant::now();
        let res_3 = Semi_int::R1(&t,&I);
        let t_end = t_start.elapsed();
        // timer end
        t_R1[i] = t_end.as_secs_f64();
        
        // Results
        println!("{:5.2e} | {:.2e}",N,t_R1[i]);
    }


    // ---------------------------------
    // Export results
    // ---------------------------------
    if let Err(e) = csv_export(FILENAME,&N_max, &t_FRLT, &t_G1, &t_R1) {
        eprintln!("{}", e)
    }
// fn main
}

/// ---------------------------------
/// Function to export the desired arrays
/// ---------------------------------
#[allow(non_snake_case)]
fn csv_export(path: &str, N:&Array1<f64>, t1:&Array1<f64>, t2:&Array1<f64>, t3:&Array1<f64>) -> Result<(), Box<dyn Error>> {
 
    // convert the arrays as strings in order to safe them
    let str_N= N.map(|e| e.to_string());
    let str_t1= t1.map(|e| e.to_string());
    let str_t2= t2.map(|e| e.to_string());
    let str_t3= t3.map(|e| e.to_string());


    // Creates new `Writer` for `stdout` 
    let mut writer = csv::Writer::from_path(path)?;

    // Write records one at a time including the header record.
    writer.write_record(&[
        "N",
        "t_FRLT",
        "t_G1",
        "t_R1",
    ])?;
    
    // loop through all elements
    for i in 0..N.len() {
        writer.write_record(&[
            &str_N[i],
            &str_t1[i],
            &str_t2[i],
            &str_t3[i],
    ])?;
    }
    // A CSV writer maintains an internal buffer, so it's important
    // to flush the buffer when you're done.
    writer.flush()?;

    Ok(())
}