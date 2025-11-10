/// -------------------------------------
/// Benchmark with Rust code
/// -------------------------------------
/// Semi-integration algorithms are (native) implemented in rust 
/// and results (e.g. nof. Elems, measured time) are exported 
/// to a *csv file, which will be later loaded in a python file
/// 
/// Error calculation NOT IMPLEMENTED yet, export computed results instead.

use csv;
use std::error::Error;
use std::time::{Instant};

use ndarray::prelude::*;
use std::f64::consts::PI;

#[allow(non_snake_case)]
mod Semi_int;


fn main() {    
    // define pi 
    let my_pi = PI as f64;

    // Define list of Element sizes
    let n_max: Array1<f64> = array![1000.,2500.,5000.,7500.,10000.,12500.,15000.,20000.];

    // Define file name of export csv
    let filename = "results/rs_benchmark";
    
    // for testing set false
    let export = true;
    
    // Initialize time arrays
    let mut t_frlt: Array1<f64> = Array1::zeros(n_max.len());
    let mut t_g1: Array1<f64> = Array1::zeros(n_max.len());
    let mut t_r1: Array1<f64> = Array1::zeros(n_max.len());

    // Print header
    println!("  Elem   | t_FRLT  | t_G1    | t_R1 ");
    println!("   [-]   |    [s]  |  [s]    |  [s] ");
    println!("---------|---------|---------|---------");
    //let mut N
    for i in 0..n_max.len() {
        // take Element size
        let n_i = usize::from(n_max[i] as usize);
    
        // test case (gaussian distribution, similar to scipy.norm.pdf)
        let t: Array1<f64> = Array1::linspace(0., 8., n_i+1);
        
        let tmp1: Array1<f64> =&t-4.;
        let tmp2 = tmp1.mapv(|tmp1| -tmp1.powi(2)/2.);
        let tmp3 = tmp2.mapv_into(|v|v.exp()); 
        let y = tmp3/(my_pi*2.).sqrt();


        // Fast Riemann-Liouville transformation ALG
        let delta_x = t[1] - t[0]; 

        let t_start = Instant::now();
        let res_1 = Semi_int::semi_integration( &Semi_int::semi_integration(&y, delta_x), delta_x);
        let t_end = t_start.elapsed();
        t_frlt[i] = t_end.as_secs_f64();

        print!("{:8.2e} | {:.2e}",n_i,t_frlt[i]);

        // Gruenwald ALG
        let t_start = Instant::now();
        let res_2 = Semi_int::G1(&Semi_int::G1(&y,&t),&t);
        let t_end = t_start.elapsed();
        t_g1[i] = t_end.as_secs_f64();
        print!(" | {:6.2e}",t_g1[i]);
  
        // Riemann-Liouville ALG
        let t_start = Instant::now();
        let res_3 = Semi_int::R1(&Semi_int::R1(&y,&t),&t);
        let t_end = t_start.elapsed();
        t_r1[i] = t_end.as_secs_f64();

        print!(" | {:5.2e}\n",t_r1[i]);
        
        // Export result values (for error calc later in python)
        if export == true {
            // define export var names
            let var_names = ["t","res_FRLT","res_G1", "res_R1"];
            // combine strings as export name
            let merged_name = &*format!("{}_{}.csv",filename,n_max[i]);
            // export time results
            if let Err(e) = csv_export(merged_name,&t, &res_1, &res_2, &res_3, 
                var_names[0], var_names[1], var_names[2], var_names[3]) {
                eprintln!("{}", e)
            }
        }    

    }   
    

    // Export time results & export path name
    if export == true {
        // define export var names
        let var_names = ["N","t_FRLT","t_G1","t_R1"];
        // export path name
        let merged_name = &*format!("{}_time.csv",filename);
    
        // export time results
        if let Err(e) = csv_export(merged_name,&n_max, &t_frlt, &t_g1, &t_r1, 
                                                     var_names[0], var_names[1], var_names[2], var_names[3]) {
            eprintln!("{}", e)
        }

        // path name
        let file_name = &*format!("Export_Name.csv");
        //  path export name
        let merged_name = &*format!("{}",filename);
        // export file name of csv files
        if let Err(e) = csv_export_name(file_name, merged_name) {
            eprintln!("{}", e)
        }
        
    } else {println!("\nNo export done per request!")}

// fn main
}


// Function to export the desired arrays
#[allow(non_snake_case)]
fn csv_export(path: &str, Val0:&Array1<f64>, Val1:&Array1<f64>, Val2:&Array1<f64>, Val3:&Array1<f64>, 
    n0:&str, n1:&str, n2:&str ,n3:&str) -> Result<(), Box<dyn Error>> {
    // INPUT:
    // path (&str)                 Path + Filename 
    // Val0-Val3(&Array1<f64>)     1D-Array of Values
    // N1-N3(&str)                 Header (Variables) Names
    //
    //  OUTPUT:
    // Result                      Returns error if export not possible

    // convert the arrays as strings in order to safe them
    let str_0= Val0.map(|e| e.to_string());
    let str_1= Val1.map(|e| e.to_string());
    let str_2= Val2.map(|e| e.to_string());
    let str_3= Val3.map(|e| e.to_string());

    // Creates new `Writer` for `stdout` 
    let mut writer = csv::Writer::from_path(path)?;

    // Write records one at a time including the header record.
    writer.write_record(&[
        n0,
        n1,
        n2,
        n3,
    ])?;
    
    // loop through all elements
    for i in 0..Val0.len() {
        writer.write_record(&[
            &str_0[i],
            &str_1[i],
            &str_2[i],
            &str_3[i],
    ])?;
    }
    // A CSV writer maintains an internal buffer, so it's important
    // to flush the buffer when you're done.
    writer.flush()?;

    Ok(())
}

/// Function to export the csv file names
#[allow(non_snake_case)]
fn csv_export_name(path: &str, n0:&str) -> Result<(), Box<dyn Error>> {
    // INPUT:
    // path (&str)                 Path + Filename 
    // n0(&str)                    Header Names
    //
    //  OUTPUT:
    // Result                      Returns error if export not possible

    // Creates new `Writer` for `stdout` 
    let mut writer = csv::Writer::from_path(path)?;

    // Write records one at a time including the header record.
    writer.write_record(&[
        n0
    ])?;

    // A CSV writer maintains an internal buffer, so it's important
    // to flush the buffer when you're done.
    writer.flush()?;

    Ok(())
}