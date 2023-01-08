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
use std::time::{Duration, Instant};

use ndarray::prelude::*;
use std::f64::consts::PI;

#[allow(non_snake_case)]
mod hemispherical_electrode;
mod Semi_int;


fn main() {    
    
    // Define list of Element sizes
    let N_max: Array1<f64> = array![1000.,2500.,5000.,7500.,10000.,12500.,15000.,20000.];

    // Define file name of export csv
    let FILENAME = "results/rs_benchmark";
    
    // for testing set false
    let Export = true;
    
    // Initialize time arrays
    let mut t_FRLT: Array1<f64> = Array1::zeros(N_max.len());
    let mut t_G1: Array1<f64> = Array1::zeros(N_max.len());
    let mut t_R1: Array1<f64> = Array1::zeros(N_max.len());

    // Print header
    println!("  Elem   | t_FRLT  | t_G1    | t_R1 ");
    println!("   [-]   |    [s]  |  [s]    |  [s] ");
    println!("---------|---------|---------|---------");
    //let mut N
    for i in 0..N_max.len() {
        // take Element size
        let N = N_max[i];

        // test case
        let (t,I) = hemispherical_electrode::create_pot(N);
        
        // Fast Riemann-Liouville transformation ALG
        let delta_x = t[1] - t[0]; 

        let t_start = Instant::now();
        let res_1 = Semi_int::semi_integration( &Semi_int::semi_integration(&I, delta_x), delta_x);
        let t_end = t_start.elapsed();
        t_FRLT[i] = t_end.as_secs_f64();

        print!("{:8.2e} | {:.2e}",N,t_FRLT[i]);

        // Gruenwald ALG
        let t_start = Instant::now();
        let res_2 = Semi_int::G1(&Semi_int::G1(&I,&t),&t);
        let t_end = t_start.elapsed();
        t_G1[i] = t_end.as_secs_f64();

        print!(" | {:6.2e}",t_G1[i]);
  
        // Riemann-Liouville ALG
        let t_start = Instant::now();
        let res_3 = Semi_int::R1(&Semi_int::R1(&I,&t),&t);
        let t_end = t_start.elapsed();
        t_R1[i] = t_end.as_secs_f64();

        print!(" | {:5.2e}\n",t_R1[i]);
        
        // Export result values (for error calc later in python)
        if Export == true {
            // define export var names
            let Var_Names = ["t","res_FRLT","res_G1", "res_R1"];
            // combine strings as export name
            let Merged_Name = &*format!("{}_{}.csv",FILENAME,N_max[i]);
            // export time results
            if let Err(e) = csv_export(Merged_Name,&t, &res_1, &res_2, &res_3, 
                Var_Names[0], Var_Names[1], Var_Names[2], Var_Names[3]) {
                eprintln!("{}", e)
            }
        }    

    }   
    

    // Export time results & export path name
    if Export == true {
        // define export var names
        let Var_Names = ["N","t_FRLT","t_G1","t_R1"];
        // export path name
        let Merged_Name = &*format!("{}_time.csv",FILENAME);
    
        // export time results
        if let Err(e) = csv_export(Merged_Name,&N_max, &t_FRLT, &t_G1, &t_R1, 
                                                     Var_Names[0], Var_Names[1], Var_Names[2], Var_Names[3]) {
            eprintln!("{}", e)
        }

        // path name
        let FILE_Name = &*format!("Export_Name.csv");
        //  path export name
        let Merged_Name = &*format!("{}",FILENAME);
        // export file name of csv files
        if let Err(e) = csv_export_name(FILE_Name, Merged_Name) {
            eprintln!("{}", e)
        }
        
    } else {println!("\nNo export done per request!")}

// fn main
}


/// Function to export the desired arrays
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