use pyo3::{pymodule, types::PyModule, PyResult, Python};
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyReadonlyArray1, PyArray1};

use std::f64::consts::PI;

#[pymodule]
fn semi_integration_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    fn prepare_kernel(q: f64, Δx: f64, N: u64, c1: u64, c2: u64) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let length = N as f64;
        let τ0 = Δx * length.powf(0.5);
        let a0 = (PI * q).sin() / (PI *q * τ0.powf(q));
        // total number of filters
        let n_filters = 2 * c1 * c2 + 1;
        // dimension according to the number of filters
        // filter weights
        let mut w1 = Array1::<f64>::zeros(n_filters as usize);
        let mut w2 = Array1::<f64>::zeros(n_filters as usize);
        // auxiliary array
        let s = Array1::<f64>::zeros(n_filters as usize);
    
        for i in 0..2*c1 as usize *c2 as usize {
            let j:f64 = i as f64 - c1 as f64 * c2 as f64;
            let a_j = (a0 / c2 as f64) * f64::exp(j as f64 / c2 as f64);
            let t_j = τ0 * f64::exp(-j as f64 / (q*c2 as f64) );
            w1[i] = t_j / (Δx + t_j);
            w2[i] = a_j * (1f64 - w1[i])};
        (s, w1, w2)
    }

    fn semi_integration(y: ArrayView1<'_, f64>, q: Option<f64>, Δx: Option<f64>, c1: Option<u64>, c2: Option<u64>) -> Array1<f64> {
        let N = y.len();
        let mut R = Array1::<f64>::zeros(N);
        let ( mut s, w1, w2) = prepare_kernel(
            q.unwrap_or( -0.5),
            Δx.unwrap_or(1.0),
            N as u64,
            c1.unwrap_or(2u64),
            c2.unwrap_or(8u64)
        );
        for k in 1..N {
            for i in 0..s.len() {
                s[i] = s[i]*w1[i] + &y[k]*w2[i];
                R[k] = R[k] + s[i];
            }
        }
    R
    } 
// wrapper of `semi_integration_rs`
#[pyfn(m)]
#[pyo3(name = "semi_integration")]
fn semi_integration_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    q: Option<f64>,
    Δx: Option<f64>,
    c1: Option<u64>,
    c2: Option<u64>,
) -> &'py PyArray1<f64> {
    semi_integration(y.as_array(), q, Δx, c1, c2).into_pyarray(py)
    // Array1::<f64>::zeros(2).into_pyarray(py)
}

// wrapper of `prepare_kernel_rs`
#[pyfn(m)]
#[pyo3(name = "prepare_kernel")]
fn prepare_kernel_py<'py>(
    py: Python<'py>,
    q: f64,
    Δx: f64,
    N: u64,
    c1: u64,
    c2: u64,
) -> (&'py PyArray1<f64>, &'py PyArray1<f64>, &'py PyArray1<f64>) {
    let (s , w1, w2) = prepare_kernel(q, Δx, N as u64, c1, c2);
    (s.into_pyarray(py) , w1.into_pyarray(py), w2.into_pyarray(py))
}
    Ok(())
}