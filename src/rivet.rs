use libc::{c_void, size_t};
use ndarray::Array2;
use num_rational::Rational64;
use std::f64;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;

#[repr(C)]
struct CBar {
    birth: f64,
    death: f64,
    multiplicity: u32,
}

#[repr(C)]
struct CBarCode {
    bars: *const CBar,
    length: size_t,
    angle: f64,
    offset: f64,
}

#[repr(C)]
struct CBarCodesResult {
    pub barcodes: *const CBarCode,
    pub length: size_t,
    pub error: *const c_char,
    pub error_length: size_t, //    pub x_low: f64,
                              //    pub y_low: f64,
                              //    pub x_high: f64,
                              //    pub y_high: f64
}

#[repr(C)]
struct CStructurePoint {
    pub x: u32,
    pub y: u32,
    pub betti_0: u32,
    pub betti_1: u32,
    pub betti_2: u32,
}

#[repr(C)]
struct CRatio {
    pub nom: i64,
    pub denom: i64,
}

#[repr(C)]
struct CExactGrades {
    pub x_grades: *const CRatio,
    pub x_length: size_t,
    pub y_grades: *const CRatio,
    pub y_length: size_t,
}

#[repr(C)]
struct CStructurePoints {
    pub grades: *const CExactGrades,
    pub points: *const CStructurePoint,
    pub length: size_t,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StructurePoint {
    pub x: u32,
    pub y: u32,
    pub betti_0: u32,
    pub betti_1: u32,
    pub betti_2: u32,
}

#[derive(Debug, Clone)]
pub struct BettiStructure {
    pub x_grades: Vec<Rational64>,
    pub y_grades: Vec<Rational64>,
    pub points: Vec<StructurePoint>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Bounds {
    pub x_low: f64,
    pub y_low: f64,
    pub x_high: f64,
    pub y_high: f64,
}

impl Bounds {
    pub fn is_degenerate(&self) -> bool {
        self.x_low == self.x_high || self.y_low == self.y_high
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarCode {
    pub angle: f64,
    pub offset: f64,
    /// barcodes is an m x (birth, death, multiplicity) array
    pub bars: Array2<f64>,
}

#[repr(C)]
struct ArrangementBounds {
    x_low: f64,
    y_low: f64,
    x_high: f64,
    y_high: f64,
}

#[repr(C)]
struct RivetComputationResult {
    computation: *mut RivetArrangement,
    error: *const c_char,
    error_length: size_t,
}

#[link(name = "rivet")]
extern "C" {
    fn read_rivet_computation(bytes: *const u8, length: size_t) -> RivetComputationResult;
    fn bounds_from_computation(computation: *mut RivetArrangement) -> ArrangementBounds;
    fn barcodes_from_computation(
        computation: *mut RivetArrangement,
        angles: *const f64,
        offsets: *const f64,
        query_length: size_t,
    ) -> CBarCodesResult;
    fn structure_from_computation(computation: *const RivetArrangement) -> *mut CStructurePoints;

    fn free_barcodes_result(result: CBarCodesResult) -> c_void;
    fn free_rivet_computation_result(result: RivetComputationResult);
    fn free_structure_points(points: *mut CStructurePoints);
}

enum RivetArrangement {}

pub struct ComputationResult {
    arr: *mut RivetArrangement,
}

impl Drop for ComputationResult {
    fn drop(&mut self) {
        unsafe {
            free_rivet_computation_result(RivetComputationResult {
                computation: self.arr,
                error: ptr::null(),
                error_length: 0,
            });
        }
    }
}

#[derive(Debug)]
pub enum RivetErrorKind {
    InputValidation,
    IO,
    Computation,
}

#[derive(Debug)]
pub struct RivetError {
    message: String,
    kind: RivetErrorKind,
}

pub fn parse(bytes: &[u8]) -> Result<ComputationResult, RivetError> {
    if bytes.len() == 0 {
        Err(RivetError {
            message: "Byte array must have non-zero length".to_string(),
            kind: RivetErrorKind::InputValidation,
        })
    } else {
        let rivet_comp = unsafe { read_rivet_computation(bytes.as_ptr(), bytes.len()) };
        if !rivet_comp.error.is_null() {
            let message = unsafe { CStr::from_ptr(rivet_comp.error) }
                .to_owned()
                .to_str()
                .expect("Could not convert C string")
                .to_string();
            unsafe {
                free_rivet_computation_result(rivet_comp);
            }
            Err(RivetError {
                message: format!("Error while reading computation: {}", message),
                kind: RivetErrorKind::IO,
            })
        } else {
            Ok(ComputationResult {
                arr: rivet_comp.computation,
            })
        }
    }
}

pub fn bounds(computation: &ComputationResult) -> Bounds {
    unsafe {
        let res = bounds_from_computation(computation.arr);

        Bounds {
            x_low: res.x_low,
            y_low: res.y_low,
            x_high: res.x_high,
            y_high: res.y_high,
        }
    }
}

pub fn barcodes(
    computation: &ComputationResult,
    angle_offsets: &[(f64, f64)],
) -> Result<Vec<BarCode>, RivetError> {
    let angles: Vec<f64> = angle_offsets.iter().map(|p| p.0).collect();
    let offsets: Vec<f64> = angle_offsets.iter().map(|p| p.1).collect();
    let mut barcodes: Vec<BarCode> = Vec::new();
    unsafe {
        let cbars = barcodes_from_computation(
            computation.arr,
            angles.as_ptr(),
            offsets.as_ptr(),
            angle_offsets.len(),
        );
        if !cbars.error.is_null() {
            let message = CStr::from_ptr(cbars.error)
                .to_owned()
                .to_str()
                .expect("Could not convert C string")
                .to_string();
            free_barcodes_result(cbars);
            Err(RivetError {
                message,
                kind: RivetErrorKind::Computation,
            })
        } else {
            for bc in 0..cbars.length {
                let bcp = cbars.barcodes.offset(bc as isize);
                let mut bars = Array2::zeros(((*bcp).length, 3));
                for b in 0..(*bcp).length {
                    let bp = (*bcp).bars.offset(b as isize);
                    bars[[b, 0]] = (*bp).birth;
                    bars[[b, 1]] = match (*bp).death {
                        d if d.is_infinite() || d.is_nan() => 1e6, //f64::MAX,
                        d => d,
                    };
                    bars[[b, 2]] = (*bp).multiplicity as f64;
                }
                let angle = (*bcp).angle;
                let offset = (*bcp).offset;
                barcodes.push(BarCode {
                    angle,
                    offset,
                    bars,
                })
            }
            free_barcodes_result(cbars);
            Ok(barcodes)
        }
    }
}

pub fn structure(computation: &ComputationResult) -> BettiStructure {
    unsafe {
        let structure = structure_from_computation(computation.arr);
        let mut points = Vec::with_capacity((*structure).length);
        for p in 0..(*structure).length {
            let cpoint = (*structure).points.offset(p as isize);
            let point = StructurePoint {
                x: (*cpoint).x,
                y: (*cpoint).y,
                betti_0: (*cpoint).betti_0,
                betti_1: (*cpoint).betti_1,
                betti_2: (*cpoint).betti_2,
            };
            points.push(point);
        }
        let x_len = (*(*structure).grades).x_length;
        let mut x_grades = Vec::with_capacity(x_len);
        for x in 0..x_len {
            let x_grade = (*(*structure).grades).x_grades.offset(x as isize);
            x_grades.push(Rational64::from(((*x_grade).nom, (*x_grade).denom)));
        }
        let y_len = (*(*structure).grades).y_length;
        let mut y_grades = Vec::with_capacity(y_len);
        for y in 0..y_len {
            let y_grade = (*(*structure).grades).y_grades.offset(y as isize);
            y_grades.push(Rational64::from(((*y_grade).nom, (*y_grade).denom)));
        }
        free_structure_points(structure);

        BettiStructure {
            x_grades,
            y_grades,
            points,
        }
    }
}

impl ComputationResult {
    pub fn bounds(self: &ComputationResult) -> Bounds {
        return bounds(self);
    }
}

impl Bounds {
    pub fn common_bounds(self: &Bounds, other: &Bounds) -> Bounds {
        Bounds {
            x_low: f64::min(self.x_low, other.x_low),
            y_low: f64::min(self.y_low, other.y_low),
            x_high: f64::max(self.x_high, other.x_high),
            y_high: f64::max(self.y_high, other.y_high),
        }
    }
}
