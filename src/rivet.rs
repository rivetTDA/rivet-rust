use libc::{c_void, size_t};
use ndarray::Array2;
use num_rational::Rational64;
use std::f64;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use noisy_float::prelude::*;
use tempdir::TempDir;
use std::fs::File;
use std::io::prelude::*;
use std::process::Command;
use itertools::Itertools;
use std::fmt::Display;
use std::fmt::Formatter;
use std::io::{BufReader, Read};
use std::convert::From;
use failure::{Fail, Context, Backtrace, ResultExt};

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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BettiStructure {
    pub x_grades: Vec<Rational64>,
    pub y_grades: Vec<Rational64>,
    pub points: Vec<StructurePoint>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Bounds {
    rect: Rectangle
}

impl Display for Bounds {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let d0_ends = self.d0().ends();
        let d1_ends = self.d1().ends();
        write!(f, "Bounds(d0: [{}, {}], d1: [{}, {}])", d0_ends.0, d0_ends.1, d1_ends.0, d1_ends.1)
    }
}

impl Bounds {
    pub fn new(y_low: f64, y_high: f64, x_low: f64, x_high: f64) -> Option<Bounds> {
        let rect = Rectangle::closed(
            (r64(y_low), r64(y_high)),
            (r64(x_low), r64(x_high)),
        )?;
        Some(Bounds { rect })
    }

    pub fn union(&self, other: &Bounds) -> Bounds {
        Bounds { rect: self.rect.union(&other.rect) }
    }
}

impl Rectangular for Bounds {
    fn d0(&self) -> Interval {
        self.rect.d0()
    }

    fn d1(&self) -> Interval {
        self.rect.d1()
    }

    fn scale(&self, scales: (R64, R64)) -> Self {
        Bounds { rect: self.rect.scale(scales) }
    }

    fn translate(&self, offsets: (R64, R64)) -> Self {
        Bounds { rect: self.rect.translate(offsets) }
    }
}


pub enum DimensionQueryResult {
    Low,
    High,
    In(usize),
}

#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub struct Dimension {
    /// The lower bound of this dimension
    pub lower_bound: R64,
    /// The upper bound of this dimension
    pub upper_bounds: Vec<R64>,
    /// A collection of all the indices (in the backing matrix of a SplitMat) that map to
    /// each of the upper bounds (i.e. upper_indexes is always the same length as upper_bounds).
    pub upper_indexes: Vec<Option<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interval {
    Open(R64, R64),
    Closed(R64, R64),
    OpenClosed(R64, R64),
    ClosedOpen(R64, R64),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenClosed {
    Open,
    Closed,
}

impl Interval {
    pub fn new(
        start: R64,
        start_included: OpenClosed,
        end: R64,
        end_included: OpenClosed,
    ) -> Option<Interval> {
        use self::OpenClosed::*;
        if start <= end {
            Some(match (start_included, end_included) {
                (Open, Open) => Interval::Open(start, end),
                (Closed, Closed) => Interval::Closed(start, end),
                (Open, Closed) => Interval::OpenClosed(start, end),
                (Closed, Open) => Interval::ClosedOpen(start, end),
            })
        } else {
            let left = match start_included {
                Open => "(",
                Closed => "["
            };
            let right = match end_included {
                Open => ")",
                Closed => "]"
            };
            warn!("Not an interval: {}{},{},{}", left, start, end, right);
            None
        }
    }

    pub fn ends(&self) -> (R64, R64) {
        match self {
            Interval::Open(start, end) => (*start, *end),
            Interval::Closed(start, end) => (*start, *end),
            Interval::OpenClosed(start, end) => (*start, *end),
            Interval::ClosedOpen(start, end) => (*start, *end),
        }
    }

    pub fn len(&self) -> R64 {
        let ends = self.ends();
        ends.1 - ends.0
    }

    pub fn end_types(&self) -> (OpenClosed, OpenClosed) {
        match self {
            Interval::Open(_, _) => (OpenClosed::Open, OpenClosed::Open),
            Interval::Closed(_, _) => (OpenClosed::Closed, OpenClosed::Closed),
            Interval::OpenClosed(_, _) => (OpenClosed::Open, OpenClosed::Closed),
            Interval::ClosedOpen(_, _) => (OpenClosed::Closed, OpenClosed::Open),
        }
    }

    pub fn intersection(&self, other: &Interval) -> Option<Interval> {
        let (self_start, self_end) = self.ends();
        let (other_start, other_end) = other.ends();
        if self_start > other_end || other_start > self_end {
            None
        } else {
            let start = std::cmp::max(self_start, other_start);
            let end = std::cmp::min(self_end, other_end);
            let (self_start_type, self_end_type) = self.end_types();
            let (other_start_type, other_end_type) = other.end_types();
            let start_type = if self_start != other_start {
                OpenClosed::Closed
            } else {
                match (self_start_type, other_start_type) {
                    (OpenClosed::Closed, OpenClosed::Closed) => OpenClosed::Closed,
                    _ => OpenClosed::Open,
                }
            };
            let end_type = if self_end != other_end {
                OpenClosed::Closed
            } else {
                match (self_end_type, other_end_type) {
                    (OpenClosed::Closed, OpenClosed::Closed) => OpenClosed::Closed,
                    _ => OpenClosed::Open,
                }
            };
            Some(
                Interval::new(start, start_type, end, end_type)
                    .expect("Couldn't create interval!"))
        }
    }


    pub fn union(&self, other: &Interval) -> Interval {
        let (self_start, self_end) = self.ends();
        let (other_start, other_end) = other.ends();
        let start = std::cmp::min(self_start, other_start);
        let end = std::cmp::max(self_end, other_end);
        let (self_start_type, self_end_type) = self.end_types();
        let (other_start_type, other_end_type) = other.end_types();
        let start_type = if self_start < other_start {
            self_start_type
        } else if other_start < self_start {
            other_start_type
        } else if other_start_type == OpenClosed::Closed || self_start_type == OpenClosed::Closed {
            OpenClosed::Closed
        } else {
            OpenClosed::Open
        };
        let end_type = if self_end > other_end {
            self_end_type
        } else if other_end > self_end {
            other_end_type
        } else if other_end_type == OpenClosed::Closed || self_end_type == OpenClosed::Closed {
            OpenClosed::Closed
        } else {
            OpenClosed::Open
        };
        Interval::new(start, start_type, end, end_type)
            .expect("Couldn't create interval!")
    }

    pub fn contains(&self, other: &Interval) -> bool {
        match self.intersection(other) {
            None => false,
            Some(inter) => inter == *other
        }
    }

    pub fn is_empty(&self) -> bool {
        let (start, end) = self.ends();
        start == end
    }

    pub fn scale(&self, scale: R64) -> Interval {
        match &self {
            Interval::Closed(s, e) => Interval::Closed(s.clone(), *e * scale),
            Interval::OpenClosed(s, e) => Interval::OpenClosed(s.clone(), *e * scale),
            Interval::ClosedOpen(s, e) => Interval::ClosedOpen(s.clone(), *e * scale),
            Interval::Open(s, e) => Interval::Open(s.clone(), *e * scale)
        }
    }

    pub fn translate(&self, offset: R64) -> Interval {
        match &self {
            Interval::Closed(s, e) => Interval::Closed(*s + offset, *e + offset),
            Interval::OpenClosed(s, e) => Interval::OpenClosed(*s + offset, *e + offset),
            Interval::ClosedOpen(s, e) => Interval::ClosedOpen(*s + offset, *e + offset),
            Interval::Open(s, e) => Interval::Open(*s + offset, *e + offset)
        }
    }
}

const DIMENSION_VERIFY: bool = false;

impl Dimension {
    pub fn new(lower_bound: R64, upper_bounds: Vec<R64>) -> Result<Dimension, RivetError> {
        let dim = if upper_bounds.len() == 0 {
            Dimension {
                lower_bound,
                upper_bounds: vec![lower_bound],
                upper_indexes: vec![Some(0)],
            }
        } else {
            if lower_bound > upper_bounds[0] {
                invalid(&format!("lower bound {} should be below upper bound {}",
                                 lower_bound, upper_bounds[0]))?
            }
            // assert sorted(list(upper_bounds)) == list(upper_bounds)
            let upper_indexes = (0..upper_bounds.len()).map(Some).collect_vec();
            Dimension {
                lower_bound,
                upper_bounds,
                upper_indexes,
            }
        };
        dim.verify();
        Ok(dim)
    }

    pub fn upper_bound(&self) -> R64 {
        *self.upper_bounds.last().unwrap()
    }

    pub fn from_f64s(lower_bound: f64, upper_bounds: &[f64]) -> Result<Dimension, RivetError> {
        Dimension::new(
            r64(lower_bound),
            upper_bounds.iter().map(|&x| r64(x)).collect_vec(),
        )
    }

    pub fn reset(&self) -> Dimension {
        let dim = Dimension {
            lower_bound: self.lower_bound,
            upper_bounds: self.upper_bounds.clone(),
            upper_indexes: (0..self.upper_bounds.len()).map(Some).collect_vec(),
        };
        if DIMENSION_VERIFY {
            dim.verify();
        }
        dim
    }

    pub fn lengths(&self) -> Vec<R64> {
        let mut bounds = vec![self.lower_bound];
        bounds.extend_from_slice(&self.upper_bounds);
        let mut lengths = Vec::<R64>::with_capacity(bounds.len() - 1);
        for i in 1..bounds.len() {
            lengths.push(bounds[i] - bounds[i - 1]);
        }
        lengths
    }

    pub fn intervals(&self) -> Vec<Interval> {
        let mut results = Vec::with_capacity(self.upper_bounds.len());
        results.push(Interval::Closed(self.lower_bound, self.upper_bounds[0]));
        for i in 1..self.len() {
            results.push(Interval::OpenClosed(
                self.upper_bounds[i - 1],
                self.upper_bounds[i],
            ));
        }
        results
    }

    pub fn len(&self) -> usize {
        self.upper_bounds.len()
    }

    pub fn translate(&self, increment: R64) -> Dimension {
        Dimension {
            lower_bound: self.lower_bound + increment,
            upper_bounds: self
                .upper_bounds
                .iter()
                .map(|x| *x + increment)
                .collect_vec(),
            upper_indexes: self.upper_indexes.clone(),
        }
    }

    pub fn scale(&self, factor: R64) -> Dimension {
        Dimension {
            lower_bound: self.lower_bound,
            upper_bounds: self.upper_bounds.iter().map(|x| *x * factor).collect_vec(),
            upper_indexes: self.upper_indexes.clone(),
        }
    }

    pub fn add_bound(&mut self, bound: R64) {
        if DIMENSION_VERIFY {
            self.verify();
        }
        if bound < self.lower_bound {
            self.upper_bounds.insert(0, self.lower_bound);
            self.upper_indexes.insert(0, None);
            self.lower_bound = bound;
        } else if bound > *self.upper_bounds.last().expect("No upper bounds?") {
            self.upper_bounds.push(bound);
            self.upper_indexes.push(None);
        } else if self.is_bound(bound) {
            //Nothing to do
        } else {
            for i in 0..self.upper_bounds.len() {
                if self.upper_bounds[i] > bound {
                    let duplicate = self.upper_indexes[i];
                    self.upper_bounds.insert(i, bound);
                    self.upper_indexes.insert(i, duplicate);
                    break;
                }
            }
        }
        if DIMENSION_VERIFY {
            self.verify();
        }
    }

    pub fn is_bound(&self, bound: R64) -> bool {
        bound == self.lower_bound || self.upper_bounds.contains(&bound)
    }

    pub fn index(&self, value: R64) -> DimensionQueryResult {
        if value < self.lower_bound {
            DimensionQueryResult::Low
        } else if value > *self.upper_bounds.last().unwrap() {
            DimensionQueryResult::High
        } else {
            for (i, bound) in self.upper_bounds.iter().enumerate() {
                if value <= *bound {
                    return DimensionQueryResult::In(i);
                }
            }
            panic!("the impossible happened - value is neither less, greater, nor in the bounds collection");
        }
    }

    pub fn merge(&self, other: &Dimension) -> Dimension {
        let mut result = self.clone();
        result.add_bound(other.lower_bound);
        for &bound in &other.upper_bounds {
            result.add_bound(bound);
        }
        if DIMENSION_VERIFY {
            result.verify();
        }
        result
    }

    //TODO: inline and remove?
    pub fn verify(&self) {
        self.valid().unwrap();
    }

    pub fn valid(&self) -> Result<&Dimension, RivetError> {
        let mut test = Vec::with_capacity(self.upper_bounds.len());
        for b in self.upper_bounds.iter() {
            if !test.contains(&b) {
                test.push(&b);
            } else {
                Err(RivetErrorKind::Validation(format!("Duplicate element {} found in Dimension", &b)))?
            }
        }
        if is_sorted(&test) {
            Ok(self)
        } else {
            Err(RivetErrorKind::Validation("Dimension is not sorted".to_owned()))?
        }
    }
}

pub fn is_sorted<T: Ord>(thing: &[T]) -> bool {
    let mut last: Option<&T> = None;
    for t in thing {
        match last {
            None => {}
            Some(t_0) => {
                if t_0 > t {
                    return false;
                }
            }
        }
        last = Some(t)
    }
    true
}

pub trait Rectangular {
    fn d0(&self) -> Interval;
    fn d1(&self) -> Interval;

    fn intersection<R: Rectangular>(&self, other: &R) -> Option<Rectangle> {
        let d0 = self.d0().intersection(&other.d0())?;
        let d1 = self.d1().intersection(&other.d1())?;
        Some(Rectangle { d0, d1 })
    }

    fn rectangle(&self) -> Rectangle {
        Rectangle::new(self.d0(), self.d1())
    }

    fn area(&self) -> R64 {
        let (start0, end0) = self.d0().ends();
        let (start1, end1) = self.d1().ends();
        (end0 - start0) * (end1 - start1)
    }

    fn contains<R: Rectangular>(&self, other: &R) -> bool {
        self.d0().contains(&other.d0()) && self.d1().contains(&other.d1())
    }

    fn is_empty(&self) -> bool {
        self.d0().is_empty() && self.d1().is_empty()
    }

    fn scale(&self, scales: (R64, R64)) -> Self;
    fn translate(&self, offsets: (R64, R64)) -> Self;
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Rectangle {
    d0: Interval,
    d1: Interval,
}

impl Rectangle {
    pub fn new(d0: Interval, d1: Interval) -> Rectangle {
        Rectangle { d0, d1 }
    }

    pub fn closed(d0: (R64, R64), d1: (R64, R64)) -> Option<Rectangle> {
        Some(Rectangle {
            d0: Interval::new(d0.0, OpenClosed::Closed, d0.1, OpenClosed::Closed)?,
            d1: Interval::new(d1.0, OpenClosed::Closed, d1.1, OpenClosed::Closed)?,
        })
    }

    pub fn union(&self, other: &Rectangle) -> Rectangle {
        Rectangle {
            d0: self.d0.union(&other.d0),
            d1: self.d1.union(&other.d1)
        }
    }
}

impl Rectangular for Rectangle {
    fn d0(&self) -> Interval {
        self.d0.clone()
    }

    fn d1(&self) -> Interval {
        self.d1.clone()
    }

    fn scale(&self, scales: (R64, R64)) -> Self {
        Rectangle {
            d0: self.d0.scale(scales.0),
            d1: self.d1.scale(scales.1),
        }
    }

    fn translate(&self, offsets: (R64, R64)) -> Self {
        Rectangle {
            d0: self.d0.translate(offsets.0),
            d1: self.d1.translate(offsets.1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Region {
    pub rectangle: Rectangle,
    pub value: Option<R64>,
}

impl Region {
    pub fn new(rectangle: Rectangle, value: Option<R64>) -> Region {
        Region { rectangle, value }
    }

    pub fn intersection(&self, other: &Rectangle) -> Option<Region> {
        self.rectangle.intersection(other).map(|rect| Region {
            rectangle: rect,
            value: self.value,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradedBounds {
    pub x: Dimension,
    pub y: Dimension,
}

impl GradedBounds {
    pub fn valid(&self) -> Result<&GradedBounds, RivetError> {
        self.x.valid()
            .and_then(|_| {
                self.y.valid()
            })
            .and_then(|_| {
                Ok(self)
            })
    }

    pub fn union(self: &GradedBounds, other: &GradedBounds) -> GradedBounds {
        GradedBounds {
            x: self.x.merge(&other.x),
            y: self.y.merge(&other.y),
        }
    }

    pub fn rectangles(&self) -> Array2<Rectangle> {
        let d0_ints = self.y.intervals();
        let d1_ints = self.x.intervals();
        let results =
            Array2::from_shape_fn((d0_ints.len(), d1_ints.len()), |(row_idx, col_idx)| {
                Rectangle {
                    d0: d0_ints[row_idx].clone(),
                    d1: d1_ints[col_idx].clone(),
                }
            });
        results
    }

    pub fn bounds(&self) -> Bounds {
        Bounds { rect: self.rectangle() }
    }
}

impl Rectangular for GradedBounds {
    fn d0(&self) -> Interval {
        Interval::Closed(self.y.lower_bound, self.y.upper_bound())
    }

    fn d1(&self) -> Interval {
        Interval::Closed(self.x.lower_bound, self.x.upper_bound())
    }

    fn scale(&self, scales: (R64, R64)) -> Self {
        GradedBounds {
            y: self.y.scale(scales.0),
            x: self.x.scale(scales.1),
        }
    }

    fn translate(&self, offsets: (R64, R64)) -> Self {
        GradedBounds {
            y: self.y.translate(offsets.0),
            x: self.x.translate(offsets.1),
        }
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
#[derive(Debug)]
struct ArrangementBounds {
    x_low: f64,
    y_low: f64,
    x_high: f64,
    y_high: f64,
}

#[repr(C)]
struct RivetModuleInvariants {
    computation: *const RivetArrangement,
    error: *const c_char,
    error_length: size_t,
}

#[link(name = "rivet")]
extern "C" {
    fn read_rivet_computation(bytes: *const u8, length: size_t) -> RivetModuleInvariants;
    fn bounds_from_computation(computation: *const RivetArrangement) -> ArrangementBounds;
    fn barcodes_from_computation(
        computation: *const RivetArrangement,
        angles: *const f64,
        offsets: *const f64,
        query_length: size_t,
    ) -> CBarCodesResult;
    fn structure_from_computation(computation: *const RivetArrangement) -> *mut CStructurePoints;

    fn free_barcodes_result(result: CBarCodesResult) -> c_void;
    fn free_rivet_computation_result(result: RivetModuleInvariants);
    fn free_structure_points(points: *mut CStructurePoints);
}

enum RivetArrangement {}

pub struct ModuleInvariants {
    arr: *const RivetArrangement,
}

/// RIVET doesn't use thread locals or locking on arrangements in memory, they're
/// read-only data
unsafe impl Send for ModuleInvariants {}

unsafe impl Sync for ModuleInvariants {}

unsafe impl Send for RivetArrangement {}

unsafe impl Sync for RivetArrangement {}

impl ModuleInvariants {
    pub fn from(data: &[u8]) -> Result<ModuleInvariants, RivetError> {
        parse(data)
    }
}

impl Drop for ModuleInvariants {
    fn drop(&mut self) {
        unsafe {
            free_rivet_computation_result(RivetModuleInvariants {
                computation: self.arr,
                error: ptr::null(),
                error_length: 0,
            });
        }
    }
}

#[derive(Debug, Fail, Clone)]
pub enum RivetErrorKind {
    #[fail(display = "Invalid floating point number in string '{}'", _0)]
    InvalidFloat(String),

    #[fail(display = "RIVET input validation failure: {}", _0)]
    Validation(String),

    #[fail(display = "RIVET I/O failure")]
    Io,

    #[fail(display = "RIVET computation failure: {}", _0)]
    Computation(String),

    #[fail(display = "The bounds {} do not contain the bounds {}", _0, _1)]
    BoundsInsufficient(Bounds, Bounds),
}

pub fn invalid<T>(message: &str) -> Result<T, RivetError> {
    Err(RivetErrorKind::Validation(message.to_owned()))?
}

#[derive(Debug)]
pub struct RivetError {
    inner: Context<RivetErrorKind>
}

impl Fail for RivetError {
    fn cause(&self) -> Option<&Fail> {
        self.inner.cause()
    }

    fn backtrace(&self) -> Option<&Backtrace> {
        self.inner.backtrace()
    }
}

impl Display for RivetError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        Display::fmt(&self.inner, f)
    }
}

impl RivetError {
    pub fn kind(&self) -> RivetErrorKind {
        self.inner.get_context().clone()
    }
}

impl From<RivetErrorKind> for RivetError {
    fn from(kind: RivetErrorKind) -> RivetError {
        RivetError { inner: Context::new(kind) }
    }
}

impl From<Context<RivetErrorKind>> for RivetError {
    fn from(inner: Context<RivetErrorKind>) -> RivetError {
        RivetError { inner: inner }
    }
}

//impl RivetError {
//    pub fn new(message: String, kind: RivetErrorKind) -> RivetError {
//        RivetError { message, kind }
//    }
//    pub fn validation(message: String) -> RivetError { RivetError{message, kind: RivetErrorKind::Validation }}
//}

//impl Display for RivetError {
//    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
//        write!(f, "RIVET error: {}", self.message)
//    }
//}

impl std::convert::From<std::io::Error> for RivetError {
    fn from(error: std::io::Error) -> RivetError {
        RivetError::from(error.context(RivetErrorKind::Io))
    }
}
//
//impl std::convert::From<std::num::ParseFloatError> for RivetError {
//    fn from(error: std::num::ParseFloatError) -> RivetError {
//        RivetError::new(format!("{}", error), RivetErrorKind::Validation)
//    }
//}

pub trait Saveable {
    fn save(&self, writer: &mut Write) -> Result<(), std::io::Error>;
}

//TODO: make streaming version of the inputs for larger datasets
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PointCloudParameters {
    pub cutoff: Option<R64>,
    //TODO: pub distance_label: String,
    pub distance_dimensions: Vec<String>,
    pub appearance_label: Option<String>,
    pub comment: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PointCloud {
    pub parameters: PointCloudParameters,
    pub points: Vec<Vec<R64>>,
    pub appearance: Vec<R64>,
}

impl PointCloud {
    fn calculate_cutoff(&self) -> R64 {
        let mut max = r64(0.0);
        let dim = self.parameters.distance_dimensions.len();
        for row in 0..self.points.len() {
            for col in 0..self.points.len() {
                if row != col {
                    let p_row = &self.points[row];
                    let p_col = &self.points[col];
                    let mut dist = r64(0.0);
                    for i in 0..dim {
                        dist += (p_row[i] - p_col[i]).powi(2);
                    }
                    if dist > max {
                        max = dist;
                    }
                }
            }
        }
        max.sqrt()
    }
}

fn write_comments(writer: &mut Write, comments: &Vec<String>) -> Result<(), std::io::Error> {
    for line in comments {
        writeln!(writer, "# {}", line)?;
    }
    Ok(())
}

impl Saveable for PointCloud {
    fn save(&self, writer: &mut Write) -> Result<(), std::io::Error> {
        write_comments(writer, &self.parameters.comment)?;
        writeln!(writer, "# dimensions: {}", self.parameters.distance_dimensions.join(","))?;
        writeln!(writer, "points")?;
        writeln!(writer, "{}", self.parameters.distance_dimensions.len())?;
        let cutoff = self.parameters.cutoff.unwrap_or(self.calculate_cutoff());
        writeln!(writer, "{}", cutoff)?;
        writeln!(writer, "{}", self.parameters.appearance_label.as_ref().unwrap_or(&"no function".to_string()))?;
        writeln!(writer)?;
        for i in 0..self.points.len() {
            write!(writer,
                   "{}", self.points[i].iter()
                       .map(|x| format!("{}", x))
                       .collect_vec().join(" "))?;
            if !self.appearance.is_empty() {
                writeln!(writer, " {}", self.appearance[i])?;
            } else {
                writeln!(writer)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MetricSpaceParameters {
    pub comment: Vec<String>,
    pub distance_label: String,
    pub appearance_label: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MetricSpace {
    pub parameters: MetricSpaceParameters,
    pub appearance_values: Vec<R64>,
    pub distance_matrix: Vec<Vec<R64>>, //TODO: ndarray?
}

impl MetricSpace {
    fn calculate_cutoff(&self) -> R64 {
        let mut max = r64(0.0);
        for row in 0..self.distance_matrix.len() {
            for col in row + 1..self.distance_matrix[0].len() {
                let dist = self.distance_matrix[row][col];
                if dist > max {
                    max = dist;
                }
            }
        }
        max
    }
}

impl Saveable for MetricSpace {
    fn save(&self, writer: &mut Write) -> Result<(), std::io::Error> {
        write_comments(writer, &self.parameters.comment)?;
        writeln!(writer, "metric")?;
        writeln!(writer, "{}", self.parameters.appearance_label.as_ref().unwrap_or(&"no function".to_string()))?;
        for app in self.appearance_values.iter() {
            write!(writer, "{} ", app)?;
        }
        writeln!(writer)?;
        writeln!(writer, "{}", self.calculate_cutoff())?;
        writeln!(writer, "{}", self.parameters.distance_label)?;
        for row in 0..self.distance_matrix.len() {
            for col in row + 1..self.distance_matrix[0].len() {
                write!(writer, "{} ", self.distance_matrix[row][col])?;
            }
            writeln!(writer)?;
        }
        Ok(())
    }
}


#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum RivetInput {
    Points(PointCloud),
    Metric(MetricSpace),
//    File(Vec<u8>)
//TODO: Bifiltration(Bifiltration)
}

impl RivetInput {
    pub fn parameters(&self) -> RivetInputParameters {
        match self {
            RivetInput::Points(pc) => RivetInputParameters::Points(pc.parameters.clone()),
            RivetInput::Metric(m) => RivetInputParameters::Metric(m.parameters.clone())
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum RivetInputParameters {
    Points(PointCloudParameters),
    Metric(MetricSpaceParameters),
//TODO: Bifiltration(BifiltrationParameters)
}

impl Saveable for RivetInput {
    fn save(&self, writer: &mut Write) -> Result<(), std::io::Error> {
        match self {
            RivetInput::Points(points) => points.save(writer),
            RivetInput::Metric(points) => points.save(writer)
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ComputationParameters {
    pub param1_bins: u32,
    pub appearance_bins: u32,
    //TODO: param1_scale: R64,
//TODO: appearance_scale: R64,
    pub homology_dimension: u32,
    pub threads: usize,
}

pub fn compute(input: &RivetInput, parameters: &ComputationParameters) -> Result<Vec<u8>, RivetError> {
    let dir = TempDir::new("rivet-").context(RivetErrorKind::Io)?;
    let input_path = dir.path().join("rivet-input.txt");
    let output_path = dir.path().join("rivet-output.rivet");
    {
        let mut input_file = File::create(&input_path).context(RivetErrorKind::Io)?;
        input.save(&mut input_file).context(RivetErrorKind::Io)?;
    }
    {
        let data = std::fs::read(&input_path).context(RivetErrorKind::Io)?;
        let check = String::from_utf8_lossy(data.as_slice());
        info!("Generated this file content: {}", check);
    }
//TODO: RIVET C api needs a way to call RIVET to precompute a file
//For now just punt, adding this to RIVET C API is too much trouble at the moment
    let mut command = Command::new("rivet_console");

    command.arg(input_path)
        .arg(&output_path)
        .arg("-H")
        .arg(format!("{}", parameters.homology_dimension))
        .arg("--num_threads")
        .arg(format!("{}", std::cmp::max(parameters.threads, 1)))
        .arg("-x")
        .arg(format!("{}", parameters.param1_bins))
        .arg("-y")
        .arg(format!("{}", parameters.appearance_bins));
    info!("Calling rivet_console: {:?}", command);
    let output = command.output().context(RivetErrorKind::Io)?;
    info!("Console exited");
    let result = if output.status.success() {
        info!("Success - reading RIVET output file back into memory");
        let bytes = std::fs::read(&output_path).context(RivetErrorKind::Io)?;
        info!("Success - returning {} bytes", bytes.len());
        Ok(bytes)
    } else {
        info!("Something went wrong");
        let message = String::from_utf8_lossy(&output.stderr).to_string();
        info!("Failure: {}", &message);
        Err(RivetErrorKind::Computation(message))
    }?;
    dir.close().context(RivetErrorKind::Io)?;
    Ok(result)
}

pub fn parse(bytes: &[u8]) -> Result<ModuleInvariants, RivetError> {
    if bytes.len() == 0 {
        invalid("Byte array must have non-zero length")?
    } else {
        let rivet_comp = unsafe { read_rivet_computation(bytes.as_ptr(), bytes.len()) };
        if !rivet_comp.error.is_null() {
            let message = unsafe { CStr::from_ptr(rivet_comp.error) }
                .to_owned()
                .to_str().context(RivetErrorKind::Io)?.to_string();
            unsafe {
                free_rivet_computation_result(rivet_comp);
            }
            Err(RivetErrorKind::Computation(format!("Error while reading computation: {}", message)))?
        } else {
            Ok(ModuleInvariants {
                arr: rivet_comp.computation,
            })
        }
    }
}

pub fn bounds(computation: &ModuleInvariants) -> Bounds {
    unsafe {
        let res = bounds_from_computation(computation.arr);

        Bounds::new(
            res.y_low,
            res.y_high,
            res.x_low,
            res.x_high,
        ).expect(&format!("RIVET returned invalid bounds {:#?}", &res))
    }
}

pub fn barcodes(
    computation: &ModuleInvariants,
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
            Err(RivetErrorKind::Computation(message))?
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

pub fn structure(computation: &ModuleInvariants) -> BettiStructure {
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

impl ModuleInvariants {
    pub fn bounds(self: &ModuleInvariants) -> Bounds {
        return bounds(self);
    }
}

impl Bounds {
    pub fn common_bounds(self: &Bounds, other: &Bounds) -> Bounds {
        let d0 = self.d0().union(&other.d0());
        let d1 = self.d1().union(&other.d1());
        Bounds {
            rect: Rectangle { d0, d1 },
        }
    }
}

pub fn parse_input<R: Read>(read: R) -> Result<RivetInput, RivetError> {
    let reader = BufReader::new(read);
    let mut comment = vec![];
    enum FileType {
        PointCloud,
        Metric,
//        Bifiltration
    };
    let file_type;
    let mut lines = reader.lines();
    loop {
        match lines.next() {
            None => invalid("Reached end of input file without determining file type")?,
            Some(line) => {
                let line = line.context(RivetErrorKind::Io)?;
                let clean = line.trim();
                if clean.starts_with("#") {
                    comment.push(clean[1..].to_owned());
                } else {
                    file_type = Some(match clean {
                        "points" => Ok(FileType::PointCloud),
                        "metric" => Ok(FileType::Metric),
                        other => invalid(&format!("Unrecognized file type '{}'", other))
                    }?);
                    break;
                }
            }
        }
    }
    let mut skip_comments =
        lines.map_results(|x| x.trim().to_string())
            .filter(|x| {
                match x {
                    Err(_) => true,
                    Ok(line) => !(line == "" || line.starts_with("#"))
                }
            });
    match file_type {
        None => invalid("No file type found")?,
        Some(ft) => match ft {
            FileType::PointCloud => parse_pointcloud(&mut skip_comments, comment),
            FileType::Metric => parse_metric(&mut skip_comments, comment)
        }
    }
}

//fn eof_error(msg: &str) -> RivetError {
//    RivetError::new(msg.to_string(), RivetErrorKind::Validation)
//}
fn line_or(val: Option<Result<String, std::io::Error>>, message: &str) -> Result<String, RivetError> {
    match val {
        None => invalid(message),
        Some(Ok(v)) => Ok(v),
        Some(Err(e)) => Err(e).context(RivetErrorKind::Io)?
    }
}

fn parse_r64(s: &str) -> Result<R64, RivetError> {
    let float = s.parse::<f64>().context(RivetErrorKind::InvalidFloat(s.to_owned()))?;
    Ok(r64(float))
}

fn parse_pointcloud(buf: &mut Iterator<Item=Result<String, std::io::Error>>, comment: Vec<String>) -> Result<RivetInput, RivetError> {
    let point_dim = line_or(buf.next(), "No dimension line!")?;
    let cutoff = parse_r64(&line_or(buf.next(), "No max distance!")?)?;
    let appearance_label = line_or(buf.next(), "No label!")?;
    let mut points = vec![];
    let mut appearance = vec![];
    loop {
        match buf.next() {
            None => break,
            Some(line) => {
                let line = line.context(RivetErrorKind::Io)?;
                let could_be_numbers = line.split_whitespace().map(|x| x.parse::<f64>()).collect_vec();
                let (f64s, errors): (Vec<Result<f64, _>>, _) = could_be_numbers.into_iter().partition(|x| x.is_ok());
                for err in errors {
                    err.context(RivetErrorKind::InvalidFloat(line.clone()))?;
                }
                let numbers = f64s.into_iter().map(|x| r64(x.unwrap())).collect_vec();
                points.push(numbers[0..numbers.len() - 1].to_vec());
                appearance.push(numbers[numbers.len() - 1])
            }
        }
    }
    //TODO: not always the case of course! Fix!
    let distance_dimensions = vec!["x", "y", "z"].into_iter().map(|x| x.to_string()).collect_vec();
    Ok(RivetInput::Points(
        PointCloud {
            parameters: PointCloudParameters {
                cutoff: Some(cutoff),
                distance_dimensions,
                appearance_label: Some(appearance_label),
                comment,
            },
            points,
            appearance,
        }
    ))
}

fn parse_metric(_buf: &mut Iterator<Item=Result<String, std::io::Error>>, _comment: Vec<String>) -> Result<RivetInput, RivetError> {
    unimplemented!()
}
