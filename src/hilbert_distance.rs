use itertools::Itertools;
use ndarray::prelude::*;
use noisy_float::prelude::*;
use num_rational::Rational64;
use crate::rivet::{Bounds, BettiStructure, invalid, RivetError, RivetErrorKind};
use std::hash::Hash;
use std::hash::Hasher;
use std::mem;
use std::ops;
use log::warn;

fn rational_to_r64(ratio: &Rational64) -> R64 {
    let num: f64 = *ratio.numer() as f64;
    let denom = *ratio.denom() as f64;
    r64(num / denom)
}

enum DimensionQueryResult {
    Low,
    High,
    In(usize),
}

// Needed to use a HashSet with floats. https://stackoverflow.com/a/39639200/224186
fn integer_decode(val: f64) -> (u64, i16, i8) {
    let bits: u64 = unsafe { mem::transmute(val) };
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}

impl Hash for HashableR64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let (mantissa, exponent, sign) = integer_decode(self.value.raw());
        state.write_u64(mantissa);
        state.write_i16(exponent);
        state.write_i8(sign);
        state.finish();
    }
}

#[derive(Eq, PartialEq, Clone, Copy)]
struct HashableR64 {
    value: R64,
}

#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub struct Dimension {
    /// The lower bound of this dimension
    lower_bound: R64,
    /// The upper bound of this dimension
    upper_bounds: Vec<R64>,
    /// A collection of all the indices (in the backing matrix of a SplitMat) that map to
    /// each of the upper bounds (i.e. upper_indexes is always the same length as upper_bounds).
    upper_indexes: Vec<Option<usize>>,
}

#[derive(Debug, Clone)]
pub enum Interval {
    Open(R64, R64),
    Closed(R64, R64),
    OpenClosed(R64, R64),
    ClosedOpen(R64, R64),
}

#[derive(Debug, Clone)]
pub enum OpenClosed {
    Open,
    Closed,
}

impl Interval {
    fn new(
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

    fn ends(&self) -> (R64, R64) {
        match self {
            Interval::Open(start, end) => (*start, *end),
            Interval::Closed(start, end) => (*start, *end),
            Interval::OpenClosed(start, end) => (*start, *end),
            Interval::ClosedOpen(start, end) => (*start, *end),
        }
    }
    fn end_types(&self) -> (OpenClosed, OpenClosed) {
        match self {
            Interval::Open(_, _) => (OpenClosed::Open, OpenClosed::Open),
            Interval::Closed(_, _) => (OpenClosed::Closed, OpenClosed::Closed),
            Interval::OpenClosed(_, _) => (OpenClosed::Open, OpenClosed::Closed),
            Interval::ClosedOpen(_, _) => (OpenClosed::Closed, OpenClosed::Open),
        }
    }

    fn intersection(&self, other: &Interval) -> Option<Interval> {
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
        //        println!("New verify {:?}", &dim);
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

    fn reset(&self) -> Dimension {
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

    fn lengths(&self) -> Vec<R64> {
        let mut bounds = vec![self.lower_bound];
        bounds.extend_from_slice(&self.upper_bounds);
        let mut lengths = Vec::<R64>::with_capacity(bounds.len() - 1);
        for i in 1..bounds.len() {
            lengths.push(bounds[i] - bounds[i - 1]);
        }
        lengths
    }

    fn intervals(&self) -> Vec<Interval> {
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

    fn len(&self) -> usize {
        self.upper_bounds.len()
    }

    fn translate(&self, increment: R64) -> Dimension {
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

    fn scale(&self, factor: R64) -> Dimension {
        Dimension {
            lower_bound: self.lower_bound,
            upper_bounds: self.upper_bounds.iter().map(|x| *x * factor).collect_vec(),
            upper_indexes: self.upper_indexes.clone(),
        }
    }

    fn add_bound(&mut self, bound: R64) {
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

    fn is_bound(&self, bound: R64) -> bool {
        bound == self.lower_bound || self.upper_bounds.contains(&bound)
    }

    fn index(&self, value: R64) -> DimensionQueryResult {
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

    fn merge(&self, other: &Dimension) -> Dimension {
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

    fn verify(&self) {
        let mut test = Vec::with_capacity(self.upper_bounds.len());
        for b in self.upper_bounds.iter() {
            assert!(!test.contains(&b));
            test.push(&b);
        }
        assert_eq!(test.len(), self.upper_bounds.len());
        assert!(is_sorted(&test));
    }
}

fn is_sorted<T: Ord>(thing: &[T]) -> bool {
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

/// A virtual matrix with both discrete and real-valued indices, and integer values. A SplitMat
/// can be subdivided at any real-valued point on either axis.
#[derive(Eq, Default, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub struct SplitMat {
    mat: Array2<i32>,
    dimensions: Vec<Dimension>,
}

pub enum SampleType {
    MEAN,
    MIN,
    MAX,
}

#[derive(Debug, Clone)]
pub struct Rectangle {
    d0: Interval,
    d1: Interval,
}

impl Rectangle {
    pub fn new(d0: Interval, d1: Interval) -> Rectangle {
        Rectangle { d0, d1 }
    }

    pub fn intersection(&self, other: &Rectangle) -> Option<Rectangle> {
        let d0 = self.d0.intersection(&other.d0)?;
        let d1 = self.d1.intersection(&other.d1)?;
        Some(Rectangle { d0, d1 })
    }

    pub fn area(&self) -> R64 {
        let (start0, end0) = self.d0.ends();
        let (start1, end1) = self.d1.ends();
        (end0 - start0) * (end1 - start1)
    }
}

//pub struct Rectangle {
//    pub position: (R64, R64),
//    pub size: (R64, R64)
//}
//
//impl Rectangle {
//    pub fn from_points(position: (R64, R64), end_position: (R64, R64)) -> Option<Rectangle> {
//        let (s1, s2) = position;
//        let (e1, e2) = end_position;
//        if s2 < s1 || e2 < e1 {
//            None
//        } else {
//            Some(
//                Rectangle {
//                    position,
//                    size: (s2 - s1, e2 - e1)
//                }
//            )
//        }
//    }
//}
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
        self.rectangle.intersection(&other).map(|rect| Region {
            rectangle: rect,
            value: self.value,
        })
    }
}

impl SplitMat {
    pub fn constant(constant: i32, dimensions: Vec<Dimension>) -> SplitMat {
        assert_eq!(dimensions.len(), 2, "Requires exactly 2 dimensions");
        SplitMat::new(
            Array2::from_elem((dimensions[0].len(), dimensions[1].len()), constant),
            dimensions,
        )
    }

    pub fn new(mat: Array2<i32>, dimensions: Vec<Dimension>) -> SplitMat {
        {
            let shape = mat.shape();

            assert_eq!(
                dimensions.len(),
                shape.len(),
                "number of dimensions does not equal array shape length"
            );
            for dimension_number in 0..shape.len() {
                for upper_index in dimensions[dimension_number]
                    .upper_indexes
                    .iter()
                    .filter(|x| x.is_some())
                    .map(|i| i.unwrap())
                    {
                        assert!(upper_index < shape[dimension_number],
                                format!("In dimension {}, dimension index {} is too big for matrix dimension {}",
                                        dimension_number, upper_index, mat.shape()[dimension_number]))
                    }
            }
        }
        SplitMat { mat, dimensions }
    }

    fn shape(&self) -> (usize, usize) {
        (
            self.dimensions[0].upper_bounds.len(),
            self.dimensions[1].upper_bounds.len(),
        )
    }

    fn row(&self, length: R64) -> DimensionQueryResult {
        self.dimensions[0].index(length)
    }

    fn col(&self, length: R64) -> DimensionQueryResult {
        self.dimensions[1].index(length)
    }

    fn index(&self, coords: (R64, R64)) -> (DimensionQueryResult, DimensionQueryResult) {
        (
            self.dimensions[0].index(coords.0),
            self.dimensions[1].index(coords.1),
        )
    }

    fn add_row(&mut self, first_length: R64) {
        self.dimensions[0].add_bound(first_length);
    }

    fn add_col(&mut self, first_length: R64) {
        self.dimensions[1].add_bound(first_length);
    }

    fn scale(&self, factors: &[R64]) -> SplitMat {
        assert_eq!(factors.len(), self.dimensions.len());
        return SplitMat {
            mat: self.mat.clone(),
            dimensions: self
                .dimensions
                .iter()
                .zip(factors.iter())
                .map(|(d, f)| d.scale(*f))
                .collect_vec(),
        };
    }

    fn translate(&self, offsets: &[R64]) -> SplitMat {
        assert_eq!(offsets.len(), self.dimensions.len());
        return SplitMat {
            mat: self.mat.clone(),
            dimensions: self
                .dimensions
                .iter()
                .zip(offsets.iter())
                .map(|(d, f)| d.translate(*f))
                .collect_vec(),
        };
    }

    fn merge(&self, rhs: &SplitMat) -> SplitMat {
        let left_dims = self
            .dimensions
            .iter()
            .zip(rhs.dimensions.iter())
            .map(|(x, y)| x.merge(y))
            .collect_vec();
        SplitMat::new(self.mat.clone(), left_dims)
    }

    fn value(&self, point: (R64, R64)) -> Option<i32> {
        match self.index(point) {
            (DimensionQueryResult::In(d0), DimensionQueryResult::In(d1)) => Some(
                self.mat[(
                    self.dimensions[0].upper_indexes[d0].unwrap(),
                    self.dimensions[1].upper_indexes[d1].unwrap(),
                )],
            ),
            _ => None,
        }
    }

    fn regions(&self) -> Array2<Region> {
        let d0_ints = self.dimensions[0].intervals();
        let d1_ints = self.dimensions[1].intervals();
        let results =
            Array2::from_shape_fn((d0_ints.len(), d1_ints.len()), |(row_idx, col_idx)| {
                Region {
                    rectangle: Rectangle {
                        d0: d0_ints[row_idx].clone(),
                        d1: d1_ints[col_idx].clone(),
                    },
                    value: Some(r64(self.mat[(
                        self.dimensions[0].upper_indexes[row_idx].unwrap(),
                        self.dimensions[1].upper_indexes[col_idx].unwrap(),
                    )] as f64)),
                }
            });
        results
    }

    fn bounds(&self) -> Bounds {
        Bounds {
            x_low: self.dimensions[1].lower_bound.raw(),
            y_low: self.dimensions[0].lower_bound.raw(),
            x_high: self.dimensions[1].upper_bound().raw(),
            y_high: self.dimensions[0].upper_bound().raw(),
        }
    }



    /// Returns a matrix with the same shape as the given template (the template should contain the
    /// dimensions of self for meaningful results), with values that are the weighted
    /// averages (or max, or min) of any values in self that fall in a cell of the template.
    pub fn sample(&self, template: &SplitMat, sample_type: SampleType) -> Result<Array2<R64>, RivetError> {
        if !template.bounds().contains(&self.bounds()) {
            Err(RivetErrorKind::Validation(format!("Bounds of template {:#?} must include bounds of sample {:#?}", template.bounds(), self.bounds()).to_owned()))?;
        }
        let merged = self.merge(template).expand();
        let template_row_intervals = template.dimensions[0].intervals();
        let template_col_intervals = template.dimensions[1].intervals();
        // To avoid a many-many comparison between all the regions in merged and all the regions in template,
        // we first figure out which indices in merged might have regions that overlap with the regions
        // for each row and column in template
        let mut min_merged_rows_for_template_rows = vec![0; template.dimensions[0].len()];
        let mut min_merged_cols_for_template_cols = vec![0; template.dimensions[1].len()];
        for row in 1..template.dimensions[0].len() {
            let row_interval = &template_row_intervals[row];
            let (row_start, _) = row_interval.ends();
            while merged.dimensions[0].upper_bounds[min_merged_rows_for_template_rows[row]]
                <= row_start
                {
                    min_merged_rows_for_template_rows[row] += 1
                }
        }
        for col in 1..template.dimensions[1].len() {
            let col_interval = &template_col_intervals[col];
            let (col_start, _) = col_interval.ends();
            while merged.dimensions[1].upper_bounds[min_merged_cols_for_template_cols[col]]
                <= col_start
                {
                    min_merged_cols_for_template_cols[col] += 1
                }
        }
        let mut max_merged_rows_for_template_rows = min_merged_rows_for_template_rows.clone();
        let mut max_merged_cols_for_template_cols = min_merged_cols_for_template_cols.clone();
        for row in 0..template.dimensions[0].len() {
            let row_interval = &template_row_intervals[row];
            let (_, row_end) = row_interval.ends();
            while merged.dimensions[0].upper_bounds[max_merged_rows_for_template_rows[row]]
                < row_end
                {
                    max_merged_rows_for_template_rows[row] += 1
                }
        }
        for col in 0..template.dimensions[1].len() {
            let col_interval = &template_col_intervals[col];
            let (_k, col_end) = col_interval.ends();
            while merged.dimensions[1].upper_bounds[max_merged_cols_for_template_cols[col]]
                < col_end
                {
                    max_merged_cols_for_template_cols[col] += 1
                }
        }

        // Now that we know which regions we need to clip, the actual sampling is fairly straightforward:
        let merged_regions = merged.regions();
        let result = Array2::from_shape_fn(template.shape(), |(row, col)| {
            let cell_regions = merged_regions
                .slice(s![
                    min_merged_rows_for_template_rows[row]
                        ..(max_merged_rows_for_template_rows[row] + 1),
                    min_merged_cols_for_template_cols[col]
                        ..(max_merged_cols_for_template_cols[col] + 1)
                ])
                .into_iter()
                .collect_vec();
            let template_rect = Rectangle::new(
                template_row_intervals[row].clone(),
                template_col_intervals[col].clone(),
            );
            let clipped = cell_regions
                .iter()
                .filter_map(|x| x.intersection(&template_rect));
            match sample_type {
                SampleType::MIN => clipped
                    .filter_map(|x| x.value)
                    .min()
                    .or(Some(r64(0.0)))
                    .unwrap(),
                SampleType::MAX => clipped
                    .filter_map(|x| x.value)
                    .max()
                    .or(Some(r64(0.0)))
                    .unwrap(),
                SampleType::MEAN => {
                    let sum: R64 = clipped
                        .filter_map(|x| x.value.map(|v: R64| v * x.rectangle.area()))
                        .sum();
                    sum / template_rect.area()
                }
            }
        });
        Ok(result)
    }

    fn expand(&self) -> SplitMat {
        let mut vec = Vec::with_capacity(self.dimensions.iter().map(|x| x.len()).product());
        for row in 0..self.dimensions[0].len() {
            for col in 0..self.dimensions[1].len() {
                if self.dimensions[0].upper_indexes[row] == None
                    || self.dimensions[1].upper_indexes[col] == None
                    {
                        vec.push(0);
                    } else {
                    vec.push(
                        self.mat[[
                            self.dimensions[0].upper_indexes[row].unwrap(),
                            self.dimensions[1].upper_indexes[col].unwrap(),
                        ]],
                    );
                }
            }
        }
        assert!(self.dimensions[0].len() > 0);
        assert!(self.dimensions[1].len() > 0);
        SplitMat::new(
            Array2::from_shape_vec((self.dimensions[0].len(), self.dimensions[1].len()), vec)
                .unwrap(),
            self.dimensions.iter().map(|d| d.reset()).collect_vec(),
        )
    }

    fn weighted_difference(&self, other: &SplitMat) -> Array2<R64> {
        let split_diff = self - other;
        let mut diff = split_diff.mat.map(|&x| r64(x as f64));
        let dim_lengths = split_diff
            .dimensions
            .iter()
            .map(|d| d.lengths())
            .collect_vec();
        for (row, &r_weight) in dim_lengths[0].iter().enumerate() {
            for (col, &c_weight) in dim_lengths[1].iter().enumerate() {
                diff[[row, col]] *= r_weight * c_weight;
            }
        }
        diff
    }

    pub fn distance(&self, other: &SplitMat) -> R64 {
        let diff = self.weighted_difference(other);
        let mat: Array2<R64> = &diff * &diff;
        let combined: f64 = mat.iter().map(|x| x.raw()).sum();
        r64(combined.sqrt())
    }

    pub fn betti_to_splitmat(betti: &BettiStructure) -> Result<SplitMat, RivetError> {
        let xs = betti.x_grades.iter().map(rational_to_r64).collect_vec();
        let ys = betti.y_grades.iter().map(rational_to_r64).collect_vec();
        if xs.len() == 0 {
            invalid("No x grades")?;
        }
        if ys.len() == 0 {
            invalid("No y grades")?;
        }
        if !is_sorted(&xs) {
            invalid("x grades from RIVET not sorted")?;
        }
        if !is_sorted(&ys) {
            invalid("y grades from RIVET not sorted")?;
        }
        //        let dimensions = vec![
        //            Dimension::new(ys[0], ys[1..].to_owned()),
        //            Dimension::new(xs[0], xs[1..].to_owned())
        //        ];

        let mut y_lengths = Vec::with_capacity(ys.len() - 1);
        for i in 1..ys.len() {
            y_lengths.push(ys[i] - ys[i - 1]);
        }
        let mut x_lengths = Vec::with_capacity(xs.len() - 1);
        for i in 1..xs.len() {
            x_lengths.push(xs[i] - xs[i - 1]);
        }
        let x_nonzero_lengths = x_lengths.iter().filter(|&&x| x != r64(0.)).count();
        let y_nonzero_lengths = y_lengths.iter().filter(|&&y| y != r64(0.)).count();
        if x_nonzero_lengths == 0 {
            invalid("No nonzero x lengths")?;
        }
        if y_nonzero_lengths == 0 {
            invalid("No nonzero y lengths")?;
        }
        let mut unique_ys = Vec::with_capacity(ys.len());
        for y in ys {
            if !unique_ys.contains(&y) {
                unique_ys.push(y);
            }
        }
        let mut unique_xs = Vec::with_capacity(xs.len());
        for x in xs {
            if !unique_xs.contains(&x) {
                unique_xs.push(x);
            }
        }
        let dimensions = vec![
            Dimension::new(unique_ys[0], unique_ys[1..].to_owned())?,
            Dimension::new(unique_xs[0], unique_xs[1..].to_owned())?,
        ];
        assert_eq!(dimensions[0].len(), y_nonzero_lengths);
        assert_eq!(dimensions[1].len(), x_nonzero_lengths);
        let mut mat = Array2::<i32>::zeros((y_nonzero_lengths, x_nonzero_lengths));
        for point in betti.points.iter() {
            let mut x = point.x as isize;
            let mut y = point.y as isize;
            for row in 0..(y_lengths.len()) {
                if y_lengths[row] == r64(0.) {
                    if row <= point.y as usize {
                        y -= 1;
                    }
                }
            }
            for col in 0..(x_lengths.len()) {
                if x_lengths[col] == r64(0.) {
                    if col <= point.x as usize {
                        x -= 1;
                    }
                }
            }
            //            println!("x = {}, y = {}", x, y);
            let mut slice = mat.slice_mut(s![y.., x..]);
            slice += point.betti_0 as i32;
            slice -= point.betti_1 as i32;
            slice -= point.betti_2 as i32;
            //            println!("Created point from structurepoint");
        }

        mat.map_inplace(|x| {
            if x < &mut 0 {
                *x = 0
            }
        });
        Ok(SplitMat { mat, dimensions })
    }
}

impl<'a, 'b> ops::Add<&'b SplitMat> for &'a SplitMat {
    type Output = SplitMat;

    fn add(self, rhs: &SplitMat) -> Self::Output {
        let left = self.merge(rhs).expand();
        let right = rhs.merge(self).expand();
        if left.mat.shape() != right.mat.shape() {
            error!(
                "Merged splitmat shapes don't match!
            \nOriginal left: {:?}
            \nOriginal right: {:?}
            \n expanded shapes: L: {:?} R: {:?}",
                self,
                rhs,
                &left.mat.shape(),
                &right.mat.shape()
            );
            panic!("Matrix shapes don't match after expansion");
        }
        SplitMat {
            mat: left.mat + right.mat,
            dimensions: left.dimensions.clone(),
        }
    }
}

impl<'a> ops::Add<SplitMat> for &'a SplitMat {
    type Output = SplitMat;

    fn add(self, rhs: SplitMat) -> Self::Output {
        let left = self.merge(&rhs).expand();
        let right = rhs.merge(self).expand();
        if left.mat.shape() != right.mat.shape() {
            println!(
                "Merged splitmat shapes don't match!
            \nOriginal left: {:?}
            \nOriginal right: {:?}
            \n expanded shapes: L: {:?} R: {:?}",
                &self,
                &rhs,
                &left.mat.shape(),
                &right.mat.shape()
            );
            println!("Left expanded dimensions: {:?}", &left.dimensions);
            println!("Right expanded dimensions: {:?}", &right.dimensions);
            panic!("Matrix shapes don't match after expansion");
        }
        SplitMat {
            mat: left.mat + right.mat,
            dimensions: left.dimensions.clone(),
        }
    }
}

impl ops::Add for SplitMat {
    type Output = SplitMat;

    fn add(self, rhs: SplitMat) -> Self::Output {
        let left = self.merge(&rhs).expand();
        let right = rhs.merge(&self).expand();

        SplitMat {
            mat: left.mat + right.mat,
            dimensions: left.dimensions.clone(),
        }
    }
}

impl ops::Neg for SplitMat {
    type Output = SplitMat;

    fn neg(self) -> Self::Output {
        SplitMat {
            mat: -self.mat,
            dimensions: self.dimensions,
        }
    }
}

impl<'a> ops::Neg for &'a SplitMat {
    type Output = SplitMat;

    fn neg(self) -> Self::Output {
        SplitMat {
            mat: -(self.mat.clone()),
            dimensions: self.dimensions.clone(),
        }
    }
}

impl<'a, 'b> ops::Sub<&'b SplitMat> for &'a SplitMat {
    type Output = SplitMat;

    fn sub(self, rhs: &SplitMat) -> Self::Output {
        self + (-rhs)
    }
}

/// Creates a fingerprint for the given `structure`, given `bounds` that describe the
/// domain of valid parameter values (and which must include the bounds of `structure`),
/// and a `granularity` that determine how heavily to weight
/// each parameter (by subdividing the structure more finely). The granularity of each parameter
/// determines how many cells there will be in the resulting fingerprint.
///
/// For example:
///
/// Granularity         Effect
/// (0, 0)              Error
/// (1, 0) or (0, 1)    Single element vector with highest (or mean, or min) Betti number for the
///                     whole structure
/// (1, 5)              Five element vector that generalizes over the whole range of param 1, but
///                     divides the structure into 5 pieces along param2, effectively making param 2
///                     5 times more important than param 1 and losing all the information in param 1.
/// (10, 10)            A 100-element vector that weights param1 and param2 evenly
pub fn fingerprint(structure: &BettiStructure,
                   bounds: &Bounds,
                   granularity: (usize, usize)) -> Result<Vec<f64>, RivetError> {
    let (y_bins, x_bins) = granularity;

    let bounds = bounds.valid()?;

    //First, generate a splitmat from the structure
    let matrix = SplitMat::betti_to_splitmat(structure)?;
    if !bounds.contains(&matrix.bounds()) {
        Err(RivetErrorKind::Validation(
            format!("Bounds {:#?} must enclose structure bounds {:#?}", bounds, matrix.bounds())
                .to_owned()))?;
    }

    //Normalize it so its bounds are between 0 and 1 in both parameters
    //TODO: change scale and translate to take tuples instead of vectors
    let translated = matrix.translate(&vec![
        -matrix.dimensions[0].lower_bound,
        -matrix.dimensions[1].lower_bound
    ]);
    let scaled = translated.scale(&vec![
        r64(1.0) / (bounds.y_high - bounds.y_low),
        r64(1.0) / (bounds.x_high - bounds.x_low),
    ]);

    //Now build a template with the right granularity for sampling
    let y_dim = Dimension::from_f64s(0.0,
                                     &Array::linspace(0.0, 1.0, y_bins).to_vec()[1..])?;
    let x_dim = Dimension::from_f64s(0.0,
                                     &Array::linspace(0.0, 1.0, x_bins).to_vec()[1..])?;
    let template =
        SplitMat::constant(0, vec![y_dim, x_dim]);

    //Take our sample and convert it to a vector

    let sample = scaled.sample(&template, SampleType::MEAN).unwrap();
    let shape = sample.shape();
    let mut vector = vec![0.0; shape[0] * shape[1]];
    let mut pos = 0;
    //TODO: surely there's a method in ndarray for this?
    for row in 0..shape[0] {
        for col in 0..shape[1] {
            vector[pos] = sample[(row, col)].raw();
            pos += 1;
        }
    }
    Ok(vector)
}

#[cfg(test)]
mod tests {
    use crate::hilbert_distance::Dimension;
    use crate::hilbert_distance::SampleType;
    use crate::hilbert_distance::SplitMat;
    use ndarray::arr2;
    use noisy_float::types::r64;
    use crate::hilbert_distance::fingerprint;

    #[test]
    fn expand() {
        let mut split = SplitMat::new(
            arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            vec![
                Dimension::from_f64s(0., &vec![0.25, 0.5, 1.0]).unwrap(),
                Dimension::from_f64s(0., &vec![0.25, 0.5, 1.0]).unwrap(),
            ],
        );

        split.add_row(r64(0.1));
        split.add_row(r64(0.05));
        split.add_row(r64(0.8));
        split.add_row(r64(1.5));
        assert_eq!(
            split.dimensions[0].upper_bounds,
            Dimension::from_f64s(0., &vec![0.05, 0.1, 0.25, 0.5, 0.8, 1.0, 1.5]).unwrap().upper_bounds
        );
        split.add_col(r64(0.4));
        assert_eq!(
            split.dimensions[1].upper_bounds,
            Dimension::from_f64s(0., &vec![0.25, 0.4, 0.5, 1.0]).unwrap().upper_bounds
        );

        let expanded = split.expand();

        assert_eq!(
            expanded.mat,
            arr2(&[
                [1, 2, 2, 3],
                [1, 2, 2, 3],
                [1, 2, 2, 3],
                [4, 5, 5, 6],
                [7, 8, 8, 9],
                [7, 8, 8, 9],
                [0, 0, 0, 0]
            ])
        );
    }

    #[test]
    fn distance() {
        let split = SplitMat::new(
            arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            vec![
                Dimension::from_f64s(0., &vec![0.25, 0.5, 1.0]).unwrap(),
                Dimension::from_f64s(0., &vec![0.25, 0.5, 1.0]).unwrap(),
            ],
        );

        let mut split2 = split.clone();

        assert_eq!(split2.distance(&split), r64(0.));
        assert_eq!(split2.distance(&split), split.distance(&split2));

        split2.mat[[0, 0]] = 3;
        assert_eq!(split2.distance(&split), r64(2. * (0.25 * 0.25)));
        assert_eq!(split2.distance(&split), split.distance(&split2));
    }

    #[test]
    fn distance_overlapping() {
        let split = SplitMat::new(
            arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            vec![
                Dimension::from_f64s(0., &vec![0.25, 0.5, 1.0]).unwrap(),
                Dimension::from_f64s(0., &vec![0.25, 0.5, 1.0]).unwrap(),
            ],
        );

        let split2 = SplitMat::new(
            arr2(&[[1, 2], [3, 4]]),
            vec![
                Dimension::from_f64s(0.2, &vec![0.25, 0.5]).unwrap(),
                Dimension::from_f64s(0.2, &vec![0.25, 0.5]).unwrap(),
            ],
        );

        println!("split.merge(&split2).expand()");
        println!("{:?}", split.merge(&split2).expand());

        println!("split2.merge(&split).expand()");
        println!("{:?}", split2.merge(&split).expand());

        let left_diff = split.merge(&split2).expand().mat - split2.merge(&split).expand().mat;
        let right_diff = split2.merge(&split).expand().mat - split.merge(&split2).expand().mat;
        assert_eq!(left_diff, -right_diff, "diffs");

        let left_weighted = split.weighted_difference(&split2);
        let right_weighted = split2.weighted_difference(&split);
        assert_eq!(
            left_weighted.clone(),
            -(right_weighted.clone()),
            "weighted diffs"
        );

        let left_squared = &left_weighted * &left_weighted;
        let right_squared = &right_weighted * &right_weighted;
        assert_eq!(&left_squared, &right_squared, "squared");

        let left_sum: f64 = left_squared.iter().map(|x| x.raw()).sum();
        let right_sum: f64 = right_squared.iter().map(|x| x.raw()).sum();
        assert_eq!(left_sum, right_sum, "sum");

        assert_eq!(left_sum.sqrt(), right_sum.sqrt(), "sqrt");

        assert_eq!(
            split2.distance(&split),
            r64(2.701316808521355),
            "exact value"
        );
        assert_eq!(split2.distance(&split), split.distance(&split2), "symmetry");
    }

    #[test]
    fn test_sample_min() {
        let split = SplitMat::new(
            arr2(&[[1, 2, 3], [2, 4, 6]]),
            vec![
                Dimension::from_f64s(0., &vec![1.0, 2.0]).unwrap(),
                Dimension::from_f64s(0., &vec![1.0, 2.0, 3.0]).unwrap(),
            ],
        );

        let template = SplitMat::constant(
            0,
            vec![
                Dimension::from_f64s(-1., &vec![0.5, 1.0, 3.0]).unwrap(),
                Dimension::from_f64s(-1., &vec![0.5, 2.5, 4.0]).unwrap(),
            ],
        );
        let min = split.sample(&template, SampleType::MIN).unwrap();
        assert_eq!(
            min,
            arr2(&[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).map(|&x| r64(x))
        );
    }

    #[test]
    fn test_sample_max() {
        let split = SplitMat::new(
            arr2(&[[1, 2, 3], [2, 4, 6]]),
            vec![
                Dimension::from_f64s(0., &vec![1.0, 2.0]).unwrap(),
                Dimension::from_f64s(0., &vec![1.0, 2.0, 3.0]).unwrap(),
            ],
        );

        let template = SplitMat::constant(
            0,
            vec![
                Dimension::from_f64s(-1., &vec![0.5, 1.0, 3.0]).unwrap(),
                Dimension::from_f64s(-1., &vec![0.5, 2.5, 4.0]).unwrap(),
            ],
        );
        let max = split.sample(&template, SampleType::MAX).unwrap();
        assert_eq!(
            max,
            arr2(&[[1.0, 3.0, 3.0], [1.0, 3.0, 3.0], [2.0, 6.0, 6.0]]).map(|&x| r64(x))
        );
    }

    #[test]
    fn test_sample_mean() {
        let split = SplitMat::new(
            arr2(&[[1, 2, 3], [2, 4, 6]]),
            vec![
                Dimension::from_f64s(0., &vec![1.0, 2.0]).unwrap(),
                Dimension::from_f64s(0., &vec![1.0, 2.0, 3.0]).unwrap(),
            ],
        );

        let template = SplitMat::constant(
            0,
            vec![
                Dimension::from_f64s(-1., &vec![0.5, 1.0, 3.0]).unwrap(),
                Dimension::from_f64s(-1., &vec![0.5, 2.5, 4.0]).unwrap(),
            ],
        );

        let mean = split.sample(&template, SampleType::MEAN).unwrap();
        let areas = template.regions().map(|r| r.rectangle.area());
        let values = arr2(&[
            [
                1.0 * 0.5 * 0.5,
                1.0 * 0.5 * 0.5 + 2.0 * 0.5 * 1.0 + 3.0 * 0.5 * 0.5,
                3.0 * 0.5 * 0.5,
            ],
            [
                1.0 * 0.5 * 0.5,
                1.0 * 0.5 * 0.5 + 2.0 * 0.5 * 1.0 + 3.0 * 0.5 * 0.5,
                3.0 * 0.5 * 0.5,
            ],
            [
                2.0 * 1.0 * 0.5,
                2.0 * 1.0 * 0.5 + 4.0 * 1.0 * 1.0 + 6.0 * 1.0 * 0.5,
                6.0 * 1.0 * 0.5,
            ],
        ])
            .map(|&x| r64(x));
        let expected = &values / &areas;
        assert_eq!(expected, mean);
    }

    #[test]
    fn test_fingerprint() {
        let input = "# Position coordinates are ['x', 'y', 'z']
points
3
5.560900
partial_charge
-0.704200 -0.950600 -1.170700 -326215.946758
0.000000 0.000000 0.000000 646900.163088
3.029400 0.714600 -0.911600 -4965384.510770
0.000000 0.000000 0.000000 329163.950870
0.000000 0.000000 0.000000 329163.950870
1.580200 -1.068500 -0.328500 -2185012.362106
0.161000 -1.106700 -0.094100 669260.279575
0.000000 0.000000 0.000000 329163.950870
0.000000 0.000000 0.000000 2930920.967559
0.000000 0.000000 0.000000 2944627.365759
-0.357100 -0.674100 1.121300 -326215.946758
1.903500 0.550600 1.340000 119085.079028
-1.544700 0.048500 1.185000 -180827.702072
0.000000 0.000000 0.000000 646900.163088
0.000000 0.000000 0.000000 662605.402542
2.281500 -0.024800 0.007700 1821429.378262
-2.086400 0.600100 0.033900 1152301.053611
-2.531500 1.925000 -0.024200 -5079642.937129
0.000000 0.000000 0.000000 662605.402542
-1.731600 -0.014100 -1.158700 -180827.702072
        ";
        use crate::rivet::{compute, parse, parse_input, ComputationParameters, structure, Bounds};
        let rivet_input = parse_input(input.as_bytes()).unwrap();

        let params = ComputationParameters {
            param1_bins: 0,
            appearance_bins: 0,
            homology_dimension: 0,
            threads: 1,
        };
        let bytes = compute(&rivet_input, &params).unwrap();
        let results = parse(&bytes).unwrap();
        let structure = structure(&results);
        let bounds = Bounds {
            x_low: -5080000.0,
            y_low: 0.0,
            x_high: 2950000.0,
            y_high: 6.0,
        };
        let fp = fingerprint(&structure, &bounds, (5,5)).unwrap();
        println!("Vector: {:#?}", fp);
    }
}

