
use itertools::Itertools;
use std::ops;
use std::mem;
use ndarray::prelude::*;
use noisy_float::prelude::*;
use std::hash::Hash;
use std::hash::Hasher;
use rivet::BettiStructure;
use num_rational::Rational64;

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
    value: R64
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

enum Interval {
    Open(R64, R64),
    Closed(R64, R64),
    OpenClosed(R64, R64),
    ClosedOpen(R64, R64)
}

enum OpenClosed {
    Open,
    Closed
}

impl Interval {
    fn new(start: R64, start_included: OpenClosed, end: R64, end_included: OpenClosed) -> Option<Interval> {
        use OpenClosed::*;
        if start < end {
            Some(match (start_included, end_included) {
                case(Open, Open) => Interval::Open(start, end),
                case(Closed, Closed) => Interval::Closed(start, end),
                case(Open,Closed) => Interval::OpenClosed(start, end),
                case(Closed, Open) => Interval::ClosedOpen(start, end)
            })
        } else {
            None
        }
    }

    fn end_types(&self) -> (OpenClosed, OpenClosed) {
        match self {
            case Interval::Open(_,_) => (OpenClosed::Open, OpenClosed::Open),
            case Interval::Closed(_,_) => (OpenClosed::Closed, OpenClosed::Closed),
            case Interval::OpenClosed(_,_) => (OpenClosed::Open, OpenClosed::Closed),
            case Interval::ClosedOpen(_, _) => (OpenClosed::Closed, OpenClosed::Open)
        }
    }

    fn intersection(&self, other: &Interval) -> Option<Interval> {
        if self.start > other.end || other.start > self.end {
            None
        } else {
            let start = std::cmp::max(self.start, other.start);
            let end = std::cmp::min(self.end, other.end);
            let (self_start_type, self_end_type) = self.end_types();
            let (other_start_type, other_end_type) = other.end_types();
            let start_type = if self.start != other.start {
                OpenClosed::Closed
            } else {
                match (self_start_type, other_start_type) {
                    case (OpenClosed::Closed, OpenClosed::Closed) => OpenClosed::Closed,
                    _ => OpenClosed::Open
                }
            };
            let end_type = if self.end != other.end {
                OpenClosed::Closed
            } else {
                match (self_end_type, other_end_type) {
                    case (OpenClosed::Closed, OpenClosed::Closed) => OpenClosed::Closed,
                    _ => OpenClosed::Open
                }
            };
            Some(Interval::new(start, start_type, end, end_type).unwrap())
        }
    }
}

const DIMENSION_VERIFY: bool = false;

impl Dimension {
    pub fn new(lower_bound: R64, upper_bounds: Vec<R64>) -> Dimension {
        let dim = if upper_bounds.len() == 0 {
            Dimension {
                lower_bound,
                upper_bounds: vec![lower_bound],
                upper_indexes: vec![Some(0)],
            }
        } else {
            assert!(lower_bound <= upper_bounds[0]);
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
        dim
    }

    pub upper_bound(&self) -> R64 {
        self.upper_bounds.last().unwrap()
    }

    pub fn from_f64s(lower_bound: f64, upper_bounds: &[f64]) -> Dimension {
        Dimension::new(r64(lower_bound), upper_bounds.iter().map(|&x| r64(x)).collect_vec())
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
            results.push(Interval::OpenClosed(self.upper_bounds[i-1], self.upper_bounds[i]));
        }
        results
    }

    fn len(&self) -> usize {
        self.upper_bounds.len()
    }

    fn translate(&self, increment: R64) -> Dimension {
        Dimension {
            lower_bound: self.lower_bound + increment,
            upper_bounds: self.upper_bounds.iter().map(
                |x| *x + increment).collect_vec(),
            upper_indexes: self.upper_indexes.clone(),
        }
    }

    fn scale(&self, factor: R64) -> Dimension {
        Dimension {
            lower_bound: self.lower_bound * factor,
            upper_bounds: self.upper_bounds.iter().map(
                |x| *x * factor).collect_vec(),
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
            (DimensionQueryResult::Low, None)
        } else if value > *self.upper_bounds.last().unwrap() {
            (DimensionQueryResult::High, None)
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

fn is_sorted<T:Ord>(thing: &[T]) -> bool {
    let mut last : Option<&T> = None;
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
    MAX
}

pub struct Rectangle {
    d0: Interval,
    d1: Interval
}

impl Rectangle {
    pub fn new(d0: Interval, d1: Interval) -> Rectangle {
        Rectangle { d0, d1 }
    }

    pub fn intersection(&self, other: &Rectangle) -> Option<Rectangle> {
        let d0 = self.d0.intersection(&other.d0)?;
        let d1 = self.d1.intersection(&other.d1)?;
        Some(Rectangle {d0, d1})
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

pub struct Region {
    pub rectangle: Rectangle,
    pub value: Option<R64>
}

impl Region {
    pub fn new(rectangle: Rectangle, value: Option<R64>) -> Region {
        Region {rectangle, value}
    }

    pub fn intersection(&self, other: &Rectangle) -> Option<Region> {
        let rectangle = self.rectangle.intersection(&other)?;
        Ok(Region{rectangle, value: self.value})
    }
}

impl SplitMat {
    pub fn constant(constant: i32, dimensions: Vec<Dimension>) -> SplitMat {
        assert_eq!(dimensions.len(), 2, "Requires exactly 2 dimensions");
        SplitMat::new(
            Array2::from_elem((dimensions[0].len(), dimensions[1].len()), constant),
            dimensions)
    }

    pub fn new(mat: Array2<i32>, dimensions: Vec<Dimension>) -> SplitMat {
        {
            let shape = mat.shape();

            assert_eq!(dimensions.len(), shape.len(),
                       "number of dimensions does not equal array shape length");
            for dimension_number in 0..shape.len() {
                for upper_index in dimensions[dimension_number].upper_indexes.iter()
                    .filter(|x| x.is_some())
                    .map(|i| i.unwrap()) {
                    assert!(upper_index < shape[dimension_number],
                            format!("In dimension {}, dimension index {} is too big for matrix dimension {}",
                                    dimension_number, upper_index, mat.shape()[dimension_number]))
                }
            }
        }
        SplitMat { mat, dimensions }
    }

    fn shape(&self) -> (usize, usize) {
        (self.dimensions[0].upper_bounds.len(),
         self.dimensions[1].upper_bounds.len())
    }

    fn row(&self, length: R64) -> DimensionQueryResult {
        self.dimensions[0].index(length)
    }

    fn col(&self, length: R64) -> DimensionQueryResult {
        self.dimensions[1].index(length)
    }

    fn index(self, coords: (R64, R64)) -> (DimensionQueryResult, DimensionQueryResult) {
        (self.dimensions[0].index(coords.0), self.dimensions[1].index(coords.1))
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
            dimensions: self.dimensions.iter().zip(factors.iter())
                .map(|(d, f)| d.scale(*f))
                .collect_vec(),
        };
    }

    fn translate(&self, offsets: &[R64]) -> SplitMat {
        assert_eq!(offsets.len(), self.dimensions.len());
        return SplitMat {
            mat: self.mat.clone(),
            dimensions: self.dimensions.iter().zip(offsets.iter())
                .map(|(d, f)| d.translate(*f))
                .collect_vec(),
        };
    }


    fn merge(&self, rhs: &SplitMat) -> SplitMat {
        let left_dims = self.dimensions.iter().zip(rhs.dimensions.iter())
            .map(|(x, y)| x.merge(y))
            .collect_vec();
        SplitMat::new(self.mat.clone(), left_dims)
    }

    fn value(&self, point: (R64, R64)) -> Option<i32> {
        match self.index(point) {
            (DimensionQueryResult::In(d0), DimensionQueryResult::In(d1)) =>
                Some(self.mat[(self.dimension[0].upper_indexes[d0], self.dimension[1].upper_indexes[d1])]),
            _ => None
        }
    }

    fn regions(&self) -> Vec<Region> {
        let mut results = Vec::new();
        let d0_ints = self.dimensions[0].intervals();
        let d1_ints = self.dimensions[1].intervals();
        for (row_idx, row_interval) in d0_ints.into_iter().enumerate() {
            for (col_idx, col_interval) in d1_ints.into_iter().enumerate() {
                results.push(Region {
                    rectangle: Rectangle {d0: row_interval, d1: col_interval},
                    value: Some(r64(self.mat[(self.dimensions[0].upper_indexes[row_idx], self.dimensions[1].upper_indexes[col_idx])] as f64))
                })
            }
        }
        results
    }

//    fn regions(&self, rectangle: Rectangle) -> Vec<Region> {
//        let start_real = rectangle.position;
//        let end_real =(rectangle.position.0 + rectangle.size.0,
//                       rectangle.position.1 + rectangle.size.1);
//        let self_start_real = (self.dimensions[0].lower_bound, self.dimensions[1].lower_bound);
//        let self_end_real = (self.dimensions[0].upper_bound(), self.dimensions[1].upper_bound());
//        let start_discrete = self.index(start_real);
//        let end_discrete = self.index(end_real);
//        let mut results = Vec::new();
//        use DimensionQueryResult::*;
//        match start_discrete {
//            (Low, Low) => {
//                //Corner before all rows and columns
//                results.push(Region {
//                    rectangle: Rectangle::from_points(start_real, (self.dimensions[0].lower_bound, self.dimensions[1].lower_bound)).unwrap(),
//                    value: None
//                });
//                //Band before all rows, not including corner we just did
//                results.push(Region {
//                    rectangle: Rectangle::from_points((start_real.0, self_start_real.1),
//                                                      (self_start_real.0, std::cmp::min(self_end_real.1, end_real.1))).unwrap(),
//                    value: None
//                });
//                //Band before all columns, not including corner
//                results.push(Region {
//                    rectangle: Rectangle::from_points((self_start_real.0, start_real.1),
//                                                      (std::cmp::min(end_real.1, self_start_real.1), )).unwrap(),
//                    value: None
//                })
//            }
//        }
//        results
//    }


    /// Returns a matrix with the same shape as the given template (the template should contain the
    /// dimensions of self for meaningful results), with values that are the weighted
    /// averages (or max, or min) of any values in self that fall in a cell of the template.
    pub fn sample(&self, template: &SplitMat, sample_type: SampleType) -> Array2<R64> {
        let merged = self.merge(template).expand();
        let mut result =
            Array2::zeros((template.dimensions[0].len(), template.dimensions[1].len()));
        let mut merged_row = 0;
        let merged_rows = &merged.dimensions[0];
        assert_ne!(merged_rows.len(), 0);
        let merged_cols = &merged.dimensions[1];
        assert_ne!(merged_cols.len(), 0);
        let merged_rows_lens = merged_rows.lengths();
        let merged_cols_lens = merged_cols.lengths();
        let result_rows_lens = template.dimensions[0].lengths();
        let result_cols_lens = template.dimensions[1].lengths();

        #[derive(Clone)]
        struct Entry {
            width: R64,
            height: R64,
            value: i32
        }
        for row in 0..template.dimensions[0].len() {
            // Skip rows in merged that can't be in this row in result
            while (row == 0 && merged_rows.upper_bounds[merged_row] < template.dimensions[0].lower_bound)
                || (row > 0 && merged_rows.upper_bounds[merged_row] < template.dimensions[0].upper_bounds[row - 1]) {
                merged_row += 1;
            }
            let mut row_areas: Vec<Vec<Entry>> = vec![Vec::new(); merged_cols.len()];
            while merged_row < merged_rows.len()
                && merged_rows.upper_bounds[merged_row] <= template.dimensions[0].upper_bounds[row] {
                let mut merged_col = 0;
                for col in 0..template.dimensions[1].len() {
                    // Skip cols in merged that can't be in this col in template
                    while (col == 0 && merged_cols.upper_bounds[merged_col] < template.dimensions[1].lower_bound)
                        || (col > 0 && merged_cols.upper_bounds[merged_col] < template.dimensions[1].upper_bounds[col - 1]) {
                        merged_col += 1;
                    }
                    while merged_col < merged_cols.len()
                        && merged_cols.upper_bounds[merged_col] <= template.dimensions[1].upper_bounds[col] {
                        let entry = Entry {
                            width: merged_cols_lens[merged_col],
                            height: merged_rows_lens[merged_row],
                            value: merged.mat[(merged_row, merged_col)]
                        };
                        row_areas[col].push(entry);
                        merged_col += 1;
                    }
                }
                merged_row += 1;
            }
            let col_totals: Vec<R64> = row_areas.into_iter().map(
                |v| {
                    let it = v.into_iter();
                    match sample_type {
                        SampleType::MEAN =>
                            it.map(|e|r64(e.value as f64) * e.height * e.width).sum(),
                        SampleType::MAX =>
                            it.map(|e|r64(e.value as f64)).max().unwrap_or(r64(0.0)),
                        SampleType::MIN =>
                            it.map(|e|r64(e.value as f64)).min().unwrap_or(r64(0.0))
                    }
                }).collect_vec();
            for col in 0..template.dimensions[1].len() {
                let total = col_totals[col];
                result[(row, col)] = match sample_type {
                    SampleType::MEAN =>
                        total / (result_rows_lens[row] * result_cols_lens[col]),
                    _ => total
                }
            }
        }
        result
    }

    fn expand(&self) -> SplitMat {
        let mut vec = Vec::with_capacity(self.dimensions.iter().map(|x| x.len()).product());
        for row in 0..self.dimensions[0].len() {
            for col in 0..self.dimensions[1].len() {
                if self.dimensions[0].upper_indexes[row] == None
                    || self.dimensions[1].upper_indexes[col] == None {
                    vec.push(0);
                } else {
                    vec.push(self.mat[
                        [self.dimensions[0].upper_indexes[row].unwrap(),
                            self.dimensions[1].upper_indexes[col].unwrap()]
                        ]);
                }
            }
        }
        assert!(self.dimensions[0].len() > 0);
        assert!(self.dimensions[1].len() > 0);
        SplitMat::new(
            Array2::from_shape_vec((self.dimensions[0].len(), self.dimensions[1].len()), vec).unwrap(),
            self.dimensions.iter().map(|d| d.reset()).collect_vec(),
        )
    }

    fn weighted_difference(&self, other: &SplitMat) -> Array2<R64> {
        let split_diff = self - other;
        let mut diff = split_diff.mat.map(|&x| r64(x as f64));
        let dim_lengths = split_diff.dimensions.iter().map(|d| d.lengths()).collect_vec();
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

    pub fn betti_to_splitmat(betti: BettiStructure) -> Result<SplitMat, String> {
        let xs = betti.x_grades.iter().map(rational_to_r64).collect_vec();
        let ys = betti.y_grades.iter().map(rational_to_r64).collect_vec();
        if xs.len() == 0 {
            return Err("No x grades".to_owned())
        }
        if ys.len() == 0 {
            return Err("No y grades".to_owned())
        }
        if !is_sorted(&xs) {
            return Err("x grades from RIVET not sorted".to_owned())
        }
        if !is_sorted(&ys) {
            return Err("y grades from RIVET not sorted".to_owned())
        }
//        let dimensions = vec![
//            Dimension::new(ys[0], ys[1..].to_owned()),
//            Dimension::new(xs[0], xs[1..].to_owned())
//        ];

        let mut y_lengths = Vec::with_capacity(ys.len() -1);
        for i in 1..ys.len() {
            y_lengths.push(ys[i] - ys[i-1]);
        }
        let mut x_lengths = Vec::with_capacity(xs.len() -1);
        for i in 1..xs.len() {
            x_lengths.push(xs[i] - xs[i-1]);
        }
        let x_nonzero_lengths =
            x_lengths.iter().filter(|&&x| x != r64(0.)).count();
        let y_nonzero_lengths =
            y_lengths.iter().filter(|&&y| y != r64(0.)).count();
        if x_nonzero_lengths == 0 {
            return Err("No nonzero x lengths".to_owned())
        }
        if y_nonzero_lengths == 0 {
            return Err("No nonzero y lengths".to_owned())
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
            Dimension::new(unique_ys[0], unique_ys[1..].to_owned()),
            Dimension::new(unique_xs[0], unique_xs[1..].to_owned())
        ];
        assert_eq!(dimensions[0].len(), y_nonzero_lengths);
        assert_eq!(dimensions[1].len(), x_nonzero_lengths);
        let mut mat = Array2::<i32>::zeros(
            (y_nonzero_lengths, x_nonzero_lengths));
        for point in betti.points {
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

        mat.map_inplace(|x| if x < &mut 0 { *x = 0 });
        Ok(SplitMat { mat, dimensions })
    }
}

impl<'a, 'b> ops::Add<&'b SplitMat> for &'a SplitMat {
    type Output = SplitMat;

    fn add(self, rhs: &SplitMat) -> Self::Output {
        let left = self.merge(rhs).expand();
        let right = rhs.merge(self).expand();
        if left.mat.shape() != right.mat.shape() {
            error!("Merged splitmat shapes don't match!
            \nOriginal left: {:?}
            \nOriginal right: {:?}
            \n expanded shapes: L: {:?} R: {:?}", self, rhs, &left.mat.shape(), &right.mat.shape());
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
            println!("Merged splitmat shapes don't match!
            \nOriginal left: {:?}
            \nOriginal right: {:?}
            \n expanded shapes: L: {:?} R: {:?}", &self, &rhs, &left.mat.shape(), &right.mat.shape());
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


#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use hilbert_distance::SampleType;
    use hilbert_distance::SplitMat;
    use hilbert_distance::Dimension;
    use noisy_float::types::r64;

    #[test]
    fn expand() {
        let mut split = SplitMat::new(arr2(&[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]),
                                      vec![
                                          Dimension::from_f64s(0., &vec![0.25, 0.5, 1.0]),
                                          Dimension::from_f64s(0., &vec![0.25, 0.5, 1.0])
                                      ],
        );

        split.add_row(r64(0.1));
        split.add_row(r64(0.05));
        split.add_row(r64(0.8));
        split.add_row(r64(1.5));
        assert_eq!(split.dimensions[0].upper_bounds,
                   Dimension::from_f64s(0., &vec![0.05, 0.1, 0.25, 0.5, 0.8, 1.0, 1.5]).upper_bounds);
        split.add_col(r64(0.4));
        assert_eq!(split.dimensions[1].upper_bounds,
                   Dimension::from_f64s(0., &vec![0.25, 0.4, 0.5, 1.0]).upper_bounds);

        let expanded = split.expand();

        assert_eq!(expanded.mat, arr2(
            &[[1, 2, 2, 3],
                [1, 2, 2, 3],
                [1, 2, 2, 3],
                [4, 5, 5, 6],
                [7, 8, 8, 9],
                [7, 8, 8, 9],
                [0, 0, 0, 0]]));
    }

    #[test]
    fn distance() {
        let mut split = SplitMat::new(arr2(&[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]),
                                      vec![
                                          Dimension::from_f64s(0., &vec![0.25, 0.5, 1.0]),
                                          Dimension::from_f64s(0., &vec![0.25, 0.5, 1.0])
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
        let mut split = SplitMat::new(arr2(&[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]),
                                      vec![
                                          Dimension::from_f64s(0., &vec![0.25, 0.5, 1.0]),
                                          Dimension::from_f64s(0., &vec![0.25, 0.5, 1.0])
                                      ],
        );

        let split2 = SplitMat::new(arr2(&[
            [1, 2],
            [3, 4]]),
                                       vec![
                                           Dimension::from_f64s(0.2, &vec![0.25, 0.5]),
                                           Dimension::from_f64s(0.2, &vec![0.25, 0.5])
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
        assert_eq!(left_weighted.clone(), -(right_weighted.clone()), "weighted diffs");

        let left_squared = &left_weighted * &left_weighted;
        let right_squared = &right_weighted * &right_weighted;
        assert_eq!(&left_squared, &right_squared, "squared");

        let left_sum: f64 = left_squared.iter().map(|x| x.raw()).sum();
        let right_sum: f64 = right_squared.iter().map(|x| x.raw()).sum();
        assert_eq!(left_sum, right_sum, "sum");

        assert_eq!(left_sum.sqrt(), right_sum.sqrt(), "sqrt");

        assert_eq!(split2.distance(&split), r64(2.701316808521355), "exact value");
        assert_eq!(split2.distance(&split), split.distance(&split2), "symmetry");
    }

    #[test]
    fn test_sample_min() {
        let mut split = SplitMat::new(arr2(&[
            [1, 2, 3],
            [2, 4, 6]]),
                                      vec![
                                          Dimension::from_f64s(0., &vec![1.0, 2.0]),
                                          Dimension::from_f64s(0., &vec![1.0, 2.0, 3.0]),
                                      ],
        );

        let template = SplitMat::constant(0,
                                          vec![
                                              Dimension::from_f64s(-1., &vec![0.5, 1.0, 3.0]),
                                              Dimension::from_f64s(-1., &vec![0.5, 2.5, 4.0]),
                                          ]);
        let min = split.sample(&template, SampleType::MIN);
        assert_eq!(min, arr2(&[
            [1.0, 1.0, 3.0],
            [1.0, 1.0, 3.0],
            [2.0, 2.0, 6.0]
        ]).map(|&x| r64(x)));
    }

    #[test]
    fn test_sample_max() {
        let mut split = SplitMat::new(arr2(&[
            [1, 2, 3],
            [2, 4, 6]]),
                                      vec![
                                          Dimension::from_f64s(0., &vec![1.0, 2.0]),
                                          Dimension::from_f64s(0., &vec![1.0, 2.0, 3.0]),
                                      ],
        );

        let template = SplitMat::constant(0,
                                          vec![
                                              Dimension::from_f64s(-1., &vec![0.5, 1.0, 3.0]),
                                              Dimension::from_f64s(-1., &vec![0.5, 2.5, 4.0]),
                                          ]);
        let max = split.sample(&template, SampleType::MAX);
        assert_eq!(max, arr2(&[
            [1.0, 3.0, 3.0],
            [1.0, 3.0, 3.0],
            [2.0, 6.0, 6.0]
        ]).map(|&x| r64(x)));
    }

    #[test]
    fn test_sample_mean() {
        let mut split = SplitMat::new(arr2(&[
            [1, 2, 3],
            [2, 4, 6]]),
                                      vec![
                                          Dimension::from_f64s(0., &vec![1.0, 2.0]),
                                          Dimension::from_f64s(0., &vec![1.0, 2.0, 3.0]),
                                      ],
        );

        let template = SplitMat::constant(0,
                                          vec![
                                              Dimension::from_f64s(-1., &vec![0.5, 1.0, 3.0]),
                                              Dimension::from_f64s(-1., &vec![0.5, 2.5, 4.0]),
                                          ]);

        let mean = split.sample(&template, SampleType::MEAN);
        assert_eq!(mean, arr2(&[
            [1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.875, 1.0]
        ]).map(|&x|r64(x)));
    }
}