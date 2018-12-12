use rivet;
use rivet::RivetError;
use std::cmp;
use hera;
use ndarray::{Array1, Zip};

fn find_offset(slope: f64, point: (f64, f64)) -> f64 {
    if slope == 90.0 { return -point.0; }

    let m = slope.to_radians().tan();
    let b = point.1 - point.0 * m;
    let x_minimizer = -1.0 * (point.1 * m - point.0 * m.powi(2)) / (1.0 + m.powi(2));
    let y_minimizer = m * x_minimizer + b;
    let unsigned_dist : f64 = (x_minimizer.powi(2) + y_minimizer.powi(2)).sqrt();
    if b > 0.0 {
        unsigned_dist
    } else {
        -unsigned_dist
    }
}

fn generate_lines(grid_size: u32,
                  upper_left: (f64, f64),
                  lower_right: (f64, f64)) -> Vec<(f64, f64)>{

    let mut lines:Vec<(f64, f64)> = Vec::new();
    for i in 0..grid_size {
        let slope = 90.0 * (i as f64 + 1.0) / (grid_size as f64 + 1.0);
        let ul_offset = find_offset(slope, upper_left);
        let lr_offset = find_offset(slope, lower_right);
        if grid_size == 1 {
            lines.push((slope, ul_offset - lr_offset));
        } else {
            for j in 0..grid_size {
                let offset = lr_offset + j as f64 * (ul_offset - lr_offset) / (grid_size as f64 - 1.0);
                lines.push((slope, offset as f64));
            }
        }
    }
    lines
}

pub struct MatchResult {
    pub distance: f64,
    pub error_count: usize
}

pub fn match_dist(multi_bars1: &[rivet::BarCode],
                  multi_bars2: &[rivet::BarCode],
                  fixed_bounds: &rivet::Bounds,
                  lines: &[(f64, f64)],
                  normalize: bool
                ) -> MatchResult {
    if fixed_bounds.is_degenerate() {
        MatchResult { distance: hera::bottleneck_distance(&multi_bars1[0].bars, &multi_bars2[0].bars).expect("Couldn't calculate distance"),
                        error_count: 0}
    } else {
        let mut raw_distances: Array1<f64> = Array1::<f64>::zeros(lines.len());
        for i in 0..lines.len() {
            let bc1 = &multi_bars1[i];
            let bc2 = &multi_bars2[i];
            let dist = hera::bottleneck_distance(&bc1.bars, &bc2.bars);
            //We take the max at the end, so negative distances will be ignored
            raw_distances[i] = dist.unwrap_or(-1.0);
        }
        let failures = raw_distances.fold(0, |x, y| if y < &0.0 { x + 1 } else { x });
        let delta_x = fixed_bounds.x_high - fixed_bounds.x_low;
        let delta_y = fixed_bounds.y_high - fixed_bounds.y_low;
        //    # now compute matching distance
        //
        //    # to determine the weight of a line with the given slope,
        //    # we need to take into account both the weight coming from slope of
        //    # the line, and also the normalization, which changes both the effective
        //    # weight and the effective bottleneck distance.
        //
        let slopes: Array1<f64> = lines.iter().map(|x| x.0).collect();
        //    raw_distance = line_distances[:, 1]
        let m = slopes.map(|x: &f64| x.to_radians().tan());
        let w = calculate_weight(&slopes, normalize, delta_x, delta_y);
        //    # moreover, normalization changes the length of a line segment along the line (slope,offset),
        //    # and hence also the bottleneck distance, by a factor of
        //    if normalize:
        //        m = np.tan(np.radians(slope))
        //    bottleneck_stretch = np.sqrt(
        //        ((m / delta_y) ** 2 + (1 / delta_x) ** 2) / (m ** 2 + 1))
        //    else:
        //    bottleneck_stretch = 1
        let bottleneck_stretch: Array1<f64> = if normalize {
            let stretch =
                m.map(|t| ((t / delta_y).powi(2)
                    + (1.0 / delta_x).powi(2) / (t.powi(2) + 1.0)).sqrt());
            stretch
        } else {
            Array1::<f64>::from_elem(m.len(), 1.0)
        };

        //    m_dist = np.max(w * raw_distance * bottleneck_stretch)
        //    return m_dist
        let m_dist: f64 = *((w * raw_distances * bottleneck_stretch).iter().max_by(
            |f1, f2| {
                f1.partial_cmp(f2).unwrap_or(cmp::Ordering::Equal)
            }).unwrap());
        MatchResult { distance: m_dist, error_count: failures }
    }
}
pub fn matching_distance(lhs: &[u8], rhs: &[u8], grid_size: u32, normalize: bool) -> Result<MatchResult, RivetError> {
    let comp1 = rivet::parse(lhs)?;
    let comp2 = rivet::parse(rhs)?;
    //    # First, use fixed_bounds to set the upper right corner and lower-left
    //    # corner to be considered.
    //    if fixed_bounds is None:
    //    # otherwise, determine bounds from the bounds of the two modules
    let fixed_bounds = comp1.bounds().common_bounds(&comp2.bounds());
    let ul = (fixed_bounds.x_low, fixed_bounds.y_high);
    let lr = (fixed_bounds.x_high, fixed_bounds.y_low);
    //    # Now we build up a list of the lines we consider in computing the matching distance.
    //    # Each line is given as a (slope,offset) pair.
    let lines = generate_lines(grid_size, ul, lr);
    //    # next, for each of the two 2-D persistence modules, get the barcode
    //    # associated to the list of lines.
    let multi_bars1 = rivet::barcodes(&comp1, &lines)?;
//    println!("bars1: {:?}", multi_bars1);
    let multi_bars2 = rivet::barcodes(&comp2, &lines)?;
//    println!("bars2: {:?}", multi_bars2);
//    # first compute the unweighted distance between the pairs
    Ok(match_dist(multi_bars1.as_ref(), multi_bars2.as_ref(), &fixed_bounds, lines.as_ref(), normalize))
}

fn recip(arr: &Array1<f64>) -> Array1<f64> {
    arr.map(|x:&f64|
            if x.partial_cmp(&0.0f64)
                .unwrap_or(cmp::Ordering::Equal) == cmp::Ordering::Equal {0.0}
            else {1.0/x})
}

fn maximum(arr: &Array1<f64>, arr2: &Array1<f64>) -> Array1<f64> {
    let mut res = Array1::<f64>::zeros(arr.len());

    Zip::from(&mut res).and(arr).and(arr2).apply(|q, &m, &r|{
        *q = f64::max(m, r);
    });

    res
}

fn calculate_weight(m: &Array1<f64>, normalize: bool, delta_x: f64, delta_y: f64) -> Array1<f64> {
    // When computing the matching distance, each line slope considered is assigned a weight.
    // This function computes that weight.  It will also be used elsewhere to compute a
    // "weighted norm" of a rank function.

    // first, let's recall how the re-weighting works in the un-normalized case.
    // According to the definition of the matching distance, we choose
    // the weight so that if the interleaving distance between Mod1 and Mod2
    // is 1, then the weighted bottleneck distance between the slices is at most 1.

    let effective_m = if normalize {
        //        next, let's consider the normalized case. If the un-normalized slope
        //        is 'slope', then the normalized slope is given as follows
        if delta_y.partial_cmp(&0.0).unwrap_or(cmp::Ordering::Equal) == cmp::Ordering::Equal {
            panic!("delta_y == 0 or otherwise invalid")
        }
        m * delta_x / delta_y
    } else {
        m.clone() // since other branch produces a new ArrayBase (not a ref),
        // we must in this one as well
    };
    let q = maximum(&effective_m, &recip(&effective_m));
    let w = (1.0 / (1.0 + q.map(|x|x.powi(2)))).map(|x|x.sqrt());
    w
}