use ndarray::Array2;

enum PointSet {}

#[link(name = "bottleneck")]
extern {
    fn pointset_new() -> *mut PointSet;
    fn pointset_delete(pointset: *mut PointSet) -> ();
    fn pointset_insert(pointset: *mut PointSet, x: f64, y: f64);
    fn distance(pointset1: *mut PointSet,
                    pointset2: *mut PointSet,
                    approx_epsilon: f64,
                    result: &mut f64) -> i32;
}

impl Drop for PointSet {
    fn drop(&mut self) {
        unsafe {
            println!("Calling pointset_delete");
            pointset_delete(self);
        }
    }
}

pub fn bottleneck_distance(lhs: &Array2<f64>, rhs: &Array2<f64>) -> Option<f64> {

    let bc1_len = lhs.shape()[0];
    let bc2_len = rhs.shape()[0];

    unsafe {
        let mut dist: f64 = 0.0;
        if bc1_len == 0 && bc2_len == 0 {
            //Do nothing, 0.0f64 is fine.
//        } else if bc1_len == 0 {
//            //                ps2.iter().map(|x| (x.1 - x.0))
//            //                    .max_by(|x, y| x.partial_cmp(y)
//            //                        .unwrap_or(cmp::Ordering::Equal))
//            //                    .unwrap()
//        } else if bc2_len == 0 {
//            //                ps1.iter().map(|x| (x.1 - x.0))
//            //                    .max_by(|x, y| x.partial_cmp(y)
//            //                        .unwrap_or(cmp::Ordering::Equal))
//            //                    .unwrap()
        } else {
            let set1 = pointset_new();
            let set2 = pointset_new();
            for i in 0..bc1_len {
                let mult = lhs[[i, 2]] as i64;
                for _ in 0..mult {
                    let b = lhs[(i, 0)];
                    let d = lhs[(i, 1)];
                    if b != d {
                        pointset_insert(set1, b, d);
                    }
                }
            }
            for i in 0..bc2_len {
                let mult = rhs[[i, 2]] as i64;
                for _ in 0..mult {
                    let b = rhs[(i, 0)];
                    let d = rhs[(i, 1)];
                    if b != d {
                        pointset_insert(set2, b, d);
                    }
                }
            }
            let res  = distance(set1, set2, 0.001, &mut dist);
            pointset_delete(set1);
            pointset_delete(set2);
            if res != 0 {
                return None;
            }
        };
        return Some(dist);
    }
}