use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::hint::black_box;
use std::thread;
use std::time::{Duration, Instant};

const RUNS: usize = 10;
const REL_TOL: f64 = 1e-8;

fn heavy_transform(value: f64) -> f64 {
    let mut result = value.sin();
    for _ in 0..8 {
        result = (result * result + 1.0).sqrt().cos();
    }
    result
}

fn generate_data(count: usize) -> Vec<f64> {
    (0..count).map(|index| index as f64 * 0.0001 + 1.0).collect()
}

fn sequential_sum(num: &[f64]) -> f64 {
    num.iter().map(|&number| heavy_transform(number)).sum()
}

// https://rust-lang-nursery.github.io/rust-cookbook/concurrency/parallel.html
fn parallel_sum(num: &[f64]) -> f64 {
    num.par_iter().map(|&number| heavy_transform(number)).sum()
}

// https://rust-lang-nursery.github.io/rust-cookbook/concurrency/parallel.html
fn parallel_reduce(num: &[f64]) -> f64 {
    num
        .par_iter()
        .map(|&number| heavy_transform(number))
        .reduce(|| 0.0, |a, b| a + b)
}

// https://doc.rust-lang.org/std/thread/fn.scope.html
fn threads_sum(num: &[f64], thread_c: usize) -> f64 {
    let chunk_size = ((num.len() + thread_c - 1) / thread_c).max(1);
    thread::scope(|scope| {
        let mut workers = Vec::new();
        for part in num.chunks(chunk_size) {
            let worker = scope.spawn(move || {
                part.iter().map(|&number| heavy_transform(number)).sum::<f64>()
            });
            workers.push(worker);
        }
        let mut total = 0.0;
        for worker in workers {
            total += worker.join().unwrap();
        }
        total
    })
}

// https://en.wikipedia.org/wiki/Xorshift
fn xorshift64(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    value
}

fn random_data(count: usize) -> Vec<u64> {
    let mut state = 88172645463325252u64;
    (0..count).map(|_| xorshift64(&mut state)).collect()
}

// https://docs.rs/rayon/latest/rayon/fn.join.html
fn quick_sort<T: PartialOrd + Send>(v: &mut [T]) {
    if v.len() > 1 {
        let mid = partition(v);
        let (lo, hi) = v.split_at_mut(mid);
        rayon::join(|| quick_sort(lo),
                    || quick_sort(hi));
    }
}

fn quick_sort_seq<T: PartialOrd>(v: &mut [T]) {
    if v.len() > 1 {
        let mid = partition(v);
        let (lo, hi) = v.split_at_mut(mid);
        quick_sort_seq(lo);
        quick_sort_seq(hi);
    }
}

// https://docs.rs/rayon/latest/rayon/fn.join.html
fn partition<T: PartialOrd>(v: &mut [T]) -> usize {
    let pivot = v.len() - 1;
    let mut boundary = 0;
    for j in 0..pivot {
        if v[j] <= v[pivot] {
            v.swap(boundary, j);
            boundary += 1;
        }
    }
    v.swap(boundary, pivot);
    boundary
}

fn min_time<T, F: FnMut() -> T>(mut task: F) -> (T, Duration) {
    let mut best_time = Duration::MAX;
    let mut last_result = None;
    for _ in 0..RUNS {
        let started_at = Instant::now();
        let result = black_box(task());
        let elapsed = started_at.elapsed();

        if elapsed < best_time {
            best_time = elapsed;
        }

        last_result = Some(result);
    }
    (last_result.unwrap(), best_time)
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn check(label: &str, expected: f64, actual: f64) {
    let diff = (expected - actual).abs();
    if diff > REL_TOL * expected.abs().max(actual.abs()).max(1.0) {
        panic!("{}: got {}, expected {}", label, actual, expected);
    }
}

fn time_sort<F: Fn(&mut [u64])>(input: &[u64], expected_sorted: &[u64], sort_fn: F) -> Duration {
    let (sorted, best_time) = min_time(|| {
        let mut copy = input.to_vec();
        sort_fn(&mut copy);
        copy
    });
    assert_eq!(sorted, expected_sorted, "sort: wrong result");
    best_time
}

fn experiment_sizes(thread_c: usize) {
    println!("\n=-=-=-=-=-=-=-=-=-=-= Experiment #1: different N (threads = {}) =-=-=-=-=-=-=-=-=-=-=", thread_c);
    // https://stackoverflow.com/questions/30379341/how-to-print-well-formatted-tables-to-the-console
    println!(
        "{:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8}",
        "N", "seq, ms", "rayon, ms", "thread, ms", "S_rayon", "S_thread"
    );
    println!("{}", "-".repeat(71));

    let sizes = [1_000usize, 10_000, 100_000, 1_000_000, 10_000_000];
    for &size in sizes.iter() {
        let num = generate_data(size);
        let (sum_seq, time_seq) = min_time(|| sequential_sum(&num));
        let (sum_rayon, time_rayon) = min_time(|| parallel_sum(&num));
        let (sum_threads, time_threads) = min_time(|| threads_sum(&num, thread_c));
        check("rayon", sum_seq, sum_rayon);
        check("thread", sum_seq, sum_threads);
        check("reduce", sum_seq, parallel_reduce(&num));
        println!(
            "{:>10} | {:>10.4} | {:>10.4} | {:>10.4} | {:>8.2} | {:>8.2}",
            size,
            ms(time_seq),
            ms(time_rayon),
            ms(time_threads),
            ms(time_seq) / ms(time_rayon),
            ms(time_seq) / ms(time_threads)
        );
    }
}

fn experiment_scaling(thread_c: usize) {
    let size = 1_000_000;
    println!("\n=-=-=-=-=-=-=-=-=-=-= Experiment #2: threads (N = {}) =-=-=-=-=-=-=-=-=-=-=", size);
    println!(
        "{:>8} | {:>10} | {:>7} | {:>7}",
        "threads", "T_p, ms", "S", "E"
    );
    println!("{}", "-".repeat(41));

    let num = generate_data(size);
    let (sum_seq, time_seq) = min_time(|| sequential_sum(&num));
    println!(
        "{:>8} | {:>10.4} | {:>7.2} | {:>7.2}",
        "seq",
        ms(time_seq),
        1.0,
        1.0
    );

    let mut thread_counts: Vec<usize> = vec![1, 2, 4, 6, 8, 12, 16];
    thread_counts.retain(|&pool_threads| pool_threads < thread_c);
    thread_counts.push(thread_c);

    for &pool_threads in thread_counts.iter() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .build()
            .unwrap();
        let (sum_pool, time_pool) = min_time(|| pool.install(|| parallel_sum(&num)));
        check("rayon pool", sum_seq, sum_pool);

        let speedup = ms(time_seq) / ms(time_pool);
        let efficiency = speedup / pool_threads as f64;
        println!(
            "{:>8} | {:>10.4} | {:>7.2} | {:>7.2}",
            pool_threads,
            ms(time_pool),
            speedup,
            efficiency
        );
    }
}

fn experiment_sort() {
    println!("\n=-=-=-=-=-=-=-=-=-=-= Experiment #3: sorting u64 =-=-=-=-=-=-=-=-=-=-=");
    println!(
        "{:>9} | {:>10} | {:>10} | {:>10} | {:>11} | {:>7} | {:>7}",
        "n", "std, ms", "par, ms", "qs_seq, ms", "qs_join, ms", "S_par", "S_join"
    );
    println!("{}", "-".repeat(82));

    let sizes = [100_000usize, 1_000_000, 5_000_000];
    for &size in sizes.iter() {
        let input = random_data(size);
        let mut expected_sorted = input.clone();

        expected_sorted.sort_unstable();

        let time_std = time_sort(&input, &expected_sorted, |items| items.sort_unstable());

        // https://rust-lang-nursery.github.io/rust-cookbook/concurrency/parallel.html
        let time_par = time_sort(&input, &expected_sorted, |items| items.par_sort_unstable());
        let time_qs = time_sort(&input, &expected_sorted, |items| quick_sort_seq(items));
        let time_join = time_sort(&input, &expected_sorted, |items| quick_sort(items));
        println!(
            "{:>9} | {:>10.2} | {:>10.2} | {:>10.2} | {:>11.2} | {:>7.2} | {:>7.2}",
            size,
            ms(time_std),
            ms(time_par),
            ms(time_qs),
            ms(time_join),
            ms(time_std) / ms(time_par),
            ms(time_qs) / ms(time_join)
        );
    }
}

fn main() {
    if cfg!(debug_assertions) {
        println!("WARNING: debug build. Use: cargo run --release");
    }

    let thread_c = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    println!("logical processors: {}", thread_c);
    println!("rayon pool size: {}", rayon::current_num_threads());
    println!("runs per measurement: {} (minimum time is reported)", RUNS);

    experiment_sizes(thread_c);
    experiment_scaling(thread_c);
    experiment_sort();
}