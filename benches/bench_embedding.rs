//! Criterion benches for the embedding forward pass. Phase 5 fills these
//! in; for now this file exists only so cargo can resolve [[bench]].

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("placeholder", |b| b.iter(|| 1 + 1));
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
