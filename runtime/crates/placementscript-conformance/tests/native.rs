//! Conformance suite, native target (TODO 1.5).

use placementscript_conformance::{check, load_cases, run_native};

#[test]
fn native_conformance() {
    let cases = load_cases();
    let mut failures = Vec::new();
    for (file, case) in &cases {
        let outcome = run_native(case);
        if let Err(e) = check(case, &outcome, "native") {
            failures.push(format!(
                "{}: {e}",
                file.file_name().unwrap().to_string_lossy()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} native conformance failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    println!("native conformance: {} cases passed", cases.len());
}
