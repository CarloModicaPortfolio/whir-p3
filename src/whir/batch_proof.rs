//! Batch opening proof for the WHIR protocol.
//!
//! Implements batch opening of two polynomials `f_a` and `f_b` at different
//! evaluation points `z_a` and `z_b` using a selector variable approach.
//!
//! # Protocol Overview
//!
//! 1. Both polynomials are committed separately.
//! 2. A selector variable `X` creates a virtual combined polynomial:
//!    `f_c(X, x) = X·f_a(x) + (1-X)·f_b(x)`
//! 3. A weight polynomial batches both evaluation claims:
//!    `w(X, x) = X·eq(x, z_a) + α·(1-X)·eq(x, z_b)`
//! 4. One round of sumcheck on X produces challenge `r_0`.
//! 5. The folded polynomial `g(x) = r_0·f_a(x) + (1-r_0)·f_b(x)` is
//!    materialized and the standard WHIR protocol continues on `g`.

use alloc::vec::Vec;
use core::fmt;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_multilinear_util::{evals::EvaluationsList, multilinear::MultilinearPoint};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{
    constraints::{Constraint, statement::EqStatement},
    fiat_shamir::errors::FiatShamirError,
    sumcheck::{SumcheckData, extrapolate_012, product_polynomial::ProductPolynomial, prover::SumcheckProver},
    whir::{
        proof::WhirProof,
        prover::{Prover, round_state::RoundState},
    },
};

/// Batch opening proof for two polynomials.
///
/// Wraps batch-specific data (selector sumcheck, OOD answers for both polynomials)
/// alongside the inner WHIR proof that operates on the folded polynomial.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct BatchWhirProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// OOD answers for both polynomials: [ood_a, ood_b]
    pub initial_ood_answers: [Vec<EF>; 2],

    /// Selector sumcheck data: stores [c0, c2] for h(X)
    pub selector_sumcheck: SumcheckData<F, EF>,

    /// Inner WHIR proof on the folded polynomial g = r_0·f_a + (1-r_0)·f_b
    pub inner_proof: WhirProof<F, EF, MT>,
}

impl<F: Send + Sync + Clone, EF, MT: Mmcs<F>> fmt::Debug for BatchWhirProof<F, EF, MT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchWhirProof").finish_non_exhaustive()
    }
}

impl<EF, F, MT, Challenger> Prover<'_, EF, F, MT, Challenger>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    MT: Mmcs<F>,
{
    /// Performs the selector sumcheck round for batch opening.
    ///
    /// This is a single round of sumcheck over the selector variable X, which
    /// folds two polynomials f_a and f_b into a single polynomial g.
    ///
    /// Returns `(sumcheck_prover, r_0)` where:
    /// - `sumcheck_prover` contains the folded polynomial g and weight w'
    /// - `r_0` is the selector challenge from Fiat-Shamir
    #[instrument(skip_all)]
    #[allow(clippy::too_many_arguments)]
    fn selector_round(
        &self,
        selector_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        f_a: &EvaluationsList<F>,
        f_b: &EvaluationsList<F>,
        z_a: &MultilinearPoint<EF>,
        z_b: &MultilinearPoint<EF>,
        v_a: EF,
        v_b: EF,
        alpha: EF,
    ) -> (SumcheckProver<F, EF>, EF) {
        let n = f_a.num_variables();
        debug_assert_eq!(n, f_b.num_variables());
        debug_assert_eq!(n, z_a.num_variables());
        debug_assert_eq!(n, z_b.num_variables());

        // Build virtual combined polynomial: f_c = [f_b | f_a]
        // X=0 half is f_b, X=1 half is f_a
        let combined_evals: Vec<F> = f_b
            .as_slice()
            .iter()
            .chain(f_a.as_slice().iter())
            .copied()
            .collect();
        let combined_poly = EvaluationsList::new(combined_evals);

        // Build combined weights: w = [α·eq(·, z_b) | eq(·, z_a)]
        let eq_z_b = EvaluationsList::new_from_point(z_b.as_slice(), alpha);
        let eq_z_a = EvaluationsList::new_from_point(z_a.as_slice(), EF::ONE);
        let combined_weights: Vec<EF> = eq_z_b
            .as_slice()
            .iter()
            .chain(eq_z_a.as_slice().iter())
            .copied()
            .collect();
        let combined_weights = EvaluationsList::new(combined_weights);

        // Compute sumcheck coefficients: h(X) = c0 + c1·X + c2·X²
        let (c0, c2) = combined_poly.sumcheck_coefficients(&combined_weights);

        // Sanity check: h(0) = sum_{x} f_b(x)·α·eq(x,z_b) = α·v_b
        debug_assert_eq!(c0, alpha * v_b);

        // Fiat-Shamir: commit (c0, c2) and receive challenge r_0
        let r_0 = selector_data.observe_and_sample::<_, F>(challenger, c0, c2, self.starting_folding_pow_bits);

        // Materialize g(x) = r_0·f_a(x) + (1-r_0)·f_b(x)
        let one_minus_r0 = EF::ONE - r_0;
        let g: Vec<EF> = f_a
            .as_slice()
            .iter()
            .zip(f_b.as_slice().iter())
            .map(|(&a, &b)| r_0 * EF::from(a) + one_minus_r0 * EF::from(b))
            .collect();
        let g = EvaluationsList::new(g);

        // Materialize w'(x) = r_0·eq(x,z_a) + α·(1-r_0)·eq(x,z_b)
        let w_prime: Vec<EF> = eq_z_a
            .as_slice()
            .iter()
            .zip(eq_z_b.as_slice().iter())
            .map(|(&a, &b)| r_0 * a + one_minus_r0 * b)
            .collect();
        let w_prime = EvaluationsList::new(w_prime);

        // Compute folded sum: σ' = h(r_0) via quadratic extrapolation
        let sigma = v_a + alpha * v_b;
        let sigma_prime = extrapolate_012(c0, sigma - c0, c2, r_0);

        // Create sumcheck prover for continuation
        let poly = ProductPolynomial::new_small(g, w_prime);
        debug_assert_eq!(poly.dot_product(), sigma_prime);

        (SumcheckProver { poly, sum: sigma_prime }, r_0)
    }

    /// Executes the batch opening proof protocol for two polynomials.
    ///
    /// # Parameters
    ///
    /// - `dft`: DFT backend for polynomial evaluation
    /// - `proof`: Mutable proof structure to populate
    /// - `challenger`: Fiat-Shamir transcript
    /// - `prover_data_a`: Merkle tree prover data for f_a
    /// - `prover_data_b`: Merkle tree prover data for f_b
    /// - `f_a`: First polynomial in evaluation form
    /// - `f_b`: Second polynomial in evaluation form
    /// - `statement_a`: Evaluation constraint for f_a: f_a(z_a) = v_a
    /// - `statement_b`: Evaluation constraint for f_b: f_b(z_b) = v_b
    /// - `ood_statement_a`: Out-of-domain constraints for f_a
    /// - `ood_statement_b`: Out-of-domain constraints for f_b
    #[instrument(skip_all)]
    #[allow(clippy::too_many_arguments)]
    pub fn batch_prove<Dft>(
        &self,
        dft: &Dft,
        proof: &mut BatchWhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        prover_data_a: MT::ProverData<DenseMatrix<F>>,
        prover_data_b: MT::ProverData<DenseMatrix<F>>,
        f_a: &EvaluationsList<F>,
        f_b: &EvaluationsList<F>,
        statement_a: &EqStatement<EF>,
        statement_b: &EqStatement<EF>,
        ood_statement_a: &EqStatement<EF>,
        ood_statement_b: &EqStatement<EF>,
    ) -> Result<(), FiatShamirError>
    where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        let num_variables = f_a.num_variables();
        assert_eq!(num_variables, f_b.num_variables());
        assert!(self.validate_parameters());

        // Extract single evaluation claims: f_a(z_a) = v_a, f_b(z_b) = v_b
        let (z_a, v_a) = single_constraint(statement_a);
        let (z_b, v_b) = single_constraint(statement_b);

        // Sample batching randomness α
        let alpha: EF = challenger.sample_algebra_element();

        // Run selector sumcheck round
        let (mut sumcheck_prover, r_0) = self.selector_round(
            &mut proof.selector_sumcheck,
            challenger,
            f_a,
            f_b,
            &z_a,
            &z_b,
            v_a,
            v_b,
            alpha,
        );

        // Fold OOD constraints: ood_folded = r_0·ood_a + (1-r_0)·ood_b
        let folded_ood = fold_ood_constraints(ood_statement_a, ood_statement_b, r_0);

        // Combine folded OOD constraints into the sumcheck (same as initial round in single-poly)
        let ood_constraint =
            Constraint::new_eq_only(challenger.sample_algebra_element(), folded_ood);

        // Run folding_factor rounds of sumcheck incorporating OOD constraints
        let folding_factor = self.folding_factor.at_round(0);
        let folding_randomness = sumcheck_prover.compute_sumcheck_polynomials(
            &mut proof.inner_proof.initial_sumcheck,
            challenger,
            folding_factor,
            self.starting_folding_pow_bits,
            Some(ood_constraint),
        );

        // Initialize RoundState with BOTH commitment trees
        let mut round_state = RoundState {
            sumcheck_prover,
            folding_randomness,
            commitment_merkle_prover_data: prover_data_a,
            merkle_prover_data: None,
            batch_base_data: Some(prover_data_b),
            batch_r0: Some(r_0),
        };

        // Run standard WHIR rounds
        for round in 0..=self.n_rounds() {
            self.round(dft, round, &mut proof.inner_proof, challenger, &mut round_state)?;
        }

        Ok(())
    }
}

/// Extracts a single evaluation constraint (point, value) from an EqStatement.
///
/// Panics if the statement does not contain exactly one constraint.
fn single_constraint<EF: Field>(statement: &EqStatement<EF>) -> (MultilinearPoint<EF>, EF) {
    assert_eq!(
        statement.len(),
        1,
        "Batch opening expects exactly one evaluation claim per polynomial"
    );
    let (point, &value) = statement.iter().next().unwrap();
    (point.clone(), value)
}

/// Folds OOD constraints from two polynomials using the selector challenge.
///
/// Both polynomials must have been evaluated at the same OOD challenge points.
/// The folded value is: `r_0 * ood_a + (1 - r_0) * ood_b`.
fn fold_ood_constraints<EF: Field>(
    ood_a: &EqStatement<EF>,
    ood_b: &EqStatement<EF>,
    r_0: EF,
) -> EqStatement<EF> {
    assert_eq!(ood_a.len(), ood_b.len());
    let num_variables = ood_a.num_variables();
    let mut folded = EqStatement::initialize(num_variables);

    let one_minus_r0 = EF::ONE - r_0;
    for ((point_a, &v_a), (point_b, &v_b)) in ood_a.iter().zip(ood_b.iter()) {
        debug_assert_eq!(point_a, point_b, "OOD points must match");
        let folded_value = r_0 * v_a + one_minus_r0 * v_b;
        folded.add_evaluated_constraint(point_a.clone(), folded_value);
    }

    folded
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    extern crate std;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
    use p3_commit::Mmcs;
    use p3_dft::TwoAdicSubgroupDft;
    use p3_field::{Field, PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrixView;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::{evals::EvaluationsList, multilinear::MultilinearPoint};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig},
        whir::prover::Prover,
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

    /// Commit a polynomial using the same Reed-Solomon encoding as CommitmentWriter
    /// but without OOD sampling or InitialStatement coupling.
    fn raw_commit<Dft: TwoAdicSubgroupDft<F>>(
        params: &WhirConfig<EF, F, MyMmcs, MyChallenger>,
        dft: &Dft,
        poly: &EvaluationsList<F>,
        challenger: &mut MyChallenger,
    ) -> <MyMmcs as Mmcs<F>>::ProverData<p3_matrix::dense::DenseMatrix<F>> {
        let num_vars = poly.num_variables();
        let mut mat = RowMajorMatrixView::new(
            poly.as_slice(),
            1 << (num_vars - params.folding_factor.at_round(0)),
        )
        .transpose();
        mat.pad_to_height(
            1 << (num_vars + params.starting_log_inv_rate - params.folding_factor.at_round(0)),
            F::ZERO,
        );

        let folded_matrix = dft.dft_batch(mat).to_row_major_matrix();
        let (root, prover_data) = params.mmcs.commit_matrix(folded_matrix);
        challenger.observe(root);
        prover_data
    }

    /// Sample OOD points for both polynomials after both commitments are observed.
    ///
    /// Returns (ood_statement_a, ood_statement_b, ood_answers_a, ood_answers_b).
    fn sample_batch_ood(
        params: &WhirConfig<EF, F, MyMmcs, MyChallenger>,
        challenger: &mut MyChallenger,
        poly_a: &EvaluationsList<F>,
        poly_b: &EvaluationsList<F>,
        num_variables: usize,
    ) -> (EqStatement<EF>, EqStatement<EF>, Vec<EF>, Vec<EF>) {
        let mut ood_a = EqStatement::initialize(num_variables);
        let mut ood_b = EqStatement::initialize(num_variables);
        let mut answers_a = Vec::new();
        let mut answers_b = Vec::new();

        for _ in 0..params.commitment_ood_samples {
            let point = MultilinearPoint::expand_from_univariate(
                challenger.sample_algebra_element(),
                num_variables,
            );
            let eval_a = poly_a.evaluate_hypercube_base(&point);
            let eval_b = poly_b.evaluate_hypercube_base(&point);
            // Both evaluations go into the transcript
            challenger.observe_algebra_element(eval_a);
            challenger.observe_algebra_element(eval_b);

            answers_a.push(eval_a);
            answers_b.push(eval_b);
            ood_a.add_evaluated_constraint(point.clone(), eval_a);
            ood_b.add_evaluated_constraint(point, eval_b);
        }

        (ood_a, ood_b, answers_a, answers_b)
    }

    fn make_batch_test_config(
        num_variables: usize,
        folding_factor: FoldingFactor,
    ) -> (
        WhirConfig<EF, F, MyMmcs, MyChallenger>,
        ProtocolParameters<MyMmcs>,
        SmallRng,
    ) {
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut rng);
        let mmcs = MyMmcs::new(
            MyHash::new(perm.clone()),
            MyCompress::new(perm),
            0,
        );

        let whir_params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor,
            mmcs,
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
        };

        let params = WhirConfig::<EF, F, MyMmcs, MyChallenger>::new(
            num_variables,
            whir_params.clone(),
        );

        (params, whir_params, rng)
    }

    /// Run the full batch prove flow and verify the proof structure is populated.
    fn run_batch_prove(num_variables: usize, folding_factor: FoldingFactor) {
        let (params, whir_params, mut rng) = make_batch_test_config(num_variables, folding_factor);

        let dft = p3_dft::Radix2DFTSmallBatch::<F>::default();

        // Create two random polynomials
        let num_evals = 1 << num_variables;
        let poly_a = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());
        let poly_b = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());

        // Create evaluation points and compute correct claimed values
        let z_a = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
        let z_b = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
        let v_a: EF = poly_a.evaluate_hypercube_base(&z_a);
        let v_b: EF = poly_b.evaluate_hypercube_base(&z_b);

        // Build evaluation statements
        let mut statement_a = EqStatement::initialize(num_variables);
        statement_a.add_evaluated_constraint(z_a, v_a);
        let mut statement_b = EqStatement::initialize(num_variables);
        statement_b.add_evaluated_constraint(z_b, v_b);

        // Set up challenger
        let perm = Perm::new_from_rng_128(&mut rng);
        let mut challenger = MyChallenger::new(perm);

        // Commit both polynomials (roots observed in transcript)
        let prover_data_a = raw_commit(&params, &dft, &poly_a, &mut challenger);
        let prover_data_b = raw_commit(&params, &dft, &poly_b, &mut challenger);

        // Sample OOD for both polynomials
        let (ood_a, ood_b, answers_a, answers_b) =
            sample_batch_ood(&params, &mut challenger, &poly_a, &poly_b, num_variables);

        // Create batch proof structure
        let mut proof = BatchWhirProof {
            initial_ood_answers: [answers_a, answers_b],
            selector_sumcheck: SumcheckData::default(),
            inner_proof: WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(
                &whir_params,
                num_variables,
            ),
        };

        // Run batch prove
        let prover = Prover(&params);
        prover
            .batch_prove(
                &dft,
                &mut proof,
                &mut challenger,
                prover_data_a,
                prover_data_b,
                &poly_a,
                &poly_b,
                &statement_a,
                &statement_b,
                &ood_a,
                &ood_b,
            )
            .unwrap();

        // Verify proof structure is populated
        assert!(proof.inner_proof.final_poly.is_some());
        assert_eq!(proof.selector_sumcheck.polynomial_evaluations().len(), 1);
    }

    #[test]
    fn test_batch_prove_folding_4() {
        run_batch_prove(10, FoldingFactor::Constant(4));
    }

    #[test]
    fn test_batch_prove_folding_2() {
        run_batch_prove(8, FoldingFactor::Constant(2));
    }

    #[test]
    fn test_batch_prove_folding_3() {
        run_batch_prove(9, FoldingFactor::Constant(3));
    }

    #[test]
    fn test_batch_prove_mixed_folding() {
        run_batch_prove(10, FoldingFactor::ConstantFromSecondRound(4, 2));
    }

    #[test]
    fn test_selector_round_correctness() {
        // Test that the selector round produces correct folded polynomial
        let (params, _whir_params, mut rng) = make_batch_test_config(6, FoldingFactor::Constant(3));

        let num_variables = 6;
        let num_evals = 1 << num_variables;
        let poly_a = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());
        let poly_b = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());

        let z_a = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
        let z_b = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
        let v_a: EF = poly_a.evaluate_hypercube_base(&z_a);
        let v_b: EF = poly_b.evaluate_hypercube_base(&z_b);

        let perm = Perm::new_from_rng_128(&mut rng);
        let mut challenger = MyChallenger::new(perm);
        let alpha: EF = challenger.sample_algebra_element();

        let mut selector_data: SumcheckData<F, EF> = SumcheckData::default();

        let prover = Prover(&params);
        let (sumcheck_prover, r_0) = prover.selector_round(
            &mut selector_data,
            &mut challenger,
            &poly_a,
            &poly_b,
            &z_a,
            &z_b,
            v_a,
            v_b,
            alpha,
        );

        // Verify: g should equal r_0·f_a + (1-r_0)·f_b at every point
        let g = sumcheck_prover.evals();
        let one_minus_r0 = EF::ONE - r_0;
        for i in 0..num_evals {
            let expected =
                r_0 * EF::from(poly_a.as_slice()[i]) + one_minus_r0 * EF::from(poly_b.as_slice()[i]);
            assert_eq!(g.as_slice()[i], expected, "g mismatch at index {i}");
        }

        // Verify: selector sumcheck stored exactly 1 round of data
        assert_eq!(selector_data.polynomial_evaluations().len(), 1);

        // Verify: c0 == α·v_b (the h(0) value)
        let [c0, _c2] = selector_data.polynomial_evaluations()[0];
        assert_eq!(c0, alpha * v_b);
    }

    #[test]
    fn test_fold_ood_constraints() {
        let num_variables = 4;
        let r_0 = EF::from_u64(7);

        let point = MultilinearPoint::new(vec![
            EF::from_u64(1),
            EF::from_u64(2),
            EF::from_u64(3),
            EF::from_u64(4),
        ]);

        let v_a = EF::from_u64(10);
        let v_b = EF::from_u64(20);

        let mut ood_a = EqStatement::initialize(num_variables);
        ood_a.add_evaluated_constraint(point.clone(), v_a);
        let mut ood_b = EqStatement::initialize(num_variables);
        ood_b.add_evaluated_constraint(point, v_b);

        let folded = fold_ood_constraints(&ood_a, &ood_b, r_0);

        assert_eq!(folded.len(), 1);
        let (_, &folded_val) = folded.iter().next().unwrap();
        let expected = r_0 * v_a + (EF::ONE - r_0) * v_b;
        assert_eq!(folded_val, expected);
    }

    #[cfg(not(debug_assertions))]
    use std::time::Instant;
    #[cfg(not(debug_assertions))]
    use crate::{
        parameters::SumcheckStrategy,
        whir::committer::writer::CommitmentWriter,
    };

    /// Run a single-polynomial commit + prove cycle. Returns elapsed time.
    #[cfg(not(debug_assertions))]
    #[allow(clippy::too_many_arguments)]
    fn single_poly_prove(
        params: &WhirConfig<EF, F, MyMmcs, MyChallenger>,
        whir_params: &ProtocolParameters<MyMmcs>,
        dft: &p3_dft::Radix2DFTSmallBatch<F>,
        poly: EvaluationsList<F>,
        num_variables: usize,
        num_points: usize,
        rng: &mut SmallRng,
    ) -> std::time::Duration {
        let mut statement = params.initial_statement(poly, SumcheckStrategy::default());
        for _ in 0..num_points {
            let point = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
            let _ = statement.evaluate(&point);
        }

        let perm = Perm::new_from_rng_128(rng);
        let mut challenger = MyChallenger::new(perm);

        let committer = CommitmentWriter::new(params);
        let mut proof =
            WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(whir_params, num_variables);

        let start = Instant::now();

        let prover_data = committer
            .commit(dft, &mut proof, &mut challenger, &mut statement)
            .unwrap();

        let prover = Prover(params);
        prover
            .prove(dft, &mut proof, &mut challenger, &statement, prover_data)
            .unwrap();

        start.elapsed()
    }

    /// Run a batch prove cycle for two polynomials. Returns elapsed time.
    #[cfg(not(debug_assertions))]
    #[allow(clippy::too_many_arguments)]
    fn batch_poly_prove(
        params: &WhirConfig<EF, F, MyMmcs, MyChallenger>,
        whir_params: &ProtocolParameters<MyMmcs>,
        dft: &p3_dft::Radix2DFTSmallBatch<F>,
        poly_a: &EvaluationsList<F>,
        poly_b: &EvaluationsList<F>,
        num_variables: usize,
        rng: &mut SmallRng,
    ) -> std::time::Duration {
        let z_a = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
        let z_b = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
        let v_a: EF = poly_a.evaluate_hypercube_base(&z_a);
        let v_b: EF = poly_b.evaluate_hypercube_base(&z_b);

        let mut statement_a = EqStatement::initialize(num_variables);
        statement_a.add_evaluated_constraint(z_a, v_a);
        let mut statement_b = EqStatement::initialize(num_variables);
        statement_b.add_evaluated_constraint(z_b, v_b);

        let perm = Perm::new_from_rng_128(rng);
        let mut challenger = MyChallenger::new(perm);

        let start = Instant::now();

        let prover_data_a = raw_commit(params, dft, poly_a, &mut challenger);
        let prover_data_b = raw_commit(params, dft, poly_b, &mut challenger);

        let (ood_a, ood_b, answers_a, answers_b) =
            sample_batch_ood(params, &mut challenger, poly_a, poly_b, num_variables);

        let mut proof = BatchWhirProof {
            initial_ood_answers: [answers_a, answers_b],
            selector_sumcheck: SumcheckData::default(),
            inner_proof: WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(
                whir_params,
                num_variables,
            ),
        };

        let prover = Prover(params);
        prover
            .batch_prove(
                dft,
                &mut proof,
                &mut challenger,
                prover_data_a,
                prover_data_b,
                poly_a,
                poly_b,
                &statement_a,
                &statement_b,
                &ood_a,
                &ood_b,
            )
            .unwrap();

        start.elapsed()
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_batch_vs_separate_performance() {
        const WARMUP: usize = 2;
        const ITERS: usize = 5;

        let num_variables = 16;
        let folding_factor = FoldingFactor::Constant(4);
        let (params, whir_params, mut rng) =
            make_batch_test_config(num_variables, folding_factor);
        let dft = p3_dft::Radix2DFTSmallBatch::<F>::default();

        let num_evals = 1 << num_variables;
        let poly_a = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());
        let poly_b = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());

        // Warmup
        for _ in 0..WARMUP {
            single_poly_prove(
                &params, &whir_params, &dft, poly_a.clone(), num_variables, 1, &mut rng,
            );
            batch_poly_prove(
                &params, &whir_params, &dft, &poly_a, &poly_b, num_variables, &mut rng,
            );
        }

        // Benchmark: two separate proofs
        let mut separate_total = std::time::Duration::ZERO;
        for _ in 0..ITERS {
            let t1 = single_poly_prove(
                &params, &whir_params, &dft, poly_a.clone(), num_variables, 1, &mut rng,
            );
            let t2 = single_poly_prove(
                &params, &whir_params, &dft, poly_b.clone(), num_variables, 1, &mut rng,
            );
            separate_total += t1 + t2;
        }

        // Benchmark: one batch proof
        let mut batch_total = std::time::Duration::ZERO;
        for _ in 0..ITERS {
            let t = batch_poly_prove(
                &params, &whir_params, &dft, &poly_a, &poly_b, num_variables, &mut rng,
            );
            batch_total += t;
        }

        let separate_avg = separate_total / ITERS as u32;
        let batch_avg = batch_total / ITERS as u32;
        let speedup = separate_avg.as_secs_f64() / batch_avg.as_secs_f64();

        std::eprintln!();
        std::eprintln!("=== Batch vs Separate Performance ({num_variables} variables, {ITERS} iterations) ===");
        std::eprintln!("  Two separate proofs: {separate_avg:?} avg");
        std::eprintln!("  One batch proof:     {batch_avg:?} avg");
        std::eprintln!("  Speedup:             {speedup:.2}x");
        std::eprintln!();

        // The batch proof should be faster than two separate proofs.
        // At minimum it should be faster than 2x (since it avoids one full
        // protocol execution), but even ~1.1x would show the optimization works.
        // We use a lenient threshold since this runs in debug mode.
        assert!(
            batch_avg < separate_avg,
            "Batch proof ({batch_avg:?}) should be faster than two separate proofs ({separate_avg:?})"
        );
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_batch_vs_separate_performance_large() {
        const WARMUP: usize = 1;
        const ITERS: usize = 3;

        let num_variables = 20;
        let folding_factor = FoldingFactor::Constant(4);
        let (params, whir_params, mut rng) =
            make_batch_test_config(num_variables, folding_factor);
        let dft = p3_dft::Radix2DFTSmallBatch::<F>::default();

        let num_evals = 1 << num_variables;
        let poly_a = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());
        let poly_b = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());

        for _ in 0..WARMUP {
            single_poly_prove(
                &params, &whir_params, &dft, poly_a.clone(), num_variables, 1, &mut rng,
            );
        }

        let mut separate_total = std::time::Duration::ZERO;
        for _ in 0..ITERS {
            let t1 = single_poly_prove(
                &params, &whir_params, &dft, poly_a.clone(), num_variables, 1, &mut rng,
            );
            let t2 = single_poly_prove(
                &params, &whir_params, &dft, poly_b.clone(), num_variables, 1, &mut rng,
            );
            separate_total += t1 + t2;
        }

        let mut batch_total = std::time::Duration::ZERO;
        for _ in 0..ITERS {
            let t = batch_poly_prove(
                &params, &whir_params, &dft, &poly_a, &poly_b, num_variables, &mut rng,
            );
            batch_total += t;
        }

        let separate_avg = separate_total / ITERS as u32;
        let batch_avg = batch_total / ITERS as u32;
        let speedup = separate_avg.as_secs_f64() / batch_avg.as_secs_f64();

        std::eprintln!();
        std::eprintln!("=== Batch vs Separate Performance ({num_variables} variables, {ITERS} iterations) ===");
        std::eprintln!("  Two separate proofs: {separate_avg:?} avg");
        std::eprintln!("  One batch proof:     {batch_avg:?} avg");
        std::eprintln!("  Speedup:             {speedup:.2}x");
        std::eprintln!();

        assert!(
            batch_avg < separate_avg,
            "Batch proof ({batch_avg:?}) should be faster than two separate proofs ({separate_avg:?})"
        );
    }
}
