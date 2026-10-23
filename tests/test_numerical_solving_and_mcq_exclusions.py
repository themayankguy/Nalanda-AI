"""Tests for parametric numerical problem solving, notes topic consolidation, and strict MCQ numerical exclusion."""
import pytest
from backend.services.atml.numerical_solver_service import NumericalSolverService
from backend.services.atml.quiz_generator import (
    clean_academic_concept,
    is_valid_definition,
    validate_quiz_question,
    QuizQuestion,
)
from backend.services.atml.notes_generator import NotesGenerator, SubtopicUnit


class TestNumericalSolver:
    def test_default_autoencoder_numerical_solution(self):
        sol = NumericalSolverService.solve_autoencoder_numerical()
        assert sol["input_x"] == [1.0, 2.0, 3.0, 4.0]
        assert sol["kernel_w"] == [0.5, -0.5]
        assert sol["conv_output"] == [-0.5, -0.5, -0.5]
        assert sol["latent_z"] == [0.0, -0.75]
        assert sol["decoder_fc_output"] == [0.0, -0.75, -0.375]
        assert sol["reconstructed"] == [0.0, -0.375, 0.1875, 0.1875]
        assert sol["mse_loss"] == 7.2715

    def test_custom_values_recalculation(self):
        # Alter input vector x and kernel weights
        custom_x = [2.0, 4.0, 6.0, 8.0]
        custom_w = [1.0, -1.0]
        sol = NumericalSolverService.solve_autoencoder_numerical(x=custom_x, w_conv=custom_w)

        assert sol["input_x"] == custom_x
        assert sol["kernel_w"] == custom_w
        # Window 1: 1*2 - 1*4 = -2
        # Window 2: 1*4 - 1*6 = -2
        # Window 3: 1*6 - 1*8 = -2
        assert sol["conv_output"] == [-2.0, -2.0, -2.0]
        # Latent z1 = 1*(-2) + 0 + (-1)*(-2) = 0
        # Latent z2 = 0.5*(-2) + 0.5*(-2) + 0.5*(-2) = -3
        assert sol["latent_z"] == [0.0, -3.0]
        assert "Comprehensive Solved Numerical" in NumericalSolverService.format_solution_markdown(sol)

    def test_extract_custom_values_variations(self):
        queries = [
            "What if input is [2, 3, 4, 5]?",
            "what if input is [2.0, 3.0, 4.0, 5.0] and kernel is [0.8, -0.2]?",
            "solve the autoencoder numerical with x = [5, 6, 7, 8] and stride: 2",
            "what if input becomes [10, 20, 30, 40]?",
            "recalculate loss when x is 2, 4, 6, 8",
        ]
        val1 = NumericalSolverService.extract_custom_values(queries[0])
        assert val1.get("x") == [2.0, 3.0, 4.0, 5.0]

        val2 = NumericalSolverService.extract_custom_values(queries[1])
        assert val2.get("x") == [2.0, 3.0, 4.0, 5.0]
        assert val2.get("w_conv") == [0.8, -0.2]

        val3 = NumericalSolverService.extract_custom_values(queries[2])
        assert val3.get("x") == [5.0, 6.0, 7.0, 8.0]
        assert val3.get("stride") == 2

        val4 = NumericalSolverService.extract_custom_values(queries[3])
        assert val4.get("x") == [10.0, 20.0, 30.0, 40.0]

        val5 = NumericalSolverService.extract_custom_values(queries[4])
        assert val5.get("x") == [2.0, 4.0, 6.0, 8.0]


class TestMCQStrictNumericalExclusion:
    def test_clean_academic_concept_rejects_numericals(self):
        # Numerical calculation terms must be rejected
        assert clean_academic_concept("W_conv") is None
        assert clean_academic_concept("Conv Output") is None
        assert clean_academic_concept("Latent Vector Z") is None
        assert clean_academic_concept("MSE Loss") is None
        assert clean_academic_concept("Step 1: Convolution") is None
        assert clean_academic_concept("Window 1") is None
        assert clean_academic_concept("Transpose Conv") is None
        assert clean_academic_concept("Reconstruction Loss") is None
        assert clean_academic_concept("Solved Numerical") is None

        # Genuine conceptual terms must still be accepted
        assert clean_academic_concept("Autoencoder") == "Autoencoder"
        assert clean_academic_concept("Dimensionality Reduction") == "Dimensionality Reduction"
        assert clean_academic_concept("Containerization") == "Containerization"

    def test_is_valid_definition_rejects_math_arithmetic(self):
        bad_math_definitions = [
            "w_conv = [0.5, -0.5] with input vector [1.0, 2.0, 3.0, 4.0]",
            "0.5 · 1.0 + (-0.5) · 2.0 = 0.5 + (-1.0) = -0.5",
            "conv output = [-0.5, -0.5, -0.5] after sliding window",
            "MSE loss is 7.2586 calculated over four output elements",
        ]
        for bad_def in bad_math_definitions:
            assert not is_valid_definition(bad_def), f"Failed to reject: {bad_def}"

        good_def = "An autoencoder is an artificial neural network trained to reconstruct input data at its output layer through an unsupervised bottleneck representation."
        assert is_valid_definition(good_def)

    def test_validate_quiz_question_rejects_numerical_stem_and_options(self):
        bad_q = QuizQuestion(
            id="q_num_1",
            question="What is the MSE loss when input is [1, 2, 3, 4] and kernel is [0.5, -0.5]?",
            options=["A) 7.2715", "B) 5.1234", "C) 3.4421", "D) 1.2980"],
            correct_option_index=0,
            correct_answer="7.2715",
            explanation="The MSE loss is 7.2715 after calculating transpose convolution.",
            source_document="lecture.pdf",
            page_number=4,
            section_title="Reconstruction Loss",
            concept="Reconstruction Loss",
        )
        is_valid, failures = validate_quiz_question(bad_q)
        assert not is_valid
        assert any("Numerical" in f for f in failures)


class TestNotesNumericalConsolidation:
    def test_merges_multi_slide_numerical_stages_into_one_subtopic(self):
        service = NotesGenerator()
        subtopics = [
            SubtopicUnit(
                title="Step 1: Encoder Setup",
                doc_name="Autoencoder_Numerical.pdf",
                page_start=1,
                page_end=1,
                definition="Step 1 sets up the 1D convolution encoder.",
                explanation="Slide 1 explains 1D convolution with kernel [0.5, -0.5].",
            ),
            SubtopicUnit(
                title="Step 2: Latent Layer",
                doc_name="Autoencoder_Numerical.pdf",
                page_start=2,
                page_end=2,
                definition="Step 2 computes the latent vector z.",
                explanation="Slide 2 multiplies feature map with W_fc.",
            ),
            SubtopicUnit(
                title="Step 3: Decoder Setup",
                doc_name="Autoencoder_Numerical.pdf",
                page_start=3,
                page_end=3,
                definition="Step 3 reconstructs decoder feature map.",
                explanation="Slide 3 performs transpose convolution.",
            ),
            SubtopicUnit(
                title="Step 4: Reconstruction Loss",
                doc_name="Autoencoder_Numerical.pdf",
                page_start=4,
                page_end=4,
                definition="Step 4 calculates the MSE reconstruction loss.",
                explanation="Slide 4 calculates final MSE loss.",
            ),
        ]

        consolidated = service._validate_and_sanitize_subtopics(
            subtopics, doc_name="Autoencoder_Numerical.pdf", total_pages=4
        )

        # Must be consolidated under a single unified topic, not 4 separate topics
        assert len(consolidated) == 1
        assert "Numerical" in consolidated[0].title
        assert consolidated[0].page_start == 1
        assert consolidated[0].page_end == 4
        assert "Encoder Convolution Layer Calculation" in consolidated[0].explanation
        assert "Latent Bottleneck Projection" in consolidated[0].explanation
        assert "Transpose Convolution Upsampling" in consolidated[0].explanation
        assert "Reconstruction Loss Evaluation" in consolidated[0].explanation
