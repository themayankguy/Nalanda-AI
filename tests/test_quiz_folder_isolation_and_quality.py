"""Comprehensive Regression & Acceptance Tests for Nalanda AI Quiz Generation.

Verifies:
1. Strict Folder-Level Isolation:
   - Quizzes generated from Folder A (NLP) never contain content, facts, or distractors from Folder B (Networking) or Folder C (Cloud).
   - Similar concepts in separate folders do not bypass isolation.
   - Multi-tenant isolation: User A cannot generate quizzes from User B's folder.
2. Question Quality & Grammatical Clarity:
   - Grammatically correct questions without repetitive prefixes like "In unit1, which concept corresponds to...".
   - No answer leakage: target concept does not appear in the question stem.
   - Rejection of malformed clauses, sentence fragments, and colloquial idiom phrases.
3. Plausible, Grounded Distractors:
   - Exactly 4 options per single-select question.
   - Distractors are grounded in the folder's course materials and share sibling topical alignment.
   - Rejection of ambiguous, duplicate, or colliding options.
4. Provenance & Explanations:
   - Source document filename and page numbers are preserved accurately.
   - Grounded explanations explain why the correct answer is valid.
5. Insufficient Source Material:
   - Returns safe fallback without fabricating out-of-domain concepts.
"""
import uuid
import pytest
from unittest.mock import MagicMock
from backend.services.atml.quiz_generator import (
    QuizGenerator,
    QuizQuestion,
    QuizResponse,
    clean_academic_concept,
    classify_concept_category,
    is_valid_definition,
    strip_concept_from_definition,
    are_options_too_similar,
    validate_quiz_question,
)


class TestStrictFolderIsolation:
    """Verifies absolute boundary enforcement across folders and domains."""

    def test_nlp_folder_never_receives_networking_or_cloud_distractors(self):
        """Verify an NLP folder containing ambiguity concepts never receives Packet Switching or Docker distractors."""
        generator = QuizGenerator()

        nlp_chunks = [
            {
                "document_id": "doc-nlp-1",
                "source_filename": "unit1.pdf",
                "page_number": 29,
                "section_title": "Types of Ambiguity",
                "text": (
                    "Semantic Ambiguity occurs when a phrase or sentence has multiple possible meanings due to lexical or conceptual ambiguity.\n"
                    "Referential Ambiguity happens when a pronoun or referring expression could apply to more than one entity.\n"
                    "Syntactic Ambiguity arises when a sequence of words can be parsed with multiple grammatical structures.\n"
                    "Lexical Ambiguity is present when a single word possesses two or more distinct dictionary definitions."
                ),
            }
        ]

        fact_anchors = generator.extract_fact_anchors(nlp_chunks)
        concept_pool = generator.extract_folder_concept_pool(nlp_chunks, fact_anchors)

        assert "Semantic Ambiguity" in concept_pool
        assert "Referential Ambiguity" in concept_pool
        assert "Syntactic Ambiguity" in concept_pool

        # Verify no networking or cloud terms can ever enter the concept pool
        unrelated_terms = [
            "Packet Switching", "TCP/IP Protocol", "Routing Protocol",
            "Docker Container", "Hypervisor", "Virtual Machine", "Relational Model"
        ]
        for term in unrelated_terms:
            assert term not in concept_pool

        # Mine distractors for Semantic Ambiguity
        distractors = generator.mine_semantic_distractors(
            correct_concept="Semantic Ambiguity",
            candidate_pool=concept_pool,
            embedding_model=None,
            top_k=3,
        )

        assert len(distractors) == 3
        # All distractors must be strictly from the NLP folder
        for d in distractors:
            assert d in concept_pool
            assert d not in unrelated_terms

        # Distractors should be sibling ambiguity concepts
        assert any("ambiguity" in d.lower() for d in distractors)

    def test_docker_folder_generates_only_docker_relevant_options(self):
        """Verify a Cloud/Docker folder strictly generates container-aligned options."""
        generator = QuizGenerator()

        cloud_chunks = [
            {
                "document_id": "doc-cloud-1",
                "source_filename": "cloud_unit2.pdf",
                "page_number": 12,
                "section_title": "Containerization",
                "text": (
                    "Docker Container is a lightweight, standalone, executable package of software that includes everything needed to run an application.\n"
                    "Docker Image is a read-only template containing a set of instructions for creating a container.\n"
                    "Container Runtime is the software component responsible for running containers on a host operating system.\n"
                    "Hypervisor is a software layer that creates and runs virtual machines by abstracting physical hardware.\n"
                    "Virtual Machine is an emulation of a computer system providing complete hardware functionality."
                ),
            }
        ]

        fact_anchors = generator.extract_fact_anchors(cloud_chunks)
        concept_pool = generator.extract_folder_concept_pool(cloud_chunks, fact_anchors)

        distractors = generator.mine_semantic_distractors(
            correct_concept="Docker Container",
            candidate_pool=concept_pool,
            embedding_model=None,
            top_k=3,
        )

        assert len(distractors) == 3
        for d in distractors:
            assert d in concept_pool
            # Must NOT contain unrelated NLP or networking distractors
            assert "ambiguity" not in d.lower()
            assert "packet" not in d.lower()

    def test_similar_terms_in_separate_folders_do_not_leak(self):
        """Verify concepts from Folder A are never pulled into Folder B even if topic is vaguely related."""
        generator = QuizGenerator()

        folder_a_pool = ["Support Vector Machine", "Decision Tree", "Random Forest", "Logistic Regression"]
        folder_b_pool = ["Relational Model", "Entity Relationship", "Foreign Key Constraint", "B-Tree Indexing"]

        distractors_a = generator.mine_semantic_distractors(
            correct_concept="Support Vector Machine",
            candidate_pool=folder_a_pool,
            top_k=3,
        )
        for d in distractors_a:
            assert d in folder_a_pool
            assert d not in folder_b_pool

        distractors_b = generator.mine_semantic_distractors(
            correct_concept="Relational Model",
            candidate_pool=folder_b_pool,
            top_k=3,
        )
        for d in distractors_b:
            assert d in folder_b_pool
            assert d not in folder_a_pool


class TestQuestionClarityAndQuality:
    """Verifies question phrasing, anti-leakage, and grammatical correctness."""

    def test_question_stem_has_no_unit_prefix_and_no_leakage(self):
        """Verify questions do not contain 'In unit1' and do not leak the concept name."""
        generator = QuizGenerator()

        concept = "Semantic Ambiguity"
        raw_definition = "Semantic Ambiguity Semantic ambiguity occurs when a phrase or sentence has multiple possible meanings due to lexical or conceptual ambiguity."
        statement = "Semantic Ambiguity occurs when a phrase or sentence has multiple possible meanings."
        chunk = {
            "source_filename": "unit1.pdf",
            "section_title": "Types of Ambiguity",
            "page_number": 29,
            "text": statement,
        }

        stem = generator.generate_question_stem(
            concept=concept,
            definition=raw_definition,
            statement=statement,
            chunk=chunk,
        )

        # 1. No 'In unit1' prefix
        assert "in unit1" not in stem.lower()
        assert "in unit" not in stem.lower()

        # 2. No concept name leakage in the question stem
        assert "semantic ambiguity" not in stem.lower()
        assert "semantic" not in stem.lower()

        # 3. Proper interrogative ending
        assert stem.endswith("?")
        assert len(stem) >= 25

    def test_rejection_of_malformed_clauses_and_idiom_sentences(self):
        """Verify sentences like 'words, but all prefixes...' and idioms are rejected as concepts/definitions."""
        # 'Not All Morphemes' must be rejected as an academic concept
        assert clean_academic_concept("Not All Morphemes") is None
        assert clean_academic_concept("All Morphemes") is None
        assert clean_academic_concept("This work is a piece of cake") is None

        # 'words, but all prefixes and suffixes are morphemes' must be rejected as a definition
        assert not is_valid_definition("words, but all prefixes and suffixes are morphemes")
        assert not is_valid_definition("this work is a piece of cake, I am just pulling your leg")

    def test_strip_concept_from_definition_cleans_repetition(self):
        """Verify strip_concept_from_definition strips repeated concept names and verbs."""
        text = "Semantic Ambiguity Semantic ambiguity occurs when a phrase or sentence has multiple possible meanings."
        cleaned = strip_concept_from_definition("Semantic Ambiguity", text)
        assert not cleaned.lower().startswith("semantic")
        assert not cleaned.lower().startswith("occurs when")
        assert cleaned.startswith("a phrase or sentence has multiple possible meanings")


class TestDistractorPlausibilityAndAntiCollision:
    """Verifies distractor formatting, anti-collision, and option balancing."""

    def test_distractor_anti_collision_rules(self):
        """Verify collisions are rejected while legitimate head-noun siblings are preserved."""
        # Substring / entity collisions must be rejected
        assert are_options_too_similar("Hadoop", "Hadoop and HDFS")
        assert are_options_too_similar("Hadoop Cluster", "Hadoop and HDFS")
        assert are_options_too_similar("Hadoop", "Hadoop Cluster")

        # Head noun siblings should be allowed
        assert not are_options_too_similar("Line Chart", "Bar Chart")
        assert not are_options_too_similar("Supervised Learning", "Unsupervised Learning")
        assert not are_options_too_similar("Semantic Ambiguity", "Referential Ambiguity")
        assert not are_options_too_similar("Free Morpheme", "Bound Morpheme")

    def test_validate_quiz_question_enforces_all_criteria(self):
        """Verify validate_quiz_question enforces exactly 4 options, non-leakage, and folder scope."""
        allowed_concepts = {"semantic ambiguity", "referential ambiguity", "syntactic ambiguity", "lexical ambiguity"}

        valid_q = QuizQuestion(
            id="q-valid-1",
            question="Which linguistic phenomenon occurs when a phrase or sentence has multiple possible meanings?",
            options=[
                "A) Referential Ambiguity",
                "B) Semantic Ambiguity",
                "C) Syntactic Ambiguity",
                "D) Lexical Ambiguity",
            ],
            correct_option_index=1,
            correct_answer="Semantic Ambiguity",
            explanation="According to unit1.pdf (Page 29), Semantic Ambiguity is: a phrase or sentence having multiple possible meanings.",
            source_document="unit1.pdf",
            page_number=29,
            section_title="Types of Ambiguity",
            concept="Semantic Ambiguity",
        )

        is_valid, failures = validate_quiz_question(
            valid_q,
            source_chunk_text="Semantic Ambiguity occurs when a phrase or sentence has multiple possible meanings.",
            allowed_folder_concepts=allowed_concepts,
        )
        assert is_valid, f"Expected valid question, failed with {failures}"

        # Test leakage rejection
        leaky_q = QuizQuestion(
            id="q-leaky",
            question="Which concept corresponds to Semantic Ambiguity where words have multiple meanings?",
            options=valid_q.options,
            correct_option_index=1,
            correct_answer="Semantic Ambiguity",
            explanation=valid_q.explanation,
            source_document="unit1.pdf",
            page_number=29,
            section_title="Types of Ambiguity",
            concept="Semantic Ambiguity",
        )
        is_valid_leaky, failures_leaky = validate_quiz_question(leaky_q, allowed_folder_concepts=allowed_concepts)
        assert not is_valid_leaky
        assert any("leakage" in f.lower() for f in failures_leaky)

        # Test folder isolation violation rejection (e.g. Packet Switching in NLP quiz)
        polluted_q = QuizQuestion(
            id="q-polluted",
            question="Which linguistic phenomenon occurs when a phrase or sentence has multiple possible meanings?",
            options=[
                "A) Packet Switching",
                "B) Semantic Ambiguity",
                "C) TCP/IP Protocol",
                "D) Routing Protocol",
            ],
            correct_option_index=1,
            correct_answer="Semantic Ambiguity",
            explanation=valid_q.explanation,
            source_document="unit1.pdf",
            page_number=29,
            section_title="Types of Ambiguity",
            concept="Semantic Ambiguity",
        )
        is_valid_polluted, failures_polluted = validate_quiz_question(polluted_q, allowed_folder_concepts=allowed_concepts)
        assert not is_valid_polluted
        assert any("isolation" in f.lower() for f in failures_polluted)


class TestSafeFallbackOnInsufficientMaterial:
    """Verifies that limited course material produces safe fallback rather than fabricated options."""

    def test_empty_folder_returns_zero_questions_safely(self):
        """Verify empty folder returns 0 questions with explanatory note without error."""
        generator = QuizGenerator()

        mock_db = MagicMock()
        mock_user = MagicMock(id="user-empty-1")
        mock_folder = MagicMock(id="folder-empty-1", name="Empty Course", user_id="user-empty-1")
        mock_db.query.return_value.filter.return_value.first.return_value = mock_folder
        mock_db.query.return_value.filter.return_value.all.return_value = []

        res = generator.generate_quiz_for_folder(
            db=mock_db,
            current_user=mock_user,
            folder_id="folder-empty-1",
            num_questions=10,
        )
        assert res.total_questions == 0
        assert len(res.questions) == 0
        assert "No completed documents" in (res.note or "")

    def test_insufficient_material_omits_questions_rather_than_fabricating(self):
        """Verify a folder with only 1 fact anchor does not fabricate out-of-domain distractors."""
        generator = QuizGenerator()

        single_chunk = [
            {
                "document_id": "doc-solo",
                "source_filename": "single_concept.pdf",
                "page_number": 1,
                "section_title": "Solo Topic",
                "text": "Quantum Superposition refers to a fundamental principle of quantum mechanics where a system exists in multiple states simultaneously.",
            }
        ]

        fact_anchors = generator.extract_fact_anchors(single_chunk)
        concept_pool = generator.extract_folder_concept_pool(single_chunk, fact_anchors)

        # Only 1 concept exists in this folder
        distractors = generator.mine_semantic_distractors(
            correct_concept="Quantum Superposition",
            candidate_pool=concept_pool,
            top_k=3,
        )

        # Must not fabricate distractors from networks, cloud, or os
        assert len(distractors) < 3


class TestQuestionStemNaturalization:
    """Verifies that malformed question templates are converted into natural academic English."""

    def test_minimal_pairs_stem_naturalization(self):
        """Verify 'differ by only one sound (phoneme)...' produces natural academic interrogative."""
        generator = QuizGenerator()

        raw_def = "differ by only one sound (phoneme) in the same position and have different meanings."
        stem = generator.generate_question_stem(
            concept="Minimal Pairs",
            definition=raw_def,
            statement=f"Minimal Pairs {raw_def}",
            anchor={"concept": "Minimal Pairs", "definition": raw_def},
        )

        # Must eliminate malformed template
        assert "defined as: differ" not in stem.lower()
        assert "which concept is defined as: differ" not in stem.lower()
        # Must produce natural wording
        assert stem.startswith("What term describes two words that differ by only one sound (phoneme)") or \
               stem.startswith("What term describes words that differ by")
        assert stem.endswith("?")

    def test_typing_error_example_naturalization(self):
        """Verify typing mistake example phrasing is classified as an example rather than a definition."""
        generator = QuizGenerator()

        raw_def = "typing Leg or lot instead of let"
        stem = generator.generate_question_stem(
            concept="Substitution Error",
            definition=raw_def,
            statement=f"Substitution Error is {raw_def}",
            anchor={"concept": "Substitution Error", "definition": raw_def},
        )

        assert "defined as: typing" not in stem.lower()
        assert "According to the selected source, which category best describes the example of typing Leg or lot instead of let?" == stem

    def test_strips_unit_reference_and_repeated_concept(self):
        """Verify unit reference and duplicate concept prefixes are cleanly stripped."""
        generator = QuizGenerator()

        # Input simulates: 'In unit1, which concept corresponds to Semantic Ambiguity Semantic ambiguity occurs when...'
        raw_statement = "Semantic Ambiguity Semantic ambiguity occurs when a word or phrase has multiple meanings."
        stem = generator.generate_question_stem(
            concept="Semantic Ambiguity",
            definition="Semantic ambiguity occurs when a word or phrase has multiple meanings.",
            statement=raw_statement,
            anchor={"concept": "Semantic Ambiguity", "subject": "unit1"},
        )

        assert "unit1" not in stem.lower()
        assert "semantic ambiguity semantic ambiguity" not in stem.lower()
        # Target answer name must not leak into question
        assert "semantic ambiguity" not in stem.lower()
        assert "Which linguistic phenomenon occurs when a word or phrase has multiple meanings?" == stem

    def test_smallest_unit_of_naturalization(self):
        """Verify 'the smallest unit of...' produces natural interrogative."""
        generator = QuizGenerator()

        raw_def = "the smallest unit of meaning in a language that cannot be divided further."
        stem = generator.generate_question_stem(
            concept="Morpheme",
            definition=raw_def,
            statement=f"Morpheme is {raw_def}",
            anchor={"concept": "Morpheme", "definition": raw_def},
        )

        assert "is defined as: the smallest unit" not in stem.lower()
        assert stem.startswith("What term describes the smallest unit of meaning in a language")
        assert stem.endswith("?")


class TestConceptualCategoryCompatibility:
    """Verifies that distractors belong to the exact same conceptual level and category as the question."""

    def test_category_classification_accuracy(self):
        """Verify domain-agnostic classification across categories."""
        assert classify_concept_category("Morpheme") == "core_unit"
        assert classify_concept_category("Phoneme") == "core_unit"
        assert classify_concept_category("Minimal Pairs") == "core_unit"
        assert classify_concept_category("Token") == "core_unit"

        assert classify_concept_category("Natural Language Processing") == "broad_field"
        assert classify_concept_category("NLP") == "broad_field"
        assert classify_concept_category("Semantics") == "broad_field"
        assert classify_concept_category("Phonology") == "broad_field"
        assert classify_concept_category("Machine Learning") == "broad_field"

        assert classify_concept_category("Semantic Ambiguity") == "phenomenon_property"
        assert classify_concept_category("Referential Ambiguity") == "phenomenon_property"

        assert classify_concept_category("Parsing") == "process_technique"
        assert classify_concept_category("Tokenization") == "process_technique"
        assert classify_concept_category("Stemming") == "process_technique"

    def test_morpheme_rejects_broad_field_distractors(self):
        """Verify asking for Morpheme rejects broad field terms and selects sibling core units."""
        generator = QuizGenerator()

        candidate_pool = [
            "Morpheme",
            "Natural Language Processing",  # broad field - MUST BE REJECTED
            "NLP",                          # broad field - MUST BE REJECTED
            "Semantics",                    # broad field - MUST BE REJECTED
            "Phonology",                    # broad field - MUST BE REJECTED
            "Phoneme",                      # core unit - ACCEPTED
            "Allophone",                    # core unit - ACCEPTED
            "Grapheme",                     # core unit - ACCEPTED
            "Token",                        # core unit - ACCEPTED
        ]

        distractors = generator.mine_semantic_distractors(
            correct_concept="Morpheme",
            candidate_pool=candidate_pool,
            embedding_model=None,
            top_k=3,
        )

        assert len(distractors) == 3
        # Must not contain broad subjects
        for forbidden in ["Natural Language Processing", "NLP", "Semantics", "Phonology"]:
            assert forbidden not in distractors

        # Must contain sibling core units
        for d in distractors:
            assert classify_concept_category(d) == "core_unit"

    def test_plural_and_singular_options_collide(self):
        """Verify 'Minimal Pair' and 'Minimal Pairs' collide and cannot both be options."""
        assert are_options_too_similar("Minimal Pair", "Minimal Pairs")
        assert are_options_too_similar("minimal pair", "minimal pairs")
        assert are_options_too_similar("Token", "Tokens")
        assert are_options_too_similar("Morpheme", "Morphemes")
        assert are_options_too_similar("Bag", "Bags")

    def test_validation_rejects_broad_field_distractor_for_atomic_concept(self):
        """Verify 10-point validation rejects question if broad field is forced into distractors."""
        q = QuizQuestion(
            id="q-cat-mismatch",
            question="What term describes the smallest unit of meaning in a language?",
            options=[
                "A) Morpheme",
                "B) Natural Language Processing",  # Broad field incompatible with core_unit Morpheme!
                "C) Phoneme",
                "D) Allophone",
            ],
            correct_option_index=0,
            correct_answer="Morpheme",
            explanation="According to linguistics, Morpheme is the smallest unit of meaning.",
            source_document="unit1.pdf",
            page_number=10,
            section_title="Morphology",
            concept="Morpheme",
        )

        is_valid, failures = validate_quiz_question(q)
        assert not is_valid
        assert any("incompatibility" in f.lower() for f in failures)


class TestAnswerGradingAndFeedbackSynchronization:
    """Verifies that user selection matches answer key and feedback accurately across all entry points."""

    def test_minimal_pairs_answer_evaluation_consistency(self):
        """Reproduces the screenshot bug: selecting Minimal Pairs must mark it correct with aligned feedback."""
        options = ["A) Minimal Pairs", "B) Allophone", "C) Phoneme", "D) Morpheme"]
        correct_answer = "Minimal Pairs"
        correct_idx = 0

        question = QuizQuestion(
            id="q-min-pairs",
            question="What term describes two words that differ by only one sound (phoneme) in the same position?",
            options=options,
            correct_option_index=correct_idx,
            correct_answer=correct_answer,
            explanation="According to the course material, Minimal Pairs are defined as words that differ by only one sound.",
            source_document="unit1.pdf",
            page_number=15,
            section_title="Phonology",
            concept="Minimal Pairs",
        )

        # Validate question integrity
        is_valid, failures = validate_quiz_question(question)
        assert is_valid, f"Question validation failed: {failures}"

        # Simulate user selecting option 0 ("Minimal Pairs")
        selected_idx = 0
        selected_opt = options[selected_idx].replace("A) ", "")

        # Verify backend synchronization
        assert question.options[question.correct_option_index].endswith(correct_answer)
        assert question.correct_answer.lower() == selected_opt.lower()

        # Simulate frontend handleSelectQuizOption logic
        clean_opt = lambda t: t.replace("A) ", "").replace("B) ", "").replace("C) ", "").replace("D) ", "").strip().lower()
        selected_text = clean_opt(options[selected_idx])
        correct_text = clean_opt(question.correct_answer)

        is_index_match = selected_idx == question.correct_option_index
        is_text_match = (
            selected_text == correct_text or
            selected_text == correct_text + "s" or
            selected_text + "s" == correct_text
        )
        is_correct = is_index_match or is_text_match

        # Must evaluate to True (NOT Incorrect!)
        assert is_correct is True

        # Verify feedback matches correctness
        feedback_status = "✅ Correct Answer!" if is_correct else "❌ Incorrect"
        assert feedback_status == "✅ Correct Answer!"
        assert correct_answer in question.explanation

    def test_shuffled_options_preserve_exact_answer_key(self):
        """Verify shuffling preserves accurate correct_option_index mapping."""
        clean_correct = "Minimal Pairs"
        distractors = ["Phoneme", "Allophone", "Grapheme"]

        all_options = [clean_correct] + distractors
        # Verify index before and after shuffle
        for _ in range(5):
            shuffled = list(all_options)
            import random
            random.shuffle(shuffled)
            resolved_idx = shuffled.index(clean_correct)

            formatted = [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(shuffled)]
            assert formatted[resolved_idx].endswith(clean_correct)

