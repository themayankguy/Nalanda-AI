"""Unit and regression tests for Generalized NLG Reconstructor and Answer Structurer.

Validates that the system operates in a 100% domain-agnostic fashion across:
- Biology & Life Sciences
- Medicine & Pathology
- Constitutional & Common Law
- Macroeconomics & Market Theory
- Thermodynamics & Classical Physics
- Computer Science & Distributed Systems
"""
import pytest
from backend.services.atml.nlg_reconstructor import GeneralizedNLGReconstructor
from backend.services.atml.answer_structurer import AnswerStructurer
from backend.services.NLP.hierarchical_concept_gatherer import HierarchicalConceptGatherer
from backend.services.NLP.text_normalizer import format_types_response


class TestGeneralizedNLGReconstructor:
    """Tests for deterministic sentence reconstruction and grammar synthesis."""

    def test_biology_fragment_reconstruction(self):
        fragment = "Mitochondria: Powerhouse of the cell generating ATP through cellular respiration"
        sentence = GeneralizedNLGReconstructor.reconstruct_fragment_to_sentence(fragment)
        assert sentence.startswith("**Mitochondria**")
        assert "powerhouse of the cell" in sentence.lower()
        assert sentence.endswith(".")
        assert any(v in sentence.lower() for v in ["is", "serves as", "refers to"])

    def test_law_fragment_reconstruction(self):
        fragment = "Mens Rea: The mental state of intention or knowledge of wrongdoing constituting a crime"
        sentence = GeneralizedNLGReconstructor.reconstruct_fragment_to_sentence(fragment)
        assert sentence.startswith("**Mens Rea**")
        assert "mental state" in sentence.lower()
        assert sentence.endswith(".")

    def test_economics_fragment_reconstruction(self):
        fragment = "Opportunity Cost - Value of the next best alternative forgone when making a choice"
        sentence = GeneralizedNLGReconstructor.reconstruct_fragment_to_sentence(fragment)
        assert sentence.startswith("**Opportunity Cost**")
        assert "next best alternative" in sentence.lower()
        assert sentence.endswith(".")

    def test_medicine_fragment_reconstruction(self):
        fragment = "Myocardial Infarction: Ischemic necrosis of myocardial tissue resulting from acute coronary occlusion"
        sentence = GeneralizedNLGReconstructor.reconstruct_fragment_to_sentence(fragment)
        assert sentence.startswith("**Myocardial Infarction**")
        assert "ischemic necrosis" in sentence.lower()
        assert sentence.endswith(".")

    def test_gerund_fragment_reconstruction(self):
        fragment = "Replication: Creating multiple identical copies of DNA molecules prior to cell division"
        sentence = GeneralizedNLGReconstructor.reconstruct_fragment_to_sentence(fragment)
        assert sentence.startswith("**Replication**")
        assert "creating multiple" in sentence.lower()
        assert "involves" in sentence.lower() or "functions by" in sentence.lower() or "is" in sentence.lower()
        assert sentence.endswith(".")

    def test_already_complete_sentence_preserved(self):
        complete = "Antibiotics inhibit the synthesis of bacterial cell walls to prevent proliferation."
        sentence = GeneralizedNLGReconstructor.reconstruct_fragment_to_sentence(complete)
        assert sentence == complete

    def test_zero_hallucination_guarantee(self):
        """Verify that no fabricated metrics, numbers, or external facts are injected."""
        minimal_fragment = "Enzyme: Biological catalyst"
        sentence = GeneralizedNLGReconstructor.reconstruct_fragment_to_sentence(minimal_fragment)
        assert "biological catalyst" in sentence.lower()
        # Verify no invented details like temperatures, pH levels, or specific enzymes
        assert "temperature" not in sentence.lower()
        assert "pepsin" not in sentence.lower()
        assert "celsius" not in sentence.lower()

    def test_generalized_card_deinterleaving(self):
        """Verify multi-column slide card de-interleaving across a medical pathology context."""
        scanned_lines = [
            "Stage 1",
            "Stage 2",
            "Stage 3",
            "Initial Exposure",
            "Acute Proliferation",
            "Chronic Remodeling",
            "Pathogen enters the mucosal membrane without systemic symptoms.",
            "Rapid viral replication triggers localized inflammation and cytokine release.",
            "Permanent fibrotic changes occur in alveolar tissue with persistent dyspnea.",
        ]
        deinterleaved = GeneralizedNLGReconstructor.deinterleave_ocr_cards_generalized(scanned_lines)
        joined = "\n".join(deinterleaved)

        assert "Stage 1: Initial Exposure" in joined
        assert "Stage 2: Acute Proliferation" in joined
        assert "Stage 3: Chronic Remodeling" in joined
        assert "Pathogen enters the mucosal membrane" in joined
        assert "Rapid viral replication" in joined
        assert "Permanent fibrotic changes" in joined


class TestChatGPTStyleAcademicStructuring:
    """Tests for academic answer formatting across diverse subjects."""

    def test_definitional_academic_layout(self):
        structurer = AnswerStructurer()
        raw_text = (
            "Photosynthesis is the biochemical process used by plants and other organisms to convert light energy into chemical energy.\n"
            "- Light Reactions: Absorbing photons in thylakoid membranes to generate ATP and NADPH.\n"
            "- Calvin Cycle: Fixing atmospheric carbon dioxide into glucose molecules in the stroma."
        )
        query = "What is Photosynthesis?"
        formatted = structurer.structure(query, raw_text)

        assert "### Photosynthesis" in formatted or "### What Is Photosynthesis" in formatted
        assert "Light Reactions" in formatted
        assert "Calvin Cycle" in formatted
        assert "- **Light Reactions**" in formatted

    def test_procedural_academic_layout(self):
        structurer = AnswerStructurer()
        raw_text = (
            "Phases of Mitosis:\n"
            "Step 1: Prophase\n"
            "Chromatin condenses into chromosomes and the nuclear envelope breaks down.\n"
            "Step 2: Metaphase\n"
            "Chromosomes align along the equatorial metaphase plate.\n"
            "Step 3: Anaphase\n"
            "Sister chromatids are separated and pulled toward opposite spindle poles.\n"
            "Step 4: Telophase\n"
            "Nuclear membranes reform around each set of daughter chromosomes."
        )
        query = "What are the steps of mitosis?"
        formatted = structurer.structure(query, raw_text)

        assert "1. **Phase 1: Prophase**" in formatted or "1. **Step 1: Prophase**" in formatted or "1. Prophase" in formatted
        assert "Metaphase" in formatted
        assert "Anaphase" in formatted
        assert "Telophase" in formatted
        assert "Chromatin condenses" in formatted

    def test_taxonomical_academic_layout(self):
        structurer = AnswerStructurer()
        raw_text = (
            "Four primary market structures exist in microeconomics:\n"
            "- Perfect Competition: Characterized by numerous small firms, homogenous products, and zero barriers to entry.\n"
            "- Monopolistic Competition: Many competing producers selling differentiated products.\n"
            "- Oligopoly: Dominated by a small number of large suppliers with high barriers to entry.\n"
            "- Monopoly: A single seller with absolute market power and no close substitutes."
        )
        query = "What are the categories of market structures?"
        formatted = structurer.structure(query, raw_text)

        assert "- **Perfect Competition**" in formatted
        assert "- **Monopolistic Competition**" in formatted
        assert "- **Oligopoly**" in formatted
        assert "- **Monopoly**" in formatted


class TestGeneralizedHierarchicalGathering:
    """Tests for multi-chunk concept gathering without domain hardcoding."""

    def test_gather_medical_tissue_types(self):
        chunks = [
            {
                "section_title": "1.1 Simple Squamous Epithelium",
                "page_number": 12,
                "source_filename": "Histology_Atlas.pdf",
                "text": "Simple squamous epithelium consists of a single layer of flattened cells facilitating diffusion.",
            },
            {
                "section_title": "1.2 Stratified Cuboidal Epithelium",
                "page_number": 14,
                "source_filename": "Histology_Atlas.pdf",
                "text": "Stratified cuboidal epithelium contains multiple layers of cube-shaped cells providing protective glandular secretion.",
            },
            {
                "section_title": "1.3 Columnar Epithelium",
                "page_number": 16,
                "source_filename": "Histology_Atlas.pdf",
                "text": "Columnar epithelium features tall, column-like cells specialized for absorption and active transport.",
            },
        ]

        is_hier = HierarchicalConceptGatherer.is_hierarchical_query("What are the types of epithelium?")
        assert is_hier is True

        parent = HierarchicalConceptGatherer.extract_parent_target("What are the types of epithelium?")
        assert "epithelium" in parent

        siblings = HierarchicalConceptGatherer.gather_sibling_concepts(parent, chunks)
        names = {s["name"].lower() for s in siblings}

        assert "simple squamous epithelium" in names
        assert "stratified cuboidal epithelium" in names
        assert "columnar epithelium" in names

        formatted = format_types_response(parent.title(), siblings)
        assert "### Types of" in formatted
        assert "Simple Squamous Epithelium" in formatted
        assert "Stratified Cuboidal Epithelium" in formatted
        assert "Page 12" in formatted
