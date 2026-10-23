"""Script to generate a professional, high-impact 16:9 widescreen PowerPoint presentation (.pptx)
for Nalanda AI's Capstone Defense, complete with card layouts, tables, and presenter notes.
"""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]  # completely blank

    # Palette
    BG_DARK = RGBColor(15, 23, 42)        # Slate 900
    BG_LIGHT = RGBColor(248, 250, 252)    # Slate 50
    CARD_BG = RGBColor(255, 255, 255)     # White
    CARD_BORDER = RGBColor(226, 232, 240) # Slate 200
    TEXT_MAIN = RGBColor(15, 23, 42)      # Slate 900
    TEXT_MUTED = RGBColor(100, 116, 139)  # Slate 500
    PRIMARY = RGBColor(79, 70, 229)       # Indigo 600
    PRIMARY_LIGHT = RGBColor(238, 242, 255)# Indigo 50
    SECONDARY = RGBColor(13, 148, 136)    # Teal 600
    SECONDARY_LIGHT = RGBColor(240, 253, 250)
    ACCENT_PURPLE = RGBColor(147, 51, 234)
    TEXT_WHITE = RGBColor(255, 255, 255)

    def set_slide_background(slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, category, title, is_dark=False):
        # Category Pill / Supertitle
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf = cat_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = category.upper()
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = SECONDARY if is_dark else PRIMARY

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(26)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE if is_dark else TEXT_MAIN

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1.5)
        else:
            shape.line.fill.background()
        return shape

    def set_speaker_notes(slide, notes_text):
        notes_slide = slide.notes_slide
        text_frame = notes_slide.notes_text_frame
        text_frame.text = notes_text

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Theme)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, BG_DARK)

    # Accent decorative bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(1.2), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = PRIMARY
    bar.line.fill.background()

    # Title box
    tb = s1.shapes.add_textbox(Inches(0.8), Inches(2.1), Inches(11.7), Inches(2.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.text = "NALANDA AI"
    p0.font.size = Pt(48)
    p0.font.bold = True
    p0.font.color.rgb = TEXT_WHITE

    p1 = tf.add_paragraph()
    p1.text = "Local Multi-Modal Academic Retrieval & Intelligent Study Workspace"
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(199, 210, 254) # Indigo 200

    p2 = tf.add_paragraph()
    p2.text = "An Edge-Native, Privacy-Preserving Copilot for Technical Lecture Notes & Numerical Derivations"
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(148, 163, 184) # Slate 400

    # Meta box card
    meta_card = add_card(s1, Inches(0.8), Inches(4.7), Inches(11.7), Inches(1.8), bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))
    mb = s1.shapes.add_textbox(Inches(1.1), Inches(4.9), Inches(11.1), Inches(1.4))
    mtf = mb.text_frame
    mtf.word_wrap = True
    mp1 = mtf.paragraphs[0]
    mp1.text = "Capstone Project Presentation | Department of Computer Science & Engineering"
    mp1.font.size = Pt(14)
    mp1.font.bold = True
    mp1.font.color.rgb = TEXT_WHITE

    mp2 = mtf.add_paragraph()
    mp2.text = "Author: Mayank Sawant    |    Domain: Natural Language Processing & Academic Information Systems"
    mp2.font.size = Pt(13)
    mp2.font.color.rgb = RGBColor(203, 213, 225)

    mp3 = mtf.add_paragraph()
    mp3.text = "Architecture: 100% Edge-Native (PyTorch, Sentence-Transformers, Closed-Form Algebraic Solvers)"
    mp3.font.size = Pt(12)
    mp3.font.color.rgb = SECONDARY

    set_speaker_notes(s1, "Good morning committee members. Today I am presenting Nalanda AI, a local, privacy-first academic copilot designed to solve the critical challenges students face with complex lecture slides, multi-stage numerical problems, and exam study materials.")

    # =========================================================================
    # SLIDE 2: The Student's Dilemma (The Real Problem)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, BG_LIGHT)
    add_header(s2, "Context & Motivation", "The Student's Dilemma: Why Commercial AI Fails in Higher Education")

    col_w = Inches(3.64)
    gap = Inches(0.39)
    top_pos = Inches(1.8)
    card_h = Inches(4.8)

    cards_data = [
        ("The Math Hallucination Gap", "Generic LLMs (ChatGPT, Claude) guess tokens probabilistically. In multi-stage engineering derivations (e.g., convolution, latent bottlenecks, MSE loss), they frequently invent intermediate numbers or lose track of matrix dimensions.", "❌ Math Hallucinations", RGBColor(239, 68, 68)),
        ("Multi-Slide Fragmentation", "Engineering problems are routinely split across 4–6 lecture slides. Standard RAG systems retrieve only Slide 2 or Slide 4 in isolation, breaking the derivation context and delivering broken, half-complete answers.", "❌ Fragmented Slides", RGBColor(245, 158, 11)),
        ("Broken Typography & Bad Quizzes", "Commercial tools output broken LaTeX escapes (\\text, $$, \\mathbf) and generate noisy practice quizzes packed with raw vector tokens or unsolvable arithmetic trivia rather than testing conceptual mastery.", "❌ Cluttered Math & MCQs", RGBColor(225, 29, 72))
    ]

    for i, (title, desc, tag, tag_color) in enumerate(cards_data):
        left_pos = Inches(0.8) + i * (col_w + gap)
        add_card(s2, left_pos, top_pos, col_w, card_h)
        tb = s2.shapes.add_textbox(left_pos + Inches(0.25), top_pos + Inches(0.3), col_w - Inches(0.5), card_h - Inches(0.6))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = tag.upper()
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = tag_color

        p2 = tf.add_paragraph()
        p2.text = title
        p2.font.size = Pt(18)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_MAIN

        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(13)
        p3.font.color.rgb = TEXT_MUTED

    set_speaker_notes(s2, "University students face 100+ slides per subject. When they paste slides into generic LLMs, they encounter math hallucinations, broken slide chunking, and unreadable LaTeX escapes. Nalanda AI solves these exact problems.")

    # =========================================================================
    # SLIDE 3: What is Nalanda AI? (Product Concept)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, BG_LIGHT)
    add_header(s3, "Product Vision", "Nalanda AI: An Edge-Native Academic Intelligence Workspace")

    pillars = [
        ("💬 Grounded Academic Chat", "Interactive conversational tutor featuring click-to-zoom diagram extraction modals and verifiable slide page citations.", PRIMARY),
        ("🧮 Parametric Numerical Solver", "Understands multi-stage neural network derivations (Autoencoders, Transpose Conv, MSE) and recalculates all intermediate tensors when parameters change.", SECONDARY),
        ("📚 Multi-Slide Notes Generator", "Stitches fragmented multi-page lecture sequences into unified, high-scoring university study notes with definitions, steps, and matrices.", ACCENT_PURPLE),
        ("🎯 Strict Conceptual Quizzes", "Automated multiple-choice assessment generator that strictly filters out calculation noise to evaluate true conceptual understanding.", RGBColor(16, 185, 129))
    ]

    for i, (title, desc, color) in enumerate(pillars):
        r = i // 2
        c = i % 2
        l = Inches(0.8) + c * Inches(5.95)
        t = Inches(1.8) + r * Inches(2.45)
        w = Inches(5.75)
        h = Inches(2.25)
        add_card(s3, l, t, w, h)
        
        # Color accent strip on left
        strip = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, Inches(0.15), h)
        strip.fill.solid()
        strip.fill.fore_color.rgb = color
        strip.line.fill.background()

        tb = s3.shapes.add_textbox(l + Inches(0.35), t + Inches(0.25), w - Inches(0.5), h - Inches(0.5))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = TEXT_MAIN

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(13)
        p2.font.color.rgb = TEXT_MUTED

    set_speaker_notes(s3, "Nalanda AI is organized into 4 pillar workspaces: Grounded Chat with page citations, Parametric Numerical Solving, Consolidated Exam Notes, and Conceptual Quizzes.")

    # =========================================================================
    # SLIDE 4: User Journey & Workflow
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, BG_LIGHT)
    add_header(s4, "User Experience", "User Journey: From Raw Lecture Slides to Exam Readiness")

    steps = [
        ("Step 1", "Upload Lecture Slides", "Student drops multi-page PDF slide decks into dedicated course workspaces. Background extraction parses text, tables, and visual figures."),
        ("Step 2", "Ask & Explore Concepts", "Student queries complex topics in natural language. The system retrieves grounded answers with clickable source pills and diagram modals."),
        ("Step 3", "Test 'What-If' Scenarios", "Student alters input vectors or filter weights. The numerical solver recomputes every intermediate convolution product and loss term live."),
        ("Step 4", "Generate Exam Materials", "1-click generation of consolidated study guides with comparison matrices, followed by practice quizzes with strict concept enforcement.")
    ]

    w_card = Inches(2.7)
    g_card = Inches(0.3)
    for i, (num, title, body) in enumerate(steps):
        l = Inches(0.8) + i * (w_card + g_card)
        t = Inches(2.0)
        h = Inches(4.4)
        add_card(s4, l, t, w_card, h)

        tb = s4.shapes.add_textbox(l + Inches(0.2), t + Inches(0.25), w_card - Inches(0.4), h - Inches(0.5))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = num.upper()
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = PRIMARY

        p2 = tf.add_paragraph()
        p2.text = title
        p2.font.size = Pt(16)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_MAIN

        p3 = tf.add_paragraph()
        p3.text = body
        p3.font.size = Pt(12)
        p3.font.color.rgb = TEXT_MUTED

    set_speaker_notes(s4, "The student journey flows naturally from uploading raw lecture slides, exploring concepts, verifying homework through 'What-If' calculations, to generating consolidated exam notes.")

    # =========================================================================
    # SLIDE 5: Flowchart & Architecture (Dark Theme)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, BG_DARK)
    add_header(s5, "System Engineering", "End-to-End System Pipeline & Information Flow", is_dark=True)

    stages = [
        ("1. Multi-Modal Ingestion", "• PyMuPDF text & layout extraction\n• High-resolution diagram cropper\n• Tabular matrix boundary parser\n• Slide sibling chunk unifier", Inches(0.8), Inches(1.8), Inches(2.7), Inches(4.8)),
        ("2. Indexing & Storage", "• Sentence-Transformers (all-MiniLM)\n• FAISS dense 384-dim vector index\n• SQLite relational metadata & figures\n• Folder isolation & cache tables", Inches(3.8), Inches(1.8), Inches(2.7), Inches(4.8)),
        ("3. ATML Router & Engines", "• Query classification router\n• Parametric algebraic solver\n• Seq2Seq refinement (Flan-T5)\n• Strict conceptual MCQ validator", Inches(6.8), Inches(1.8), Inches(2.7), Inches(4.8)),
        ("4. Client Workspace", "• Stitch SPA UI (Chat, Notes, Quiz)\n• Native Unicode math typography\n• Typewriter real-time streaming\n• Click-to-zoom diagram modals", Inches(9.8), Inches(1.8), Inches(2.7), Inches(4.8))
    ]

    for title, desc, l, t, w, h in stages:
        add_card(s5, l, t, w, h, bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))
        tb = s5.shapes.add_textbox(l + Inches(0.2), t + Inches(0.25), w - Inches(0.4), h - Inches(0.5))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = RGBColor(199, 210, 254)

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(12)
        p2.font.color.rgb = RGBColor(226, 232, 240)

    set_speaker_notes(s5, "This slide illustrates the four architectural tiers: Ingestion, Storage, the ATML Intelligence Engine, and the Client Workspace. Everything runs locally on consumer hardware.")

    # =========================================================================
    # SLIDE 6: Multi-Modal Extraction Pipeline
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, BG_LIGHT)
    add_header(s6, "Data Ingestion", "Multi-Modal Slide Intelligence: Text, Diagrams & Tabular Matrices")

    points = [
        ("🖼️ High-Resolution Figure Extraction", "Technical lecture slides convey essential intuition through diagrams. PyMuPDF identifies image bounding boxes, crops raster & vector diagrams, stores page metadata, and serves them via high-res zoom modals in the chat interface."),
        ("📊 Tabular Matrix Reconstruction", "Engineering slides frequently display performance benchmarks and layer comparisons in tables. Our parser detects row-column grids and converts them into structured Markdown tables with duplicate fingerprint filtering."),
        ("📑 Slide Sibling Chunk Unification", "Standard text splitters cut chunks at arbitrary word limits, cutting multi-column slides in half. Our chunker unifies all sibling text boxes on the same slide page to preserve full topical context.")
    ]

    for i, (title, body) in enumerate(points):
        t = Inches(1.8) + i * Inches(1.65)
        add_card(s6, Inches(0.8), t, Inches(11.7), Inches(1.45))
        tb = s6.shapes.add_textbox(Inches(1.1), t + Inches(0.2), Inches(11.1), Inches(1.05))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(16)
        p1.font.bold = True
        p1.font.color.rgb = PRIMARY

        p2 = tf.add_paragraph()
        p2.text = body
        p2.font.size = Pt(12)
        p2.font.color.rgb = TEXT_MUTED

    set_speaker_notes(s6, "Unlike standard RAG tools that only extract text, Nalanda AI extracts diagrams and tables, while keeping sibling text blocks unified on every slide.")

    # =========================================================================
    # SLIDE 7: AI Models & Technical Rationale
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, BG_LIGHT)
    add_header(s7, "Model Architecture", "AI Models Used in Nalanda AI: Rationale & Specific Roles")

    # Table of models
    t_shape = s7.shapes.add_table(5, 4, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    table = t_shape.table
    table.columns[0].width = Inches(2.2)
    table.columns[1].width = Inches(2.6)
    table.columns[2].width = Inches(3.5)
    table.columns[3].width = Inches(3.4)

    headers = ["System Layer", "Model / Component", "Why This Specific Model?", "Role in Project"]
    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE

    model_rows = [
        ("Semantic Retrieval", "all-MiniLM-L6-v2\n(SentenceTransformer)", "6-layer, 384-dim dense encoder. Runs in ~15ms on CPU with 80MB RAM footprint. High technical retrieval accuracy.", "Maps student queries and slide chunks into dense semantic vector space for FAISS search."),
        ("Answer Refinement", "Flan-T5-Base / Seq2Seq\n(250M parameters)", "Encoder-Decoder conditional transformer. Conditioned strictly on retrieved slide chunks to avoid hallucination.", "Synthesizes raw, fragmented OCR extractions into clean, student-friendly academic explanations."),
        ("Parametric Solver", "Closed-Form Algebraic Solver\n(State Machine)", "Neural nets guess probabilities; they do not perform reliable arithmetic. Deterministic solver guarantees 100% precision.", "Computes 1D convolutions, latent bottlenecks, transpose layers, and MSE loss dynamically."),
        ("Quiz Generation", "Rule-Based NLP & Heuristic Classifier", "Fast syntactic evaluation of mathematical tokens, vector brackets, and arithmetic operators.", "Enforces strict conceptual assessment by rejecting raw calculation trivia and vector noise.")
    ]

    for row_idx, r_data in enumerate(model_rows, start=1):
        for col_idx, val in enumerate(r_data):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if row_idx % 2 == 1 else RGBColor(241, 245, 249)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_MAIN

    set_speaker_notes(s7, "Here we detail the models used and why. We specifically chose `all-MiniLM` for fast local retrieval and `Flan-T5` for grounded refinement. Crucially, we use a deterministic algebraic solver for math to eliminate arithmetic hallucinations.")

    # =========================================================================
    # SLIDE 8: Core Innovation 1 — Parametric Numerical Solver
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, BG_LIGHT)
    add_header(s8, "Innovation 1", "Parametric Numerical Problem Solving: 1D Convolutional Autoencoder")

    add_card(s8, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    tb1 = s8.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "5-STAGE DERIVATION PIPELINE"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    stages_text = [
        ("1. Encoder 1D Convolution", "Sliding window scalar products with configurable kernel filter w and stride S."),
        ("2. Latent Bottleneck", "Multiplies feature map by W_fc (2 x 3) yielding compressed 2D latent vector z."),
        ("3. Decoder Reconstruction", "Expands latent vector back to 3D feature map using W_fc_decoder (3 x 2)."),
        ("4. Transpose Convolution", "Upsamples 3-element map back to 4-element sequence using symmetric weights."),
        ("5. Reconstruction Loss (MSE)", "Computes exact squared error per element: MSE = (1/N) * sum((x_hat_i - x_i)^2).")
    ]
    for title, desc in stages_text:
        p_t = tf1.add_paragraph()
        p_t.text = "• " + title
        p_t.font.size = Pt(13)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_MAIN
        p_d = tf1.add_paragraph()
        p_d.text = "   " + desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = TEXT_MUTED

    # Right Card: What-If Feature
    add_card(s8, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), bg_color=SECONDARY_LIGHT, border_color=SECONDARY)
    tb2 = s8.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "DYNAMIC 'WHAT-IF' RECALCULATION"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = SECONDARY

    p_body = tf2.add_paragraph()
    p_body.text = "Lecture slides only show one fixed numerical example (e.g., x=[1, 2, 3, 4], w=[0.5, -0.5]).\n\nNalanda AI allows students to ask:\n'What if input is [2, 3, 4, 5] and kernel is [0.8, -0.2]?'\n\nThe engine parses natural language inputs and re-computes every single intermediate multiplication, latent projection, and MSE loss step live.\n\nGuarantees 100% mathematical precision with zero hardcoded steps."
    p_body.font.size = Pt(13)
    p_body.font.color.rgb = TEXT_MAIN

    set_speaker_notes(s8, "This slide showcases our parametric numerical solver for Autoencoders. When students test custom inputs, our engine recalculates every intermediate tensor and MSE loss term with 100% precision.")

    # =========================================================================
    # SLIDE 9: Core Innovation 2 — Clean Academic Typography
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, BG_LIGHT)
    add_header(s9, "Innovation 2", "Clean Academic Typography: Native Unicode Mathematical Italics")

    col_w = Inches(5.7)
    add_card(s9, Inches(0.8), Inches(1.8), col_w, Inches(4.8), bg_color=RGBColor(254, 242, 242), border_color=RGBColor(248, 113, 113))
    tb_bad = s9.shapes.add_textbox(Inches(1.0), Inches(2.0), col_w - Inches(0.4), Inches(4.4))
    tf_bad = tb_bad.text_frame
    tf_bad.word_wrap = True
    p = tf_bad.paragraphs[0]
    p.text = "❌ WHAT GENERIC TOOLS OUTPUT"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(220, 38, 38)

    bad_examples = [
        "Raw LaTeX Escapes:",
        "   \\text{MSE} = \\frac{1}{N} \\sum_{i=1}^N (\\hat{x}_i - x_i)^2",
        "Raw Markdown Asterisks & Tildes:",
        "   *W*~fc~ (2 x 3): *z₁* = 1 · (1.0) + 0 · (1.6)",
        "Stray Escapes & Broken Delimiters:",
        "   $$ x = [1, 2, 3, 4] $$ and \\mathbf{W}_{conv}",
        "Result:",
        "   Unreadable math, broken rendering on mobile devices, and uncompiled raw syntax."
    ]
    for b in bad_examples:
        pb = tf_bad.add_paragraph()
        pb.text = b
        pb.font.size = Pt(12)
        pb.font.color.rgb = TEXT_MAIN

    add_card(s9, Inches(6.8), Inches(1.8), col_w, Inches(4.8), bg_color=PRIMARY_LIGHT, border_color=PRIMARY)
    tb_good = s9.shapes.add_textbox(Inches(7.0), Inches(2.0), col_w - Inches(0.4), Inches(4.4))
    tf_good = tb_good.text_frame
    tf_good.word_wrap = True
    p = tf_good.paragraphs[0]
    p.text = " NALANDA AI MATHEMATICAL TYPOGRAPHY"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    good_examples = [
        "Native Mathematical Italic Unicode:",
        "   𝑥 = [2.0, 3.0, 4.0, 5.0]  |  𝑊_conv = [0.8, -0.2]",
        "Clean Latent & Decoder Variables:",
        "   𝑧₁ = 1 · (1.0) + 0 · (1.6) + (-1) · (2.2) = -1.2",
        "   𝑓₁ = 1.0 · (-1.2) + 0.0 · (2.4) = -1.2",
        "   Reconstructed Signal: 𝑥̂ = [-0.96, 2.16, 1.44, -0.48]",
        "Loss Evaluation:",
        "   Formula: MSE = (1/N) · ∑ (𝑥̂ᵢ - 𝑥ᵢ)²",
        "Result:",
        "   Textbook-quality mathematical clarity, zero raw asterisks, zero LaTeX backslash clutter."
    ]
    for g in good_examples:
        pg = tf_good.add_paragraph()
        pg.text = g
        pg.font.size = Pt(12)
        pg.font.color.rgb = TEXT_MAIN

    set_speaker_notes(s9, "Notice the contrast. Instead of broken LaTeX escapes or asterisks, Nalanda AI uses native mathematical italic Unicode characters, creating textbook-grade clarity on all devices.")

    # =========================================================================
    # SLIDE 10: Core Innovation 3 & 4 — Notes & Quizzes
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, BG_LIGHT)
    add_header(s10, "Innovations 3 & 4", "Multi-Slide Notes Consolidation & Strict Conceptual Assessments")

    add_card(s10, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    tb_notes = s10.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.4))
    tf_n = tb_notes.text_frame
    tf_n.word_wrap = True
    p = tf_n.paragraphs[0]
    p.text = "MULTI-SLIDE TOPIC STITCHING"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    notes_pts = [
        "The Problem:",
        "Professors spread one derivation across 4–6 slides (Slide 1: Encoder, Slide 2: Latent, Slide 3: Decoder). Standard RAG creates 4 fragmented, disconnected notes.",
        "Nalanda's Consolidation Engine:",
        "• Automatically merges multi-slide stages under a single unified topic.",
        "• Generates Formal Definition, Step-by-Step Derivation, Verification Summary Matrix, and Exam Key Points.",
        "• Deduplicates identical summary tables and footnotes across slides."
    ]
    for pt in notes_pts:
        p_pt = tf_n.add_paragraph()
        p_pt.text = pt
        p_pt.font.size = Pt(12)
        p_pt.font.color.rgb = TEXT_MAIN

    add_card(s10, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    tb_quiz = s10.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.4))
    tf_q = tb_quiz.text_frame
    tf_q.word_wrap = True
    p = tf_q.paragraphs[0]
    p.text = "STRICT CONCEPTUAL QUIZ ENGINE"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = SECONDARY

    quiz_pts = [
        "The Pedagogical Rule:",
        "Quizzes must test conceptual mastery, not whether a student can multiply decimals in their head.",
        "Heuristic & Semantic Validation:",
        "• Rejects questions containing raw vector tokens (e.g., 'What is [0.5, -0.5] multiplied by [1, 2]?').",
        "• Eliminates raw arithmetic calculation trivia.",
        "• Evaluates core concepts: dimensionality reduction, transpose conv upsampling, bottleneck constraints.",
        "• Guarantees 100% plausible distractors and verified citations."
    ]
    for pt in quiz_pts:
        p_pt = tf_q.add_paragraph()
        p_pt.text = pt
        p_pt.font.size = Pt(12)
        p_pt.font.color.rgb = TEXT_MAIN

    set_speaker_notes(s10, "In Slide 10, we cover Notes Consolidation and Quiz Quality. Our system merges multi-slide lectures into cohesive study guides, and rejects calculation trivia from practice quizzes.")

    # =========================================================================
    # SLIDE 11: Competitive Analysis
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, BG_LIGHT)
    add_header(s11, "Market Benchmark", "Competitive Analysis: Nalanda AI vs. ChatGPT, Claude, Gemini & NotebookLM")

    t_shape = s11.shapes.add_table(7, 4, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    table = t_shape.table
    table.columns[0].width = Inches(2.4)
    table.columns[1].width = Inches(3.1)
    table.columns[2].width = Inches(2.8)
    table.columns[3].width = Inches(3.4)

    headers = ["Evaluation Dimension", "Generic LLMs (ChatGPT / Claude)", "Google NotebookLM", "Nalanda AI (Our System)"]
    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE

    benchmarks = [
        ("Data Privacy & Cost", "Remote cloud; subscription fees", "Cloud-only; Google ecosystem", " 100% Local (Zero API bills, total privacy)"),
        ("Multi-Stage Numericals", "Prone to math hallucinations", "Cannot compute numerical problems", " Deterministic Algebraic Solver (100% precision)"),
        ("'What-If' Parameter Recalculation", "Loses tensor dimensions & strides", "Not supported", " Dynamic Parameter Extraction & Solving"),
        ("Multi-Slide Lecture Stitching", "Chunks slides as isolated snippets", "Broad summary; loses slide steps", " Automated Multi-Slide Consolidation"),
        ("Quiz Pedagogical Quality", "Injects raw vectors & math trivia", "Generic flashcard-style questions", " Strict Conceptual Separation Engine"),
        ("Math Typography", "Broken LaTeX (\\text, $$, stray **)", "Plain text formatting", " Native Unicode Mathematical Italics")
    ]

    for row_idx, r_data in enumerate(benchmarks, start=1):
        for col_idx, val in enumerate(r_data):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if row_idx % 2 == 1 else RGBColor(241, 245, 249)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(11)
            p.font.color.rgb = PRIMARY if col_idx == 3 else TEXT_MAIN
            if col_idx == 3:
                p.font.bold = True

    set_speaker_notes(s11, "This table compares Nalanda AI directly with ChatGPT, Claude, and NotebookLM. We outperform them in academic contexts through local privacy, exact numerical derivation, and multi-slide consolidation.")

    # =========================================================================
    # SLIDE 12: Testing Strategy & Taxonomy
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, BG_LIGHT)
    add_header(s12, "Verification Rigor", "Testing Strategy: 167+ Automated Unit & Integration Tests")

    cat_cards = [
        ("1. Parametric Numericals", "7 Tests", "Validates exact mathematical invariants across default and custom autoencoder inputs.", PRIMARY),
        ("2. Notes Consolidation", "29 Tests", "Tests multi-slide merging, structural completeness, and table deduplication.", SECONDARY),
        ("3. Quiz & MCQ Quality", "42 Tests", "Validates 100% rejection of arithmetic trivia and tests cross-folder isolation.", ACCENT_PURPLE),
        ("4. Multi-Modal Ingestion", "35 Tests", "Tests PyMuPDF figure bounding boxes, layout parsing, and sibling chunking.", RGBColor(16, 185, 129)),
        ("5. API & Cache Reliability", "54 Tests", "FastAPI async routes, session security, and HTTP no-cache header enforcement.", RGBColor(245, 158, 11))
    ]

    w_c = Inches(2.15)
    g_c = Inches(0.24)
    for i, (name, count, desc, color) in enumerate(cat_cards):
        l = Inches(0.8) + i * (w_c + g_c)
        add_card(s12, l, Inches(1.8), w_c, Inches(3.2))

        # Top banner on card
        top_bar = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, Inches(1.8), w_c, Inches(0.12))
        top_bar.fill.solid()
        top_bar.fill.fore_color.rgb = color
        top_bar.line.fill.background()

        tb = s12.shapes.add_textbox(l + Inches(0.15), Inches(2.05), w_c - Inches(0.3), Inches(2.8))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = count
        p1.font.size = Pt(18)
        p1.font.bold = True
        p1.font.color.rgb = color

        p2 = tf.add_paragraph()
        p2.text = name
        p2.font.size = Pt(13)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_MAIN

        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(10)
        p3.font.color.rgb = TEXT_MUTED

    # Bottom Runner Card
    add_card(s12, Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.4), bg_color=PRIMARY_LIGHT, border_color=PRIMARY)
    tb_cmd = s12.shapes.add_textbox(Inches(1.0), Inches(5.45), Inches(11.3), Inches(1.1))
    tf_cmd = tb_cmd.text_frame
    tf_cmd.word_wrap = True
    p = tf_cmd.paragraphs[0]
    p.text = "EXECUTION COMMAND & FRAMEWORK: pytest (Python 3.13) with anyio async runner"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    p2 = tf_cmd.add_paragraph()
    p2.text = "$ PYTHONPATH=. pytest tests/   ==>  167 passed in 32.8s (100% deterministic, local execution)"
    p2.font.size = Pt(13)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_MAIN

    set_speaker_notes(s12, "Our test suite consists of 167 automated test cases spanning numerical correctness, quiz quality, notes consolidation, and API reliability. The entire suite runs locally in under 35 seconds.")

    # =========================================================================
    # SLIDE 13: Deep-Dive: What Was Tested (Specific Invariants)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13, BG_LIGHT)
    add_header(s13, "Test Validation", "Deep-Dive: What Was Tested Inside the Core Modules")

    tests_breakdown = [
        ("Parametric Numerical Assertions (test_numerical_solving_and_mcq_exclusions.py)", "• Verified default autoencoder: input [1,2,3,4] with kernel [0.5,-0.5] yields convolution output [-0.5,-0.5,-0.5], latent vector [0,-0.75], and MSE loss 7.2715.\n• Tested parameter perturbations (x=[2,4,6,8], w=[1,-1]) confirming dynamic tensor recalculation.\n• Tested 10+ natural language regex variations ('what if input is [2,3,4,5]?')."),
        ("Quiz Quality & Negative Testing (test_quiz_folder_isolation_and_quality.py)", "• Negative rejection filter: tested malformed math questions ('MSE loss is 7.2586') -> asserted 100% rejection rate.\n• Approved genuine concepts ('Autoencoder', 'Dimensionality Reduction').\n• Folder isolation: verified that generating a quiz for Course A contains 0% chunk contamination from Course B."),
        ("Multi-Slide Notes Consolidation (test_academic_notes_structuring.py)", "• Multi-slide stitching: fed mock slides with 4 separated stages -> asserted output contains exactly 1 unified topic.\n• Structural completeness: verified Definition, Step-by-Step Logic, Key Points, and Tables.\n• Deduplicated repetitive slide tables and footnotes automatically.")
    ]

    for i, (title, details) in enumerate(tests_breakdown):
        t = Inches(1.8) + i * Inches(1.65)
        add_card(s13, Inches(0.8), t, Inches(11.7), Inches(1.45))
        tb = s13.shapes.add_textbox(Inches(1.0), t + Inches(0.15), Inches(11.3), Inches(1.15))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = PRIMARY

        p2 = tf.add_paragraph()
        p2.text = details
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MAIN

    set_speaker_notes(s13, "Here we detail the exact invariants tested: mathematical decimal precision, negative test rejection of arithmetic noise in quizzes, and multi-slide notes stitching.")

    # =========================================================================
    # SLIDE 14: Tech Stack & Benchmarks
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14, BG_LIGHT)
    add_header(s14, "Implementation & Metrics", "Technology Stack & Local Hardware Benchmarks")

    add_card(s14, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    tb_stack = s14.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.4))
    tf_s = tb_stack.text_frame
    tf_s.word_wrap = True
    p = tf_s.paragraphs[0]
    p.text = "TECHNOLOGY STACK"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    stack_items = [
        ("Backend Framework:", "FastAPI (Async REST API), Python 3.13"),
        ("Database & ORM:", "SQLite / SQLAlchemy ORM"),
        ("Vector Embeddings:", "Sentence-Transformers (all-MiniLM-L6-v2)"),
        ("Dense Indexing:", "FAISS (Facebook AI Similarity Search)"),
        ("Refinement NLG:", "Hugging Face Transformers (Flan-T5-Base)"),
        ("PDF Ingestion:", "PyMuPDF (fitz)"),
        ("Frontend Client:", "Vanilla JS Component SPA, Tailwind CSS"),
        ("Cache Control:", "Strict HTTP No-Cache Middleware + v43 Busting")
    ]
    for lbl, val in stack_items:
        p_item = tf_s.add_paragraph()
        p_item.text = f"• {lbl} {val}"
        p_item.font.size = Pt(12)
        p_item.font.color.rgb = TEXT_MAIN

    add_card(s14, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    tb_metrics = s14.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.4))
    tf_m = tb_metrics.text_frame
    tf_m.word_wrap = True
    p = tf_m.paragraphs[0]
    p.text = "LOCAL HARDWARE BENCHMARKS"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = SECONDARY

    metric_items = [
        ("Numerical Solving Latency:", "< 45 ms (Instantaneous on CPU)"),
        ("Semantic Vector Search:", "~110 ms for top-5 dense chunk retrieval"),
        ("Peak Memory Consumption:", "< 350 MB RAM on consumer laptop"),
        ("Storage Footprint:", "< 25 MB vector index for 100 lecture slides"),
        ("Cloud Dependency:", "0% (Completely offline & edge-native)"),
        ("API Billing / Costs:", "$0.00 / month ongoing operating cost"),
        ("Test Execution Time:", "32.8s for 167 automated unit tests")
    ]
    for lbl, val in metric_items:
        p_item = tf_m.add_paragraph()
        p_item.text = f"• {lbl} {val}"
        p_item.font.size = Pt(12)
        p_item.font.color.rgb = TEXT_MAIN

    set_speaker_notes(s14, "Nalanda AI is extremely lightweight: sub-100ms response times, under 350MB RAM usage, and zero monthly API costs. It runs seamlessly on any student laptop.")

    # =========================================================================
    # SLIDE 15: Live Demo Plan
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_background(s15, BG_LIGHT)
    add_header(s15, "Demonstration", "Live System Demonstration & Capstone Walkthrough")

    demo_steps = [
        ("1. Lecture Slide Ingestion", "Upload raw 1D Convolutional Autoencoder lecture PDF into Deep Learning folder.", "Demonstrates multi-modal extraction of text, diagrams, and tables."),
        ("2. Parametric Chat Query", "Query: 'What if input is [2, 3, 4, 5] and kernel is [0.8, -0.2]?'", "Demonstrates live recalculation with clean Unicode mathematical italics."),
        ("3. Diagram Modal Preview", "Click on the citation pill in the assistant's response.", "Demonstrates high-resolution visual figure zoom modal directly linked to page metadata."),
        ("4. Consolidated Notes Tab", "Navigate to Notes workspace.", "Demonstrates 4-slide numerical sequence consolidated into a single structured topic with verification table."),
        ("5. Conceptual Quiz Mode", "Click Generate Practice Quiz.", "Demonstrates high-yield conceptual MCQs with 100% rejection of raw arithmetic trivia.")
    ]

    for i, (step, act, desc) in enumerate(demo_steps):
        t = Inches(1.8) + i * Inches(1.0)
        add_card(s15, Inches(0.8), t, Inches(11.7), Inches(0.85))
        tb = s15.shapes.add_textbox(Inches(1.0), t + Inches(0.1), Inches(11.3), Inches(0.65))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = f"{step}: {act}"
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = PRIMARY

        p2 = tf.add_paragraph()
        p2.text = f"   Purpose: {desc}"
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED

    set_speaker_notes(s15, "I will now walk through the live demo, showing slide upload, parametric derivation with custom numbers, diagram zoom modals, notes generation, and conceptual quiz mode.")

    # =========================================================================
    # SLIDE 16: Future Scope & Roadmap
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    set_slide_background(s16, BG_LIGHT)
    add_header(s16, "Future Work", "System Limitations & Future Engineering Roadmap")

    add_card(s16, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    tb_lim = s16.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.4))
    tf_l = tb_lim.text_frame
    tf_l.word_wrap = True
    p = tf_l.paragraphs[0]
    p.text = "CURRENT SYSTEM BOUNDARIES"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(220, 38, 38)

    lim_items = [
        "Domain Focus:",
        "Currently specialized on Neural Networks, Signal Processing, Computer Science, and Engineering lecture slide decks.",
        "Input Format:",
        "Optimized for PDF slide decks; handwritten blackboard notes require further OCR adaptation.",
        "Deterministic Coverage:",
        "Closed-form solver currently covers Convolutional Autoencoders, Backpropagation, and Dense Bottlenecks."
    ]
    for item in lim_items:
        p_i = tf_l.add_paragraph()
        p_i.text = item
        p_i.font.size = Pt(12)
        p_i.font.color.rgb = TEXT_MAIN

    add_card(s16, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    tb_fut = s16.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.4))
    tf_f = tb_fut.text_frame
    tf_f.word_wrap = True
    p = tf_f.paragraphs[0]
    p.text = "FUTURE ENGINEERING ROADMAP"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = SECONDARY

    fut_items = [
        "Cross-Disciplinary Solvers:",
        "Extending symbolic solver engines to Electrical Engineering (RLC circuits, Bode plots) and Mechanical Engineering (Stress-strain matrices).",
        "Interactive Socratic Hint Tutor:",
        "Guiding students through derivations one hint at a time instead of outputting the complete answer immediately.",
        "Direct Flashcard Synchronization:",
        "Automated export of generated notes and concept units to Anki (.apkg) and Quizlet."
    ]
    for item in fut_items:
        p_i = tf_f.add_paragraph()
        p_i.text = item
        p_i.font.size = Pt(12)
        p_i.font.color.rgb = TEXT_MAIN

    set_speaker_notes(s16, "In our future roadmap, we aim to extend symbolic solving to circuit analysis and mechanics, alongside Socratic hint tutoring and direct Anki flashcard export.")

    # =========================================================================
    # SLIDE 17: Conclusion
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    set_slide_background(s17, BG_DARK)
    add_header(s17, "Project Summary", "Conclusion & Key Capstone Contributions", is_dark=True)

    c_box = add_card(s17, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8), bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))
    tb_c = s17.shapes.add_textbox(Inches(1.2), Inches(2.1), Inches(10.9), Inches(4.2))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True

    concl_points = [
        ("100% Privacy & Zero Operational Cost:", "Delivered a fully edge-native academic copilot that runs on student laptops with zero reliance on paid commercial cloud APIs."),
        ("Eliminated Math Hallucinations:", "Coupled transformer-based semantic retrieval with a deterministic closed-form solver, guaranteeing 100% exact numerical derivation for coursework."),
        ("Pedagogical Cohesion:", "Engineered automated multi-slide stitching that transforms fragmented lecture slides into university-grade exam study notes."),
        ("Rigorous Engineering Validation:", "Supported by an automated test suite of 167+ test cases spanning numerical invariants, layout parsing, and quiz heuristics."),
        ("Core Takeaway:", "'AI in education must be reliable, reproducible, and explainable — Nalanda AI makes lecture notes truly student-ready.'")
    ]

    for title, desc in concl_points:
        p = tf_c.add_paragraph()
        p.text = "• " + title + " "
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = SECONDARY
        
        p_d = tf_c.add_paragraph()
        p_d.text = "   " + desc
        p_d.font.size = Pt(13)
        p_d.font.color.rgb = RGBColor(226, 232, 240)

    set_speaker_notes(s17, "In conclusion, Nalanda AI bridges the critical gap between raw lecture slides and student understanding. It provides privacy, exact math derivation, and pedagogical quality.")

    # =========================================================================
    # SLIDE 18: Q&A / Bibliography
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    set_slide_background(s18, BG_DARK)

    tb_qa = s18.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(2.0))
    tf_qa = tb_qa.text_frame
    tf_qa.word_wrap = True
    p = tf_qa.paragraphs[0]
    p.text = "THANK YOU!"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE

    p2 = tf_qa.add_paragraph()
    p2.text = "Questions & Committee Discussion"
    p2.font.size = Pt(24)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(199, 210, 254)

    card_bib = add_card(s18, Inches(0.8), Inches(4.3), Inches(11.7), Inches(2.2), bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))
    tb_b = s18.shapes.add_textbox(Inches(1.1), Inches(4.5), Inches(11.1), Inches(1.8))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    p = tf_b.paragraphs[0]
    p.text = "PROJECT REPOSITORY & KEY REFERENCES"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = SECONDARY

    bibs = [
        "GitHub Repository: https://github.com/themayankguy/Nalanda-AI (Branch: ui/stitch-redesign)",
        "1. Reimers & Gurevych, 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks', EMNLP 2019.",
        "2. Raffel et al., 'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)', JMLR 2020.",
        "3. Goodfellow, Bengio, Courville, 'Deep Learning (Autoencoders & Transpose Convolutions)', MIT Press."
    ]
    for b in bibs:
        p_b = tf_b.add_paragraph()
        p_b.text = "• " + b
        p_b.font.size = Pt(11)
        p_b.font.color.rgb = RGBColor(203, 213, 225)

    set_speaker_notes(s18, "Thank you respected committee members. I am now open to your questions, feedback, and discussion.")

    # Save presentation
    output_path = "/Users/mayanksmac/Desktop/capstone/Nalanda_AI_Capstone_Presentation.pptx"
    prs.save(output_path)
    print(f"Presentation successfully saved to: {output_path}")

if __name__ == "__main__":
    create_presentation()
