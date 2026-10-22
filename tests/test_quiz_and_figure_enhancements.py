"""Tests for generalized quiz filtering, OCR HTML tag sanitization, and diagram cropping."""
import io
import re
from pathlib import Path
from PIL import Image
import pytest

from backend.services.atml.content_normalizer import ContentNormalizer
from backend.services.atml.output_validator import sanitize_text
from backend.services.atml.answer_structurer import AnswerStructurer
from nalanda_extractor import extract_and_crop_ppstructure_figures, clean_text


def test_is_assessment_or_quiz_content():
    """Verify that quiz, exercise, and assessment slides are accurately detected while educational content is preserved."""
    # Quiz slide with explicit header and questions
    quiz_header = "Quiz Time!"
    quiz_text = (
        "1. What does meta-learning mean in simple terms?\n"
        "2. Why is traditional ML poor with small datasets?\n"
        "3. What's an example of one-shot learning in humans?\n"
        "4. What is zero-shot learning?\n"
        "5. Name a real-world use-case of few-shot learning."
    )
    assert ContentNormalizer.is_assessment_or_quiz_content(quiz_header, quiz_text) is True

    # Review questions without header
    review_text = (
        "1. What is an encoder?\n"
        "2. How does 4-to-2 encoder work?\n"
        "3. Which signal is active high?"
    )
    assert ContentNormalizer.is_assessment_or_quiz_content("", review_text) is True

    # Substantive content slide with a title that has an interrogative question
    content_header = "What is Meta-learning?"
    content_text = (
        "“Learning to learn” — a model improves how it learns\n"
        "Learns from a distribution of tasks instead of just one\n"
        "Quickly adapts to new tasks with few examples"
    )
    assert ContentNormalizer.is_assessment_or_quiz_content(content_header, content_text) is False

    # Educational tutorial with numbered steps
    step_header = "Deployment Pipeline"
    step_text = (
        "1. Build the Docker image using multi-stage builds.\n"
        "2. Push the container to the enterprise artifact registry.\n"
        "3. Deploy using Kubernetes deployment rollout."
    )
    assert ContentNormalizer.is_assessment_or_quiz_content(step_header, step_text) is False


def test_clean_ocr_html_tags():
    """Verify that raw OCR HTML diagram markup is completely removed from text."""
    raw_ocr = (
        "For example, a 4-to-2 encoder has four inputs and produces a 2-bit output code.\n\n"
        "<div style=\"text-align: center;\"><img src=\"imgs/img_in_image_box_277_834_1470_1288.jpg\" alt=\"Image\" width=\"62%\" /></div>\n\n"
        "<div style=\"text-align: center;\">Encoder </div>"
    )
    cleaned_ocr = clean_text(raw_ocr)
    assert "<img" not in cleaned_ocr
    assert "<div" not in cleaned_ocr
    assert "imgs/img_in_image_box" not in cleaned_ocr
    assert "4-to-2 encoder" in cleaned_ocr

    # Also test sanitize_text in output_validator
    sanitized = sanitize_text(raw_ocr)
    assert "<img" not in sanitized
    assert "<div" not in sanitized
    assert "4-to-2 encoder" in sanitized


def test_answer_structurer_quiz_filtering():
    """Verify that AnswerStructurer does not output quiz questions as explanatory steps for conceptual queries."""
    structurer = AnswerStructurer()
    quiz_content = (
        "Quiz Time!\n\n"
        "1. What does meta-learning mean in simple terms?\n"
        "2. Why is traditional ML poor with small datasets?\n"
        "3. What's an example of one-shot learning in humans?\n"
        "4. What is zero-shot learning?\n"
        "5. Name a real-world use-case of few-shot learning."
    )
    result = structurer.structure("what is meta learning", quiz_content)
    # The quiz questions should not be formatted as answer steps
    assert "1. What does meta-learning mean in simple terms?" not in result
    assert "Quiz Time!" not in result


def test_extract_and_crop_ppstructure_figures(tmp_path):
    """Verify that PP-Structure diagram bounding boxes are accurately parsed and cropped."""
    # Create a synthetic image of 500x500
    img_path = tmp_path / "diagram_test.png"
    img = Image.new("RGB", (500, 500), color=(255, 255, 255))
    # Draw a colored rectangle representing the diagram box at [50, 100, 400, 350]
    for x in range(50, 400):
        for y in range(100, 350):
            img.putpixel((x, y), (100, 150, 200))
    img.save(img_path)

    dirs = {"figures": tmp_path / "figures"}
    dirs["figures"].mkdir(parents=True, exist_ok=True)

    text_with_box = (
        "Encoder overview.\n\n"
        "<div style=\"text-align: center;\"><img src=\"imgs/img_in_image_box_50_100_400_350.jpg\" alt=\"Image\" /></div>\n\n"
        "<div style=\"text-align: center;\">Encoder Diagram</div>"
    )

    figures, cleaned = extract_and_crop_ppstructure_figures(
        image_path=img_path,
        raw_text=text_with_box,
        dirs=dirs,
        page_number=1,
        stem="diagram_test",
    )

    assert len(figures) == 1
    assert figures[0]["width"] == 350
    assert figures[0]["height"] == 250
    assert figures[0]["caption"] == "Encoder Diagram"
    assert "<img" not in cleaned
    assert "<div" not in cleaned

    # Check the cropped image on disk
    saved_crop = dirs["figures"] / figures[0]["filename"]
    assert saved_crop.exists()
    with Image.open(saved_crop) as crop_img:
        assert crop_img.size == (350, 250)
