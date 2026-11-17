"""Generates the comprehensive ViLiS Interview Mastery Guide in LaTeX, HTML, and compiled PDF."""

import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/sameerkumar/Desktop/vilis")
DOCS_DIR = BASE_DIR / "interview_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

HTML_PATH = DOCS_DIR / "ViLiS_Interview_Mastery_Guide.html"
TEX_PATH = DOCS_DIR / "ViLiS_Interview_Mastery_Guide.tex"
PDF_DEST = BASE_DIR / "ViLiS_Comprehensive_Interview_Mastery_Guide.pdf"

print("Creating detailed Interview Mastery Guide files...")
