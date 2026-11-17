# -*- coding: utf-8 -*-
"""
Builds the comprehensive technical report for ViLiS in LaTeX, HTML, and PDF formats.
Designed specifically as an exhaustive reference document and for direct ingestion into RAG systems.
"""

import os
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/sameerkumar/Desktop/vilis")
DOCS_DIR = BASE_DIR / "interview_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

HTML_PATH = DOCS_DIR / "ViLiS_Complete_Technical_Report.html"
TEX_PATH = DOCS_DIR / "ViLiS_Complete_Technical_Report.tex"
PDF_PATH = BASE_DIR / "ViLiS_Complete_Technical_Report.pdf"

print("Starting generation of ViLiS Complete Technical Report...")
