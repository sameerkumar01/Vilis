import sys
import os
from pathlib import Path
import subprocess

BASE_DIR = Path("/Users/sameerkumar/Desktop/vilis")
DOCS_DIR = BASE_DIR / "interview_docs"
HTML_FILE = DOCS_DIR / "ViLiS_Interview_Mastery_Guide.html"
TEX_FILE = DOCS_DIR / "ViLiS_Interview_Mastery_Guide.tex"
PDF_FILE = BASE_DIR / "ViLiS_Comprehensive_Interview_Mastery_Guide.pdf"

print("Starting generation of comprehensive interview preparation guide...")
