# ADMET-Engine v2.0 — Reproducible Preprint Build
.PHONY: all figures pdf clean

all: figures pdf

figures:
	../.venv/bin/python generate_figures.py

pdf: admet_engine_preprint.tex references.bib
	pdflatex -interaction=nonstopmode admet_engine_preprint.tex
	bibtex admet_engine_preprint
	pdflatex -interaction=nonstopmode admet_engine_preprint.tex
	pdflatex -interaction=nonstopmode admet_engine_preprint.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk
