# ADMET-Engine v2.0 — Reproducible White Paper Build
.PHONY: all figures pdf clean

all: figures pdf

figures:
	../.venv/bin/python generate_figures.py

pdf: admet_engine_whitepaper.tex references.bib
	pdflatex -interaction=nonstopmode admet_engine_whitepaper.tex
	bibtex admet_engine_whitepaper
	pdflatex -interaction=nonstopmode admet_engine_whitepaper.tex
	pdflatex -interaction=nonstopmode admet_engine_whitepaper.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk
