SHELL=/bin/bash

MAINDOC = final
TEXFILES = $(MAINDOC).tex $(shell echo tex/*.tex)

final.pdf: $(TEXFILES)
	pdflatex $<
	bibtex $(MAINDOC)
	pdflatex $<
	pdflatex $<

clean:
	rm -f $(MAINDOC).{aux,log,bbl,blg}

spotless: clean
	rm -f $(MAINDOC).pdf

.PHONY: clean spotless
