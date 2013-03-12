SHELL=/bin/bash

proposal.pdf: proposal.tex bibliography.bib
	pdflatex $< && bibtex $(basename $<) && pdflatex $< && pdflatex $<

midterm.pdf: midterm.tex bibliography.bib
	pdflatex $< && bibtex $(basename $<) && pdflatex $< && pdflatex $<

clean:
	rm -f proposal.{aux,log,bbl,blg}
	rm -f midterm.{aux,log,bbl,blg}

spotless: clean
	rm -f proposal.pdf
	rm -f midterm.pdf

.PHONY: clean spotless
