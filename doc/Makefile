proposal.pdf: proposal.tex bibliography.bib
	pdflatex proposal.tex && bibtex proposal && pdflatex proposal.tex && pdflatex proposal.tex

clean:
	rm -f proposal.{aux,log,bbl,blg}
