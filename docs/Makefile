# docs/Makefile

# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help Makefile

help:
	@echo "Sphinx documentation commands:"
	@echo "  make html       # build HTML in _build/html"
	@echo "  make latexpdf   # build PDF in _build/latex"
	@echo "  make clean      # remove build output"
	@echo

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

latexpdf:
	$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) && \
	cd "$(BUILDDIR)/latex" && make all-pdf

clean:
	rm -rf $(BUILDDIR)
