# Minimal makefile for Sphinx documentation at least I hope

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
PAPER         ?=
BUILDDIR      ?= _build

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the Sphinx executable. Alternatively you can add the directory with the Sphinx executable to your PATH. If you do not have Sphinx installed, via pip install -e then the following should work for you: pip install sphinx)
endif

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
SPHINXLANG      ?= -D language=en
I18NSPHINXOPTS  = $(SPHINXLANG) -d $(BUILDDIR)/doctrees .

.PHONY: help Makefile clean html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest coverage

# Catch-all target: route all unknown targets to Sphinx using the "new"
# "make mode" option.  $(O) is meant as a shortcut for $(S)/SPHINXOPTS.
%: Makefile
	@$(SPHINXBUILD) $(ALLSPHINXOPTS) $(SPHINXLANG) -b $@ $(BUILDDIR) $@

help:
	@$(SPHINXBUILD) -M help $(ALLSPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/*

html:
	$(SPHINXBUILD) $(ALLSPHINXOPTS) $(SPHINXLANG) -b html $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML documentation is generated in $(BUILDDIR)/html."
