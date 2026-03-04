# Makefile for Sphinx documentation.

SPHINXOPTS ?=
SPHINXBUILD ?= sphinx-build
SOURCEDIR = .
CONFDIR = source
BUILDDIR = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" -c "$(CONFDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" -c "$(CONFDIR)" $(SPHINXOPTS) $(O)
