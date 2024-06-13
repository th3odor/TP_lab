all: build/axions.pdf

texoptions = \
	     --lualatex \
	     --interaction=nonstopmode \
	     --halt-on-error \
	     --output-directory=build

build/axions.pdf: FORCE | build
	latexmk $(texoptions) axions.tex

preview: FORCE | build
	latexmk $(texoptions) -pvc axions.tex

FORCE:

build:
	mkdir -p build

clean:
	rm -r build
