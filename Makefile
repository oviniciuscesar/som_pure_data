#nome do objeto
lib.name = som

#nome do arquivo que contém o código do objeto
class.sources = som.c 

# include paths for homebrew

cflags += -I/usr/local/include
ldflags += -L/usr/local/lib -lgsl -static -lgslcblas

#incluir o help do objeto
datafiles = som-help.pd


# include Makefile.pdlibbuilder from submodule directory 'pd-lib-builder'
# PDLIBBUILDER_DIR=pd-lib-builder/
# include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder

include ../pd-lib-builder-master/Makefile.pdlibbuilder