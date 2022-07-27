################################################################
####
#### Makefile for building EduDL network
####
#### 2021
####
################################################################

info ::
	@echo && echo "%%%% Edu DL %%%%" && echo "Available rules:"

include Make.inc

#
# for now just a single build line
#
LIBSRCS := vector2.cpp matrix.cpp net.cpp dataset.cpp layer.cpp funcs.cpp loss.cpp vector.cpp trace.cpp book.cpp
ifeq "${USE_BLIS}" "1"
 LIBSRCS += matrix_impl_blis.cpp vector_impl_blis.cpp vectorbatch_impl_blis.cpp
else
 LIBSRCS += matrix_impl_reference.cpp vector_impl_reference.cpp vectorbatch_impl_reference.cpp
endif
LIBOBJS = $(patsubst %.cpp,%.o,${LIBSRCS})

ECHO = 0
%.o : %.cpp
	@echo "compiling $<"
	@if [ "${DEBUG}" = "1" ] ; then debugopt="-DDEBUG" ; fi \
	 && if [ "${USE_BLIS}" = "1" ] ; then blisopt="-DBLISNN -I${BLIS_INC_DIR}" ; fi \
	 && if [ "${USE_GSL}" = "1" ] ; then gslopt="-DUSE_GSL -I${GSL_INC_DIR}" ; fi \
	 && cmdline="${CXX} -c $< $$debugopt $$blisopt $$gslopt -I${CXXOPTS}/include -I${CATCH_INCLUDES}" \
	 && if [ ${ECHO} = 1 ] ; then echo "$$cmdline" ; fi \
	 && eval $$cmdline

book.o : book.h funcs.h matrix.h
vector2.o vector_impl_blis.o vectorbatch_impl_blis.o : vector2.h
dataset.o layer.o matrix.o net.: matrix.h
dataset.o : dataset.h 
funcs.o : funcs.h matrix.h
loss.o : loss.h
layer.o : funcs.h layer.h
net.o : net.h dataset.h layer.h
test.o : matrix.h net.h dataset.h layer.h funcs.h
dataset.o funcs.o net.o layer.o vectorbatch_impl_reference.o vectorbatch_impl_blis.o \
    trace.o : trace.h

#
# implementation specific files have to be recompiled
# if we tinker with settings
#
BLAS_FILES = $(wildcard *_impl_*.cpp)
BLAS_OBJS = $(patsubst %.cpp,%.o,${BLAS_FILES})
${BLAS_OBJS} : Make.inc

TESTS = mnist posneg linear
TEST = mnist
info ::
	@echo "make test TEST=.... (out of: ${TESTS}, default=${TEST})"
	@echo "make compileall : all tests"
.PHONY: test compileall
compileall ::
	@for t in ${TESTS} ; do \
	    echo "================================ Compiling: $$t" \
	     && make --no-print-directory test TEST=$$t \
	    ; done
test :  ${LIBOBJS}
	@program=test_${TEST} \
	 && make --no-print-directory $$program.o DEBUG=${DEBUG} \
	 && echo && echo "Linking test program <<$$program>>" \
	 && ${CXX} -o $$program $$program.o ${LIBOBJS} \
	    ` if [ "${USE_BLIS}" = "1" ] ; then echo "-L${BLIS_LIB_DIR} -lblis -lm" ; fi ` \
	 && echo ".. done"

info ::
	@echo "make run TEST=... RUN_OPTIONS=..."
.PHONY: run
run :
	@echo "Running: test_${TEST} ${RUN_OPTIONS}" \
	 && if [ ! -z "${TACC_TAU_DIR}" ] ; then \
	        tau_out=tau_out.${TEST} \
	         && echo ".. tau output in: $${tau_out}" \
	         && export TAU_TRACE=1 \
	         && export TAU_PROFILE=1 \
	         && export TRACEDIR=$${tau_out} \
	         && export PROFILEDIR=$${tau_out} \
	         && rm -rf $${tau_out} && mkdir -p $${tau_out} \
	      ; fi \
	 && ./test_${TEST} ${RUN_OPTIONS}

CATCH_INCLUDES = ${TACC_CATCH2_INC}
CATCH_LIBS = -L${TACC_CATCH2_LIB} -lCatch2Main -lCatch2
info ::
	@echo "make unittest run_unittest : catch-based"
unittest : unittest.o ${LIBOBJS}
	${CXX} -o $@ $@.o ${LIBOBJS} ${CATCH_LIBS}
compileall ::
	@echo "================================ Compiling: unittest" \
	 && make --no-print-directory unittest
unittest.o : funcs.h net.h
.PHONY: run_unittest
run_unittest : unittest
	./unittest

info ::
	@echo "================================"
	@echo "make docs : doxygen"
.PHONY: docs
docs :
	doxygen edudl.doxy
	cd docs/latex && make && cp refman.pdf ../../edudl.pdf \
	 && if [ "$(whoami)" = "eijkhout" ] ; then \
	        cp refman.pdf ${HOME}/Dropbox/Scicomp/ ; fi 

# ##
# ## old rules. must go
# ##
# PROGRAMS = testdl posneg mpidl
# info ::
# 	@echo "make PROGRAM (where PROGRAM in ${PROGRAMS}) : make application"
# .SECONDEXPANSION:
# testdl : test.o ${LIBOBJS}
# 	@echo "Linking test program <<$@>>"
# 	@${CXX} -o $@ $^ \
# 	    ` if [ "${USE_BLIS}" = "1" ] ; then echo "-L${BLIS_LIB_DIR} -lblis -lm" ; fi `
# mpidl : test_mpi.o ${LIBOBJS} net_mpi.o
# 	@echo "Linking test program <<$@>>"
# 	@${CXX} -o $@ $^ \
# 	    ` if [ "${USE_BLIS}" = "1" ] ; then echo "-L${BLIS_LIB_DIR} -lblis -lm" ; fi `
# posneg : $$@.o ${LIBOBJS}
# 	@echo "Linking test program <<$@>>"
# 	@${CXX} -o $@ $^ \
# 	    ` if [ "${USE_BLIS}" = "1" ] ; then echo "-L${BLIS_LIB_DIR} -lblis -lm" ; fi `
# test.o posneg.o : vector2.h net.h dataset.h vector.h

.PHONY: clean
info ::
	@echo "make clean : remove junk and executables"
clean ::
	@rm -f *~ *.o *.o.tmp a.out ${PROGRAMS}
	@for t in ${TESTS} ; do rm -f test_$$t ; done
