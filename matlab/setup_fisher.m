% compile

disp('Making mexFisherEncodeHelperSP...');
mex mexFisherEncodeHelperSP.cxx ../fisher.cxx ../gmm.cxx ../stat.cxx ../simd_math.cxx -largeArrayDims CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"

