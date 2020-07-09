rm -f ./generate_rcn_hdf5.o ./opengm_solvers.o

g++ generate_rcn_hdf5.cpp -lhdf5 -lopencv_core -lopencv_highgui -o ./generate_rcn_hdf5.o

g++ opengm_solvers.cpp -lhdf5 -lad3 -lcb -lqpbo -lmplp -lsrmp -o ./opengm_solvers.o