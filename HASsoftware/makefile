# makefile

ModelDir = models/WarpedTorus3D


mainExec : main.o model.o HAS.o
	g++ -std=c++14 -lclapack main.o model.o HAS.o -o mainExec

model.o : $(ModelDir)/model.cpp $(ModelDir)/model.hpp
	g++ -std=c++14 -I$(ModelDir) -c $(ModelDir)/model.cpp

HAS.o : HAS.cpp HAS.hpp $(ModelDir)/model.hpp
	g++ -std=c++14 -I$(ModelDir) -c HAS.cpp

main.o : main.cpp $(ModelDir)/model.hpp HAS.hpp
	g++ -std=c++14 -I$(ModelDir) -c main.cpp

