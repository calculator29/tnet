INCLUDE=-I../include
LIB=-L../lib
LINK=-ltnet

CC=g++
CFLAGS=-ansi -Wall -O3 -std=c++14 $(INCLUDE) $(LIB) -funroll-loops

%: %.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LINK)
clean :
	rm -f *.o *~ core *test
