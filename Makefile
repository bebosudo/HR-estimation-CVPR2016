TARGET	= HR-estimation-CVPR2016
LIBSRCS	= `pkg-config --cflags --libs opencv`
LIBOBJS	= src/main.o

CXX	= g++

OPT	= -g
PIC	=

CFLAGS= $(OPT) $(PIC)


$(TARGET): $(LIBOBJS)
	$(CXX) -o $@ $(CFLAGS) $(LIBSRCS) $(LIBOBJS)

clean:
	@/bin/rm -f $(TARGET) $(LIBOBJS)
