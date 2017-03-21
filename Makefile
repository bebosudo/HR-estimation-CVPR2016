TARGET	= HR-estimation-CVPR2016
LIBSRCS	=
LIBOBJS	= src/main.o

OPT	= -g
PIC	=

CXX	= g++
# CFLAGS= $(OPT) $(PIC) $(XOPTS)
CFLAGS= $(OPT)


$(TARGET): $(LIBOBJS)
	$(CXX) -o $@ $(CFLAGS) $(LIBOBJS)

clean:
	@/bin/rm -f $(TARGET) $(LIBOBJS)
