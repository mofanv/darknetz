CC      ?= $(CROSS_COMPILE)gcc
LD      ?= $(CROSS_COMPILE)ld
AR      ?= $(CROSS_COMPILE)ar
NM      ?= $(CROSS_COMPILE)nm
OBJCOPY ?= $(CROSS_COMPILE)objcopy
OBJDUMP ?= $(CROSS_COMPILE)objdump
READELF ?= $(CROSS_COMPILE)readelf

#ALIB=libdarknet.a
#ARFLAGS = -rcs
LDFLAGS= -lm -pthread

OBJSCR = src/
EXECSCR = examples/

OBJ = tcp.transfer.o diffprivate.o gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o upsample_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o l2norm_layer.o yolo_layer.o iseg_layer.o

EXECOBJ = captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o instance-segmenter.o darknet.o

OBJS = $(addprefix $(OBJSCR), $(OBJ))
EXECOBJS = $(addprefix $(EXECSCR), $(EXECOBJ))

EXECMAIN = main.o

CFLAGS += -Wall -I../ta/include -I$(TEEC_EXPORT)/include -I./include
CFLAGS += -Iinclude/ -Isrc/ -Iexamples/  -I./src/ -I./examples/

#Add/link other required libraries here
LDADD += -lteec -L$(TEEC_EXPORT)/lib
#DEPS = $(wildcard src/*.h) inlcude/darknet.h inlcude/main.h

BINARY = optee_example_darknetp

.PHONY: all
#all: $(ALIB) $(BINARY)
all: $(BINARY)

$(BINARY): $(EXECMAIN) $(EXECOBJS) $(OBJS)
	#$(CC) $(CFLAGS) -g $^ -o $@ $(LDFLAGS) $(LDADD) $(ALIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(LDADD)

#$(ALIB): $(OBJS)
#	$(AR) $(ARFLAGS) $@ $^

$(OBJSCR)%.o: $(OBJSCR)%.c
		$(CC) $(CFLAGS) -c $< -o $@

$(EXECSCR)%.o: $(EXECSCR)%.c
		$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
		$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(EXECOBJS) $(EXECMAIN) $(BINARY)
