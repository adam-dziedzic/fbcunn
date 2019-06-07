CXX := nvcc
TARGET := conv
CUDNN_PATH := cudnn
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAGS := -arch=sm_35 -std=c++11 -O2
DEBUG_FLAGS := -g

all: conv

conv: $(TARGET).cu
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -o \
	$(TARGET) -lcudnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
