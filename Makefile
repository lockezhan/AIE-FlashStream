XILINX_XRT ?= /opt/xilinx/xrt
VITIS_ROOT ?= /s3/Xilinx_Vitis_2022.2/Vitis/2022.2
HOST_EXE := host/llama3_attention_host.exe
XCLBIN := llama3_attention.xclbin

all: $(HOST_EXE)

$(HOST_EXE): host/llama3_attention_host.cpp
	@mkdir -p $(dir $@)
	g++ -std=c++17 -O2 -Wall -Wextra \
		-I$(XILINX_XRT)/include -L$(XILINX_XRT)/lib \
		-Wl,-rpath,$(XILINX_XRT)/lib -pthread -o $@ $< -lxrt_coreutil

run: $(HOST_EXE)
	./$(HOST_EXE) --xclbin $(XCLBIN)

clean:
	rm -rf host/*.exe build/

.PHONY: all run clean
