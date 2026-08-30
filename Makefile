XILINX_XRT ?= /opt/xilinx/xrt
VITIS_ROOT ?= /s3/Xilinx_Vitis_2022.2/Vitis/2022.2
HOST_EXE := host/llama3_attention_host.exe
XCLBIN ?= prebuilt/vck5000/llama3_attention.xclbin

all: $(HOST_EXE)

$(HOST_EXE): host/llama3_attention_host.cpp
	@mkdir -p $(dir $@)
	g++ -std=c++17 -O2 -Wall -Wextra \
		-I$(XILINX_XRT)/include -L$(XILINX_XRT)/lib \
		-Wl,-rpath,$(XILINX_XRT)/lib -pthread -o $@ $< -lxrt_coreutil

run: $(HOST_EXE)
	./$(HOST_EXE) --xclbin $(XCLBIN) --batch 1 --warmup 0 --runs 1 \
		--seed 7 --verify --profile

clean:
	rm -rf host/*.exe build/

.PHONY: all run clean
