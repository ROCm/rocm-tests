# Cross-GPU dma-buf pin userspace binaries (built on the test node).
HIPCC ?= hipcc
ROCM_PATH ?= /opt/rocm
REPLICATE := replicate_count_dmabufsize
FAKE_4MB := fake_oom_4mb

ifeq ($(wildcard $(ROCM_PATH)/include/hip/hip_runtime.h),)
  ifneq ($(wildcard /opt/rocm-10.3.0/include/hip/hip_runtime.h),)
    ROCM_PATH := /opt/rocm-10.3.0
  endif
endif

CPPFLAGS := -I$(ROCM_PATH)/include -D__HIP_PLATFORM_AMD__
LDFLAGS := -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib
CXXFLAGS := -O2 -Wall -Wextra -std=c++17

.PHONY: all clean

all: $(REPLICATE) $(FAKE_4MB)

$(REPLICATE): replicate_count_dmabufsize.cpp pin_dmabuf_kmod/pin_dmabuf.h
	$(HIPCC) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $< -lamdhip64 -lhsa-runtime64

$(FAKE_4MB): fake_oom_4mb.cpp pin_dmabuf_kmod/pin_dmabuf.h
	$(HIPCC) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $< -lamdhip64 -lhsa-runtime64

clean:
	rm -f $(REPLICATE) $(FAKE_4MB)
