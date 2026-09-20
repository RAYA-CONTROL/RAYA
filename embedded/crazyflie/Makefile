# Crazyflie 2.1 Brushless out-of-tree RAYA controller.
CRAZYFLIE_BASE ?= $(abspath ../../third_party/crazyflie-firmware)
RAYA_ROOT := $(abspath ../..)
TRAJECTORY ?= 4
ACTIVE_HORIZON ?= 1
BENCH_PROFILE ?= 0
NUMERIC_O2 ?= 0

ifeq ($(shell uname -s),Darwin)
RAYA_COREUTILS := $(shell brew --prefix coreutils 2>/dev/null)/libexec/gnubin
RAYA_GNU_SED := $(shell brew --prefix gnu-sed 2>/dev/null)/libexec/gnubin
ifeq ($(wildcard $(RAYA_COREUTILS)/readlink),)
$(error macOS builds require Homebrew coreutils)
endif
ifeq ($(wildcard $(RAYA_GNU_SED)/sed),)
$(error macOS builds require Homebrew gnu-sed)
endif
export PATH := $(RAYA_GNU_SED):$(RAYA_COREUTILS):$(PATH)
endif

ifneq ($(filter $(TRAJECTORY),0 1 2 3 4),$(TRAJECTORY))
$(error TRAJECTORY must be 0, 1, 2, 3, or 4)
endif
ifneq ($(filter $(BENCH_PROFILE),0 1),$(BENCH_PROFILE))
$(error BENCH_PROFILE must be 0 or 1)
endif
ifneq ($(filter $(NUMERIC_O2),0 1),$(NUMERIC_O2))
$(error NUMERIC_O2 must be 0 or 1)
endif

EXTRA_CFLAGS += -I$(CURDIR)/include -I$(RAYA_ROOT)/third_party/eigen
EXTRA_CFLAGS += -I$(CURDIR)/solver
EXTRA_CFLAGS += -DEIGEN_INITIALIZE_MATRICES_BY_ZERO -DEIGEN_NO_MALLOC
EXTRA_CFLAGS += -DNDEBUG -DEIGEN_FAST_MATH -Wno-error
OOT_CONFIG := $(CURDIR)/app-config
OOT_USES_CXX := 1
OOT_ARGS := -C $(CRAZYFLIE_BASE) OOT=$(CURDIR) EXTRA_CFLAGS="$(EXTRA_CFLAGS)"
OOT_ARGS += LDFLAGS="--specs=nosys.specs --specs=nano.specs -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -nostdlib"
OOT_ARGS += TRAJECTORY=$(TRAJECTORY) ACTIVE_HORIZON=$(ACTIVE_HORIZON)
OOT_ARGS += BENCH_PROFILE=$(BENCH_PROFILE) NUMERIC_O2=$(NUMERIC_O2)
include $(CRAZYFLIE_BASE)/tools/make/oot.mk
