#ifndef LLAMA3_RUNTIME_STUBS_HPP
#define LLAMA3_RUNTIME_STUBS_HPP

#include <stdlib.h>

#if !defined(__X86SIM__)
extern "C" __attribute__((noinline, used)) void __cxa_finalize(void*) {}
#endif

#endif
