
#ifndef LLVM_PASSES_HIP_DEFROST_H
#define LLVM_PASSES_HIP_DEFROST_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR > 11
class HipDefrostPass
    : public PassInfoMixin<HipDefrostPass> {
public:
  PreservedAnalyses run(Function &M, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};
#endif

#endif
