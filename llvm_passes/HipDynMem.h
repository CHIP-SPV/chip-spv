
#ifndef LLVM_PASSES_HIP_DYN_MEM_H
#define LLVM_PASSES_HIP_DYN_MEM_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR > 11
class HipDynMemExternReplaceNewPass
    : public PassInfoMixin<HipDynMemExternReplaceNewPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};
#endif

#endif
