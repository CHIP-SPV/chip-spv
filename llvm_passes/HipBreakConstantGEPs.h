//===- BreakConstantGEPs.h - Change constant GEPs into GEP instructions --- --//
// 
// pocl note: This pass is taken from The SAFECode project with trivial modifications.
//            Automatic locals might cause constant GEPs which cause problems during 
//            converting the locals to kernel function arguments for thread safety.
//
//                          The SAFECode Compiler 
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass changes all GEP constant expressions into GEP instructions.  This
// permits the rest of SAFECode to put run-time checks on them if necessary.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_BREAKCONSTANTGEPS_H
#define LLVM_PASSES_BREAKCONSTANTGEPS_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

//
// Pass: BreakConstantGEPs
//
// Description:
//  This pass modifies a function so that it uses GEP instructions instead of
//  GEP constant expressions.
//
#if LLVM_VERSION_MAJOR > 11
class HipBreakConstantGEPsPass : public PassInfoMixin<HipBreakConstantGEPsPass> {
  public:
    PreservedAnalyses run (Module &M, ModuleAnalysisManager &AM);
    static bool isRequired() { return true; }
};
#endif

#endif

