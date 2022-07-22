// Define a pass plugin that runs a collection of HIP passes.

#include "HipAbort.h"
#include "HipDefrost.h"
#include "HipDynMem.h"
#include "HipStripUsedIntrinsics.h"
#include "HipPrintf.h"
#include "HipGlobalVariables.h"
#include "HipTextureLowering.h"
#include "HipBreakConstantGEPs.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Scalar/SROA.h"


using namespace llvm;

// A pass that removes noinline and optnone attributes from functions.
class RemoveNoInlineOptNoneAttrsPass
    : public PassInfoMixin<RemoveNoInlineOptNoneAttrsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    for (auto &F : M) {
      F.removeFnAttr(Attribute::NoInline);
      F.removeFnAttr(Attribute::OptimizeNone);
    }
    return PreservedAnalyses::none();
  }
  static bool isRequired() { return true; }
};

static void addFullLinkTimePasses(ModulePassManager &MPM) {
  // Remove attributes that may prevent the device code from being optimized.
  MPM.addPass(RemoveNoInlineOptNoneAttrsPass());

  // replace constantexpr GEPs with GEP instructions
  MPM.addPass(HipBreakConstantGEPsPass());

  // transform dynamic shared memory (extern __shared__) global variables into kernel arguments
  MPM.addPass(HipDynMemExternReplaceNewPass());

  // Prepare device code for texture function lowering which does not yet work
  // on non-inlined code and local variables of hipTextureObject_t type.
  MPM.addPass(RemoveNoInlineOptNoneAttrsPass());
  // Increase getInlineParams argument for more aggressive inlining.
  MPM.addPass(ModuleInlinerWrapperPass(getInlineParams(1000)));
#if LLVM_VERSION_MAJOR < 14
  MPM.addPass(createModuleToFunctionPassAdaptor(SROA()));
#else
  MPM.addPass(createModuleToFunctionPassAdaptor(SROAPass()));
#endif

  MPM.addPass(HipTextureLoweringPass());

  // TODO: Update printf pass for HIP-Clang 14+. It now triggers an assert:
  //
  //  Assertion `isa<X>(Val) && "cast<Ty>() argument of incompatible type!"'
  //  failed.
  MPM.addPass(HipPrintfToOpenCLPrintfPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(HipDefrostPass()));
  MPM.addPass(HipAbortPass());
  // This pass must appear after HipDynMemExternReplaceNewPass.
  MPM.addPass(HipGlobalVariablesPass());

  // Remove dead code left over by HIP lowering passes and kept alive by
  // llvm.used and llvm.compiler.used intrinsic variable.
  MPM.addPass(HipStripUsedIntrinsicsPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(DCEPass()));
  MPM.addPass(GlobalDCEPass());
}

#if LLVM_VERSION_MAJOR < 14
#define PASS_ID "hip-link-time-passes"
#else
#define PASS_ID "hip-post-link-passes"
#endif

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-passes", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_ID) {
                    addFullLinkTimePasses(MPM);
                    return true;
                  }
                  return false;
                });
          }};
}
