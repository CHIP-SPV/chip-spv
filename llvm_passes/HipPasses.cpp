// Define a pass plugin that runs a collection of HIP passes.

#include "HipAbort.h"
#include "HipDefrost.h"
#include "HipDynMem.h"
#include "HipStripUsedIntrinsics.h"
#include "HipPrintf.h"
#include "HipGlobalVariables.h"
#include "HipTextureLowering.h"

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

// LLVM commit 15a1769631ff0b2b3e830b03e51ae5f54f08a0ab introduces
// 'opencl.ocl.version' module metadata into device code. This triggers an
// assertion in SPIRV-LLVM Translator if the HIP device code and linked device
// bitcode has mixed OpenCL version metadata. The commit in question inserts
// OpenCL version 0.0 in HIP compilation mode (in CUDA mode it is 2.0).
//
// This pass works around the issue until some fix is introduced in Clang. The
// issue is fixed by setting OpenCL version to the same as bitcode library
// (2.0).
class HipFixOpenCLMDPass : public PassInfoMixin<HipFixOpenCLMDPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    constexpr auto OCLVersionMDName = "opencl.ocl.version";
    if (auto *OCLVersionMD = M.getNamedMetadata(OCLVersionMDName)) {
      auto &Ctx = M.getContext();
      auto *Int32Ty = IntegerType::get(Ctx, 32);
      M.eraseNamedMetadata(OCLVersionMD);
      Metadata *OCLVerElts[] = {
          ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 2)),
          ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 0))};
      OCLVersionMD = M.getOrInsertNamedMetadata(OCLVersionMDName);
      OCLVersionMD->addOperand(MDNode::get(Ctx, OCLVerElts));
    }
    // Altering OpenCL metadata probably does not invalidate any analyses.
    return PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

static void addFullLinkTimePasses(ModulePassManager &MPM) {
  // Remove attributes that may prevent the device code from being optimized.
  MPM.addPass(RemoveNoInlineOptNoneAttrsPass());

  // Run a collection of passes run at device link time.
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

  MPM.addPass(HipFixOpenCLMDPass());
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
