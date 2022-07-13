//===- HipPrintf.cpp -----------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM IR pass to convert calls to the CUDA/HIP printf() to OpenCL/SPIR-V
// compatible ones.
//
// (c) 2021-2022 Pekka Jääskeläinen / Parmance for Argonne National Laboratory
//===----------------------------------------------------------------------===//
//
// This pass moves the format string from global to the OpenCL constant address
// space before we pass the IR to SPIR-V emission, which is required by the
// SPIR-V's OpenCL profile.
//
// More annoyingly, in OpenCL 1.2 the printf string args must be pointers to
// _literal srings_ according to the specs.
// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
// printfFunction.html
//
// This deviates from the C99 printf (and thus CUDA) which allows dynamic
// string arguments. To implement this on top of the OpenCL printf, we split the
// format string such that whenever there's a %s we print the string using
// a printf("%c", c) loop and delegate the surrounding parts format string
// parts to the printf as is.
//
// E.g. printf("%d %s %d", ...) would be converted to a chain of calls:
// printf("%d ", ...), _putstr(...), ... printf(" %d", ...)
//
// This works as long as we can analyze the format string at compile time to
// detect the string arg positions,  thus dynamic _format_ strings are still
// not supported.
//
// However, since SPIR-V doesn't mention a need of the string arguments to
// be in the constant space, we by default let them pass since implementations
// appear to allow it.
//
// Also counts the number of format args for replacing the return value of
// the printf() call with it for rough CUDA-behavior emulation.
//
//===----------------------------------------------------------------------===//

#include "HipPrintf.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include <vector>
#include <string>
#include <iostream>

#define DEBUG_TYPE "hip-printf"

#define SPIRV_OPENCL_CONSTANT_AS 2
#define SPIRV_OPENCL_GENERIC_AS 4
#define SPIRV_OPENCL_PRINTF_FMT_ARG_AS SPIRV_OPENCL_CONSTANT_AS

#define ORIG_PRINTF_FUNC_NAME "_hip_printf"
#define ORIG_PRINT_STRING_FUNC_NAME "_cl_print_str"

// The SPIR-V specification doesn't especially forbid %s
// arguments that are not "literals" (in constant address
// space as specified by OpenCL). Assume a SPIR-V printf
// can print strings that are not moved to constant space
// and no special functions is needed for their printout.
// The SPIR-V implementations seen so far do not care about this.
#define ASSUME_PRINTF_SUPPORTS_GLOBAL_STRING_ARGS 0

using namespace llvm;

unsigned NumFormatSpecs(StringRef FmtString) {
  return FmtString.count("%") - 2 * FmtString.count("%%");
}

// Strip address space casts, GEPs etc. towards the global string
// literal and return it.
GlobalVariable *findGlobalStr(Value *Arg) {

  if (auto *GV = dyn_cast<GlobalVariable>(Arg))
    return GV;

  if (auto GEP = dyn_cast<GetElementPtrInst>(Arg)) {
    assert(GEP->hasAllZeroIndices());
    return findGlobalStr(GEP->getPointerOperand());
  }
  if (auto ASCast = dyn_cast<AddrSpaceCastInst>(Arg))
    return findGlobalStr(ASCast->getPointerOperand());
  else if (auto CE = dyn_cast<ConstantExpr>(Arg)) {
    if (CE->getOpcode() == llvm::Instruction::AddrSpaceCast ||
        CE->getOpcode() == llvm::Instruction::GetElementPtr)
      return findGlobalStr(CE->getOperand(0));
    else
      llvm_unreachable("Unexpected printf format string format!");
  }
  llvm_unreachable("Unrecognized instruction or constant expression.");
}

// Extracts the format string in the printf argument passed as FmtStrArg,
// and (possibly) splits it to a number of format strings if there are %s
// format args. In such a case the %s format string is returned as a separate
// "%s" format string. Also counts the total number of format specifiers in
// NumberOfFormatSpecs to returns it as a final return value.
static std::vector<std::string>
getFormatStringPieces(Value *FmtStrArg, unsigned &NumberOfFormatSpecs) {
  auto *OrigFmtStr = findGlobalStr(FmtStrArg);
  std::vector<std::string> FmtStrPieces;
  ConstantDataSequential *FmtStrData =
      dyn_cast<ConstantDataSequential>(OrigFmtStr->getInitializer());

  if (FmtStrData == nullptr) {
    assert(OrigFmtStr->getInitializer()->isZeroValue());
    FmtStrPieces.push_back("");
    NumberOfFormatSpecs = 0;
    return FmtStrPieces;
  }

  auto FmtString = FmtStrData->getAsString();

  // Counting the format specs is a more accurate approximation for the return
  // value than counting the passed arguments due to the dynamic width
  // specifier (*) which is given as an argument and should not be counted
  // in to the processed arguments (I believe).
  NumberOfFormatSpecs = NumFormatSpecs(FmtString);

  // FmtStrData has one byte larger size since it includes the trailing
  // byte, so if we convert it directly to a std::string, it gets a corrupted
  // string due to the wrong size passed to it (occurs at least with LLVM 13)?
  // TO CHECK again.
  std::string FmtStr = std::string(FmtStrData->getAsCString());

  // TODO: %s can also have * specifier. We have to treat it as a special
  // case, adding strlen() - width for padding.
  size_t Pos = 0;
  while ((Pos = FmtStr.find("%s")) != std::string::npos &&
         // Without this we'd handle %%s wrongly.
         !(Pos > 0 && FmtStr[Pos - 1] == '%')) {
    std::string TokenBefore = FmtStr.substr(0, Pos);
    if (TokenBefore != "")
      FmtStrPieces.push_back(TokenBefore);
    assert(FmtStr.size() > Pos + 2 - 1);
    FmtStr.erase(0, Pos + 2);
    FmtStrPieces.push_back("%s");
  }
  if (FmtStr != "")
    FmtStrPieces.push_back(FmtStr);
  return FmtStrPieces;
}

std::string getCDSAsString(GlobalVariable *OrigStr, bool &IsEmpty) {

  assert(OrigStr != nullptr && OrigStr->hasInitializer());

  ConstantDataSequential *CDSInitializer =
      dyn_cast<ConstantDataSequential>(OrigStr->getInitializer());

  assert(OrigStr->getInitializer()->isZeroValue() || CDSInitializer != nullptr);

  IsEmpty = OrigStr->getInitializer()->isZeroValue();
  return (IsEmpty ? "" : std::string(CDSInitializer->getAsCString()));
}

// Tries to convert the string argument to an OpenCL compatible one
// by creating a copy of it in the module global scope. Returns nullptr
// if unable to do so.
Value *HipPrintfToOpenCLPrintfPass::cloneStrArgToConstantAS(
    Value *StrArg, llvm::IRBuilder<> &B, bool *IsEmpty) {

  ConstantExpr *CE = dyn_cast<ConstantExpr>(StrArg);
  if (CE == nullptr)
    return nullptr;

  Value *StrOpr = CE->getOperand(0);

  GlobalVariable *OrigStr = findGlobalStr(StrOpr);
  if (OrigStr == nullptr || !OrigStr->hasInitializer())
    return nullptr;

  std::string NewStr = getCDSAsString(OrigStr, *IsEmpty);
  return getOrCreateStrLiteralArg(NewStr, B);
}

// Checks if the GlobalVariable a literal string.
bool isLiteralString(const GlobalVariable &Var) {

  if (!Var.isConstant())
    return false;

  auto ArrayTy = dyn_cast<ArrayType>(Var.getType()->getArrayElementType());
  if (!ArrayTy)
    return false;

  auto IntTy = dyn_cast<IntegerType>(ArrayTy->getArrayElementType());
  if (!IntTy)
    return false;

  return IntTy->getBitWidth() == 8;
}

// Create a new string literal that can be passed to an OpenCL printf
// as an argument. Try to reuse previously created ones, if possible.
Constant *
HipPrintfToOpenCLPrintfPass::getOrCreateStrLiteralArg(const std::string &Str,
                                                      llvm::IRBuilder<> &B) {

  auto &LiteralArg = LiteralArgs_[Str];
  if (LiteralArg != nullptr)
    return LiteralArg;

#if LLVM_VERSION_MAJOR <= 14
  auto *LiteralStr = B.CreateGlobalString(Str.c_str(), ".cl_printf_fmt_str",
                                          SPIRV_OPENCL_PRINTF_FMT_ARG_AS);

  IntegerType *Int64Ty = Type::getInt64Ty(M_->getContext());
  ConstantInt *Zero = ConstantInt::get(Int64Ty, 0);
  std::array<Constant *, 2> Indices = {Zero, Zero};

  PointerType *PtrTy =
      cast<PointerType>(LiteralStr->getType()->getScalarType());

  return LiteralArg = llvm::ConstantExpr::getGetElementPtr(
             PtrTy->getElementType(), LiteralStr, Indices);
#else
  return B.CreateGlobalString(Str.c_str(), ".cl_printf_fmt_str",
                              SPIRV_OPENCL_PRINTF_FMT_ARG_AS);
#endif
}

Function *HipPrintfToOpenCLPrintfPass::getOrCreatePrintStringF() {

  if (GlobalValue *OldPrintStrF =
          M_->getNamedValue(ORIG_PRINT_STRING_FUNC_NAME))
    return cast<Function>(OldPrintStrF);

  auto *Int8Ty = IntegerType::get(M_->getContext(), 8);
  PointerType *GenericCStrArgT =
      PointerType::get(Int8Ty, SPIRV_OPENCL_GENERIC_AS);

  FunctionType *PrintStrFTy = FunctionType::get(
      Type::getVoidTy(M_->getContext()), {GenericCStrArgT}, false);

  FunctionCallee PrintStrF =
      M_->getOrInsertFunction(ORIG_PRINT_STRING_FUNC_NAME, PrintStrFTy);
  cast<Function>(PrintStrF.getCallee())
      ->setCallingConv(llvm::CallingConv::SPIR_FUNC);
  return cast<Function>(PrintStrF.getCallee());
}

// Get called function from 'CI' call or return nullptr the call is indirect.
static Function* getCalledFunction(CallInst *CI, const LLVMContext &Ctx) {
  assert(CI);
  if (auto *Callee = CI->getCalledFunction())
    return Callee;

  if (CI->isIndirectCall())
    return nullptr;
  // A call with mismatched call signature.

#if LLVM_VERSION_MAJOR > 14
  if (Ctx.hasSetOpaquePointersValue())
    return cast<Function>(CI->getCalledOperand());
#endif

  auto *CalledOp = dyn_cast<ConstantExpr>(CI->getCalledOperand());
  assert(CalledOp && CalledOp->getOpcode() == Instruction::BitCast);
  return cast<Function>(CalledOp->getOperand(0));
}

PreservedAnalyses HipPrintfToOpenCLPrintfPass::run(Module &Mod,
                                                   ModuleAnalysisManager &AM) {

  M_ = &Mod;
  LiteralArgs_.clear();

  GlobalValue *Printf = Mod.getNamedValue("printf");
  GlobalValue *HipPrintf = Mod.getNamedValue(ORIG_PRINTF_FUNC_NAME);

  // No printf decl in the module, no printf calls to handle.
  if (Printf == nullptr)
    return PreservedAnalyses::all();
  LLVM_DEBUG(dbgs() << "Found printf decl: "; Printf->dump());

  Function *PrintfF = cast<Function>(Printf);

  LLVMContext &Ctx = Mod.getContext();
  auto *Int8Ty = IntegerType::get(Ctx, 8);
  auto *Int32Ty = IntegerType::get(Ctx, 32);

  PointerType *ConstStrPtrT =
      PointerType::get(Int8Ty, SPIRV_OPENCL_CONSTANT_AS);

  PointerType *OCLPrintfFmtArgT = ConstStrPtrT;

  FunctionType *OpenCLPrintfTy =
      FunctionType::get(Int32Ty, {OCLPrintfFmtArgT}, true);

  FunctionCallee OpenCLPrintfF;

  if (HipPrintf == nullptr) {
    // Create the OpenCL printf decl which will be used instead. Rename the
    // old one away to _hip_printf.
    PrintfF->setName(ORIG_PRINTF_FUNC_NAME);
    HipPrintf = PrintfF;
    OpenCLPrintfF = Mod.getOrInsertFunction(
        "printf", OpenCLPrintfTy, cast<Function>(HipPrintf)->getAttributes());
    Function *PrintfF = cast<Function>(OpenCLPrintfF.getCallee());
    PrintfF->setCallingConv(llvm::CallingConv::SPIR_FUNC);
    PrintfF->setVisibility(llvm::GlobalValue::HiddenVisibility);
  } else {
    OpenCLPrintfF = FunctionCallee(OpenCLPrintfTy, PrintfF);
  }

  bool Modified = false;

  for (auto &F : Mod) {
    SmallPtrSet<Instruction *, 8> EraseList;
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI)
          continue;
        auto *Callee = getCalledFunction(CI, Ctx);
        if (!Callee)
          continue; // CI is indirect call.
        if (Callee->getName() != ORIG_PRINTF_FUNC_NAME)
          continue;

        LLVM_DEBUG(dbgs() << "Original printf call: "; CI->dump());

        CallInst &OrigCall = cast<CallInst>(I);
        unsigned TotalFmtSpecCount;
        auto FmtSpecPieces =
            getFormatStringPieces(*OrigCall.args().begin(), TotalFmtSpecCount);

        IRBuilder<> B(&I);

        auto OrigCallArgs = OrigCall.args();
        auto OrigCallArgI = OrigCallArgs.begin();
        // Skip the original format string arg and recreate it.
        OrigCallArgI++;

        for (auto FmtStr : FmtSpecPieces) {
          unsigned FormatSpecCount = NumFormatSpecs(FmtStr);
          std::vector<Value *> Args;

          // TODO: handle a (compile time known) null ptr format string arg and
          // return -1 like CUDA does.
          if (FmtStr == "") {
            // No output, the return value suffices.
            continue;
          } else if (FmtStr == "%s" || FmtStr == "%*s") {
            // This is a string printout, we do not pass the format string
            // in that case, but just call a string printer.
            // We can use the normal printf if the %s arg can be resolved to
            // a literal C string. However, it must be moved to constant AS
            // for OpenCL compatibility.
            Value *OrigArg = *OrigCallArgI++;
            bool IsEmpty = false;
            if (Value *ConstantSpaceCStr =
                    cloneStrArgToConstantAS(OrigArg, B, &IsEmpty)) {
              if (IsEmpty)
                continue; // empty str arg to a %s, no output needed
              // We could copy the arg to constant space, printf it
              // directly. No format string needed.
              Args.push_back(ConstantSpaceCStr);
              CallInst::Create(OpenCLPrintfF, Args, "", &OrigCall);
            } else if (ASSUME_PRINTF_SUPPORTS_GLOBAL_STRING_ARGS) {
              if (IsEmpty)
                continue;
              // Create a constant space format string for %s.
              Args.push_back(getOrCreateStrLiteralArg("%s", B));
              // ...and then assume the data arg can point to a generic
              // address space string (eventually residing in global AS).
              Args.push_back(OrigArg);
              CallInst::Create(OpenCLPrintfF, Args, "", &OrigCall);
            } else {
              Args.push_back(OrigArg);
              CallInst::Create(getOrCreatePrintStringF(), Args, "", &OrigCall);
            }
            continue;
          }

          // Handle as a normal printf() call.
          Args.push_back(getOrCreateStrLiteralArg(FmtStr, B));
          while (FormatSpecCount--) {
            assert(OrigCallArgI != OrigCallArgs.end());
            Value *OrigArg = *OrigCallArgI++;
            Args.push_back(OrigArg);
          }
          CallInst::Create(OpenCLPrintfF, Args, "", &OrigCall);
        }

        // Instead of returning the success/failure from the OpenCL printf(),
        // assume that the parsing succeeds and return the number of format
        // strings. A slight improvement would be to return 0 in case of a
        // failure, but it still would not necessarily match CUDA nor HIP
        // since it should return the number of _valid_ format replacements.
        IntegerType *Int32Ty = Type::getInt32Ty(Ctx);
        ConstantInt *RV = ConstantInt::get(Int32Ty, TotalFmtSpecCount);
        OrigCall.replaceAllUsesWith(RV);

        EraseList.insert(&I);
      }
    }
    for (auto I : EraseList)
      I->eraseFromParent();
    Modified |= EraseList.size() > 0;
  }

  SmallPtrSet<GlobalVariable *, 8> UnusedGlobals;
  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

namespace {

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-printf", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-printf") {
                    FPM.addPass(HipPrintfToOpenCLPrintfPass());
                    return true;
                  }
                  return false;
                });
          }};
}
} // namespace
