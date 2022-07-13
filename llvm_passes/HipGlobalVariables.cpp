// TODO: copyright.

#include "HipGlobalVariables.h"

#include "../src/common.hh"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#define DEBUG_TYPE "hip-lower-gv"

using namespace llvm;

namespace {

using GVarMapT = std::map<GlobalVariable *, GlobalVariable *>;
using Const2InstMapT = std::map<Constant *, Instruction *>;

// SPIR-V address spaces.
constexpr unsigned SpirvCrossWorkGroupAS = 1;

// Create kernel function stub, returns its return instruction.
static Instruction *createKernelStub(Module &M, StringRef Name,
                                     ArrayRef<Type *> ArgTypes = None) {
  Function *F = cast<Function>(
      M.getOrInsertFunction(
           Name,
           FunctionType::get(Type::getVoidTy(M.getContext()), ArgTypes, false))
          .getCallee());
  F->setCallingConv(CallingConv::SPIR_KERNEL);
  // HIP-CLang marks kernels hidden. Do the same here for consistency.
  F->setVisibility(GlobalValue::HiddenVisibility);
  assert(F->empty() && "Function name clash?");
  IRBuilder<> B(BasicBlock::Create(M.getContext(), "entry", F));
  return B.CreateRetVoid();
}

// Emit a shadow kernel for relaying properties about the original variable.
static void emitGlobalVarInfoShadowKernel(Module &M,
                                          const GlobalVariable *GVar) {
  // Emit the following shadow kernel in pseudo code:
  //
  // void <ChipVarInfoPrefix><GVar-name>(int64_t *info) {
  //   info[0] = <GVar-size>;      // In bytes.
  //   info[1] = <GVar-alignment>; // In bytes.
  //   info[2] = <HasInitializer>; // [0, 1].
  // }

  auto Name = std::string(ChipVarInfoPrefix) + GVar->getName().str();
  IRBuilder<> Builder(createKernelStub(
      M, Name, {Type::getInt64PtrTy(M.getContext(), SpirvCrossWorkGroupAS)}));

  const auto &DL = M.getDataLayout();
  auto *InfoArg = Builder.GetInsertBlock()->getParent()->getArg(0);

  // info[0] = <GVar-size>;
  auto Size = DL.getTypeStoreSize(GVar->getValueType());
  Builder.CreateStore(Builder.getInt64(Size), InfoArg);

  // info[1] = <GVar-alignment>;
  uint64_t Alignment = GVar->getAlign().valueOrOne().value();
  Value *Ptr =
      Builder.CreateConstInBoundsGEP1_64(Builder.getInt64Ty(), InfoArg, 1);
  Builder.CreateStore(Builder.getInt64(Alignment), Ptr);

  // info[2] = <HasInitializer>;
  Ptr = Builder.CreateConstInBoundsGEP1_64(Builder.getInt64Ty(), InfoArg, 2);
  Builder.CreateStore(Builder.getInt64(GVar->hasInitializer()), Ptr);
}

// Emit a shadow kernel for setting the transformed global variable to point to
// the actual allocation.
static void emitGlobalVarBindShadowKernel(Module &M, GlobalVariable *GVar,
                                          const GlobalVariable *OriginalGVar) {
  // Emit the following shadow kernel in pseudo code:
  //
  //   void<ChipVarBindPrefix><GVar-name>(void *buffer) {
  //     <ChipVarPrefix><GVar-name> = (<GVar-type>)buffer;
  //   }

  auto Name = std::string(ChipVarBindPrefix) + OriginalGVar->getName().str();
  IRBuilder<> Builder(createKernelStub(
      M, Name, {Type::getInt8PtrTy(M.getContext(), SpirvCrossWorkGroupAS)}));
  Value *BindArg = Builder.GetInsertBlock()->getParent()->getArg(0);
  BindArg = Builder.CreatePointerBitCastOrAddrSpaceCast(BindArg,
                                                        GVar->getValueType());
  Builder.CreateStore(BindArg, GVar);
}

// Returns a constant expression rewritten as instructions if needed.
//
// Global variable references found in the GVarMap are replaced with a load from
// the mapped pointer value.  New instructions will be added at Builder's
// current insertion point.
static Value *expandConstant(Constant *C, GVarMapT &GVarMap,
                             IRBuilder<> &Builder, Const2InstMapT &InsnCache) {
  if (InsnCache.count(C)) return InsnCache[C];

  if (isa<ConstantData>(C)) return C;

  if (auto *GVar = dyn_cast<GlobalVariable>(C)) {
    if (GVarMap.count(GVar)) {
      // Replace with pointer load. All constant expressions depending
      // on this will be rewritten as instructions.
      auto *NewGVar = GVarMap[GVar];
      auto *LD = Builder.CreateLoad(NewGVar->getValueType(), NewGVar);
      InsnCache[GVar] = LD;
      return LD;
    }
    return GVar;
  }

  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    SmallVector<Value *, 4> Ops;  // Collect potentially expanded operands.
    bool AnyOpExpanded = false;
    for (Value *Op : CE->operand_values()) {
      Value *V =
          expandConstant(cast<Constant>(Op), GVarMap, Builder, InsnCache);
      Ops.push_back(V);
      AnyOpExpanded |= !isa<Constant>(V);
    }

    if (!AnyOpExpanded) return CE;

    auto *AsInsn = Builder.Insert(CE->getAsInstruction());
    // Replace constant operands with expanded ones.
    for (auto &U : AsInsn->operands()) U.set(Ops[U.getOperandNo()]);
    InsnCache[CE] = AsInsn;
    return AsInsn;
  }

  llvm_unreachable("Unexpected constant kind.");
}

// Emit a shadow kernel for initialing the global variable.
static void emitGlobalVarInitShadowKernel(Module &M, GlobalVariable *GVar,
                                          GlobalVariable *OriginalGVar,
                                          GVarMapT GVarMap) {
  // Emit the following shadow kernel in pseudo code:
  //
  //  void <ChipVarInitPrefix><GVar-name>() {
  //    *<GVar> = <initializer>;
  //  }

  assert(GVar->getValueType()->isPointerTy());
  assert(OriginalGVar->hasInitializer());

  auto Name = std::string(ChipVarInitPrefix) + OriginalGVar->getName().str();
  IRBuilder<> Builder(createKernelStub(M, Name, {}));

  // Initializers are constant expressions.  If they have references to a global
  // variables we are going to replace with load instructions we need to rewrite
  // the constant expression as instructions.
  Const2InstMapT Cache;
  Value *Init =
      expandConstant(OriginalGVar->getInitializer(), GVarMap, Builder, Cache);

  // *<GVar>
  Value *Ptr = Builder.CreateLoad(GVar->getValueType(), GVar);

  // *<GVar> = <initializer>;
  Builder.CreateStore(Init, Ptr);
}

static bool shouldLower(const GlobalVariable &GVar) {
  if (!GVar.hasName()) return false;

  if (GVar.getName().startswith(ChipVarPrefix))
    return false;  // Already lowered.

  // All host accessible global device variables are marked to be externally
  // initialized and does not have COMDAT (so far).
  if (!GVar.isExternallyInitialized() || GVar.hasComdat())
    return false;

  // String literals get an unnamed_addr attribute, we know by it to
  // skip them.
  if (GVar.hasAtLeastLocalUnnamedAddr())
    return false;

  // Only objects in cross-workgroup address space are considered. LLVM IR
  // straight out from the HIP-Clang does not have objects in constant address
  // space so we don't look for them.
  if (GVar.getAddressSpace() != SpirvCrossWorkGroupAS) return false;

  // Catch globals with unexpected attributes.
  assert(!GVar.isThreadLocal());

  return true;
}

static void findInstructionUsesImpl(Use &U, std::vector<Use *> &Uses,
                                    std::set<Use *> &Visited) {
  if (Visited.count(&U))
    return;
  Visited.insert(&U);

  assert(isa<Constant>(*U));
  if (isa<Instruction>(U.getUser())) {
    Uses.push_back(&U);
    return;
  }
  if (isa<Constant>(U.getUser())) {
    for (auto &U : U.getUser()->uses())
      findInstructionUsesImpl(U, Uses, Visited);
    return;
  }

  // Catch other user kinds - we may need to process them (somewhere but not
  // here).
  llvm_unreachable("Unexpected user kind.");
}

// Return list of non-constant leaf use edges whose users are instructions.
static std::vector<Use *> findInstructionUses(GlobalVariable *GVar) {
  std::vector<Use *> Uses;
  std::set<Use *> Visited;
  for (auto &U : GVar->uses())
    findInstructionUsesImpl(U, Uses, Visited);
  return Uses;
}

static void replaceGlobalVariableUses(GVarMapT &GVarMap) {
  std::map<Function *, Const2InstMapT> Fn2InsnCache;
  std::map<Function *, std::unique_ptr<IRBuilder<>>> Fn2Builder;

  auto getBuilder = [&Fn2Builder](Function *F) -> IRBuilder<> & {
    auto &BuilderPtr = Fn2Builder[F];
    if (!BuilderPtr) {
      auto &E = F->getEntryBlock();
      auto InsPt = E.getFirstInsertionPt();
      // Put insertion point after allocas. SPIRV-LLVM translator panics (or at
      // least used to) if all allocas are not put in the entry block as the
      // first instructions.
      while (isa<AllocaInst>(*InsPt)) InsPt = std::next(InsPt);
      BuilderPtr = std::make_unique<IRBuilder<>>(&E, InsPt);
    }
    return *BuilderPtr;
  };

  for (auto pair : GVarMap) {
    GlobalVariable *Old = pair.first;
    GlobalVariable *New = pair.second;
    LLVM_DEBUG(dbgs() << "Replace: " << Old->getName() << "\n";
               dbgs() << "   with: load from" << New->getName() << "\n";);
    for (auto *U : findInstructionUses(Old)) {
      auto *IUser = cast<Instruction>(U->getUser());
      auto *FnUser = IUser->getParent()->getParent();
      auto &Builder = getBuilder(FnUser);
      auto &Cache = Fn2InsnCache[FnUser];
      LLVM_DEBUG(dbgs() << "in user: "; IUser->print(dbgs()); dbgs() << "\n";);
      Value *V = expandConstant(cast<Constant>(*U), GVarMap, Builder, Cache);
      U->set(V);
    }
  }
}

static void eraseMappedGlobalVariables(GVarMapT &GVarMap) {
  for (auto &pair : GVarMap) {
    auto *OldGVar = pair.first;
    if (OldGVar->hasNUses(0) ||
        // There might still be constantExpr users but no instructions should
        // depend on them.
        findInstructionUses(OldGVar).size() == 0) {
      OldGVar->replaceAllUsesWith(PoisonValue::get(OldGVar->getType()));
      OldGVar->eraseFromParent();
    } else
      // A non-instruction and non-constantExpr user?
      llvm_unreachable("Original variable still has uses!");
  }
}

static GlobalVariable *emitIndirectGlobalVariable(Module &M,
                                                  GlobalVariable *GVar) {
  // Create new global variable.
  assert(GVar->hasName());
  auto NewGVarName = std::string(ChipVarPrefix) + GVar->getName().str();
  auto *NewGVarTy = PointerType::get(GVar->getValueType(),
                                     GVar->getType()->getAddressSpace());
  GlobalVariable *NewGVar = new GlobalVariable(
      M, NewGVarTy, GVar->isConstant(), GVar->getLinkage(),
      Constant::getNullValue(NewGVarTy), NewGVarName, (GlobalVariable *)nullptr,
      GVar->getThreadLocalMode(), SpirvCrossWorkGroupAS,
      GVar->isExternallyInitialized());
  // Original GVars emitted by HIP-Clang are hidden. Make new GVars hidden too
  // for consistency.
  NewGVar->setVisibility(GlobalValue::HiddenVisibility);

  // Transfer debug info.
  SmallVector<DIGlobalVariableExpression *, 1> DIExprs;
  GVar->getDebugInfo(DIExprs);
  for (auto *DI : DIExprs) NewGVar->addDebugInfo(DI);

  return NewGVar;
}

static GVarMapT emitIndirectGlobalVariables(Module &M) {
  GVarMapT GVarMap;
  for (GlobalVariable &GVar : M.globals())
    if (shouldLower(GVar)) {
      auto *NewGVar = emitIndirectGlobalVariable(M, &GVar);
      GVarMap.insert(std::make_pair(&GVar, NewGVar));
    }
  return GVarMap;
}

// Find global device variables that are not host accessible but which should be
// reinitialized on hipDeviceReset() call - for example, static local variables.
static std::vector<GlobalVariable *> findResettableNonSymbolGVs(Module &M) {
  std::vector<GlobalVariable *> Result;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.hasSection()) // Non-user defined variable - e.g. llvm.used
                         // intrinsic.
      continue;
    // So far, all host-inaccessible global device variables either has a COMDAT
    // section or lacks the externally_initialized attribute.
    if (GV.isExternallyInitialized() && !GV.hasComdat())
      continue;
    if (!GV.hasInitializer() || GV.isConstant())
      continue;
    LLVM_DEBUG(dbgs() << "Host-inaccessible resettable GV: " << GV);
    Result.push_back(&GV);
  }
  return Result;
}

// Emit a kernel for resetting GVs back to their initialization value.
// Returns true if any code emitted and false otherwise.
bool emitNonSymbolInitializerKernel(const std::vector<GlobalVariable *> GVs,
                                    Module &M) {
  if (GVs.empty())
    return false;
  IRBuilder<> Builder(createKernelStub(M, ChipNonSymbolResetKernelName));
  for (auto *GV : GVs) {
    assert(GV->hasInitializer());
    Builder.CreateStore(GV->getInitializer(), GV);
  }
  return true;
}

static bool lowerGlobalVariables(Module &M) {
  bool Changed = false;

  // Lower host accessible global device variables.
  GVarMapT GVarMap = emitIndirectGlobalVariables(M);
  if (!GVarMap.empty()) {
    for (auto Kv : GVarMap) {
      emitGlobalVarInfoShadowKernel(M, Kv.first);
      emitGlobalVarBindShadowKernel(M, Kv.second, Kv.first);
      if (Kv.first->hasInitializer())
        emitGlobalVarInitShadowKernel(M, Kv.second, Kv.first, GVarMap);
    }
    replaceGlobalVariableUses(GVarMap);
    eraseMappedGlobalVariables(GVarMap);
    Changed |= true;
  }

  // Lower global device variables which are not accessible by the host but
  // should be reset on hipDeviceReset() call. For example: static function
  // local variables.
  auto NonSymbolGVs = findResettableNonSymbolGVs(M);
  Changed |= emitNonSymbolInitializerKernel(NonSymbolGVs, M);

  return Changed;
}
}  // namespace

PreservedAnalyses HipGlobalVariablesPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  return lowerGlobalVariables(M) ? PreservedAnalyses::none()
                                 : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-lower-gv", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-lower-gv") {
                    MPM.addPass(HipGlobalVariablesPass());
                    return true;
                  }
                  return false;
                });
          }};
}
