// LLVM Pass to replace dynamically sized shared arrays ("extern __shared__ type[]")
// with a function argument. This is required because CUDA/HIP use a "magic variable"
// for dynamically sized shared memory, while OpenCL API uses a kernel argument

#include "HipDynMem.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <iostream>
#include <set>

using namespace llvm;

#define SPIR_LOCAL_AS 3
#define GENERIC_AS 4

typedef llvm::SmallPtrSet<Function *, 16> FSet;
typedef llvm::SetVector<Function *> OrderedFSet;

class HipDynMemExternReplacePass : public ModulePass {
private:

  static bool isGVarUsedInFunction(GlobalVariable *GV, Function *F) {
    for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
      for (BasicBlock::iterator i = BB->begin(); i != BB->end(); ++i) {
        //
        // Scan through the operands of this instruction & check for GV
        //
        Instruction * I = &*i;
        for (unsigned index = 0; index < I->getNumOperands(); ++index) {
          if (GlobalVariable *ArgGV = dyn_cast<GlobalVariable>(I->getOperand(index))) {
            if (ArgGV == GV)
              return true;
          }
        }
      }
    }
    return false;
  }

  static void replaceGVarUsesWith(GlobalVariable *GV, Function *F, Value *Repl) {
    SmallVector<unsigned, 8> OperToReplace;
    for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
      for (BasicBlock::iterator i = BB->begin(); i != BB->end(); ++i) {
        //
        // Scan through the operands of this instruction & check for GV
        //
        Instruction * I = &*i;
        OperToReplace.clear();
        for (unsigned index = 0; index < I->getNumOperands(); ++index) {
          if (GlobalVariable *ArgGV = dyn_cast<GlobalVariable>(I->getOperand(index))) {
            if (ArgGV == GV)
              OperToReplace.push_back(index);
          }
        }
        for (unsigned index : OperToReplace) {
          I->setOperand(index, Repl);
        }
      }
    }
  }

  static void recursivelyFindDirectUsers(Value *V, FSet &FS) {
    for (auto U : V->users()) {
      Instruction *Inst = dyn_cast<Instruction>(U);
      if (Inst) {
        Function *IF = Inst->getFunction();
        if (!IF)
          continue;
        FS.insert(IF);
      } else {
        recursivelyFindDirectUsers(U, FS);
      }
    }
  }

  static void recursivelyFindIndirectUsers(Value *V, OrderedFSet &FS) {
    OrderedFSet Temp;
    for (auto U : V->users()) {
      Instruction *Inst = dyn_cast<Instruction>(U);
      if (Inst) {
        Function *IF = Inst->getFunction();
        if (!IF)
          continue;
        if (FS.count(IF) == 0) {
          FS.insert(IF);
          Temp.insert(IF);
        }
      }
    }
    for (auto F : Temp) {
     recursivelyFindIndirectUsers(F, FS);
    }
  }

  // float AS3*, [0xfloat] AS3*,
  // [0x float], float
  static void recursivelyReplaceArrayWithPointer(Value *DestV, Value *SrcV, Type *ArrayType, Type *ElemType, Function *F, IRBuilder<> &B) {
    //std::cerr << "############ REPLACING (SrcV): \n";
    //SrcV->dump(); // AScast to [0 x float] addrspace(4)*
    //std::cerr << "############ WITH (DestV): \n";
    //DestV->dump(); // AScast to float addrspace(4)*
    //std::cerr << "######################## REPL BEGIN users: \n";
    SmallVector<Instruction *> InstsToDelete;

    for (auto U : SrcV->users()) {

      if (U->getType() == nullptr)
        continue;

      if (llvm::AddrSpaceCastInst *ASCI = dyn_cast<AddrSpaceCastInst>(U)) {
        //std::cerr << "is ASCI\n";
        B.SetInsertPoint(ASCI);
        PointerType *PT = PointerType::get(ElemType, ASCI->getDestAddressSpace());
        Value *NewASCI = B.CreateAddrSpaceCast(DestV, PT);

        recursivelyReplaceArrayWithPointer(NewASCI, ASCI, ArrayType, ElemType, F, B);

        // check users == 0, delete old ASCI
        if (ASCI->getNumUses() == 0) {
          //std::cerr << "ASCI users empty, deleting\n";
          InstsToDelete.push_back(ASCI);
        }
        continue;
      }

      if (llvm::GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
        //std::cerr << "@@@@@@@@@@@@@@@@@@@@@ \n is GEP; OLD GEP:\n";
        //GEP->dump(); // getelementptr inbounds [0 x float], [0 x float] addrspace(4)* %1, i64 0, i64 %idxprom79

        //float addrspace(4)*
        //GEP->getResultElementType()->dump();
        // 3
        //std::cerr << "num operands : " << GEP->getNumOperands() << "\n";
        // 2
        //std::cerr << "num indices : " << GEP->getNumIndices() << "\n";

        B.SetInsertPoint(GEP);
        SmallVector<Value *> Indices;
        // we skip the first Operand (pointer) and also the first Index
        for (unsigned i = 1; i < GEP->getNumIndices(); ++i) {
          //std::cerr << "OPERAND  " << 1+i << " : ";
          //GEP->getOperand(1+i)->dump();
          //std::cerr << "\n";
          Indices.push_back(GEP->getOperand(1+i));
        }
        //std::cerr << "DestV gettype getScalartype: \n";
        //DestV->getType()->getScalarType()->dump();
        //std::cerr << "DestV gettype getArrayElementType: \n";
        //DestV->getType()->getArrayElementType()->dump();
        //std::cerr << "DestV gettype getArrayElementType: \n";
        //DestV->getType()->getArrayElementType()->dump();

        Value *VV = B.CreateGEP(ElemType, DestV, Indices);
        GetElementPtrInst *NewGEP = dyn_cast<GetElementPtrInst>(VV);
        //std::cerr << "NEW GEP: \n";
        //NewGEP->dump();

        //std::cerr << "RESULT TYPES: \n";
        //GEP->getResultElementType()->dump();
        //NewGEP->getResultElementType()->dump();

        GEP->replaceAllUsesWith(NewGEP);
        if (GEP->getNumUses() == 0) {
          //std::cerr << "GEP users empty, deleting\n";
          InstsToDelete.push_back(GEP);
        }

        continue;
      }

      if (llvm::BitCastInst *BCI = dyn_cast<BitCastInst>(U)) {
        //std::cerr << "is BCI\n";
        //BCI->dump();

          B.SetInsertPoint(BCI);
          //PointerType *SrcPT = dyn_cast<PointerType>(BCI->getSrcTy());
          //PointerType *PT = PointerType::get(ElemType, SrcPT->getAddressSpace());
          Value *NewBCI = B.CreateBitCast(DestV, BCI->getDestTy());
          BCI->replaceAllUsesWith(NewBCI);

          //recursivelyReplaceArrayWithPointer(NewBCI, BCI, ArrayType, ElemType, F, B);

          // check users == 0, delete old BCI
          if (BCI->getNumUses() == 0) {
          //  std::cerr << "BCI users empty, deleting\n";
            InstsToDelete.push_back(BCI);
          }
        continue;
      }

      if (llvm::ReturnInst *RI = dyn_cast<ReturnInst>(U)) {
        continue;
      }

      U->dump();
      llvm_unreachable("Unknown user type (not GEP & not AScast)");
    }

    for (auto I : InstsToDelete) {
      I->eraseFromParent();
    }

    //std::cerr << "######################## REPL END users \n";
  }

  // get Function metadata "MDName" and append NN to it
  static void appendMD(Function *F, StringRef MDName, MDNode *NN) {
    unsigned MDKind = F->getContext().getMDKindID(MDName);
    MDNode *OldMD = F->getMetadata(MDKind);

    assert(OldMD != nullptr && OldMD->getNumOperands() > 0);

    llvm::SmallVector<llvm::Metadata *, 8> NewMDNodes;
    // copy MDnodes for original args
    for (unsigned i = 0; i < (F->arg_size() - 1); ++i) {
      Metadata *N = cast<Metadata>(OldMD->getOperand(i).get());
      assert(N != nullptr);
      NewMDNodes.push_back(N);
    }
    NewMDNodes.push_back(NN->getOperand(0).get());
    F->setMetadata(MDKind, MDNode::get(F->getContext(), NewMDNodes));
  }

  static void updateFunctionMD(Function *F, Module &M,
                               PointerType *ArgTypeWithoutAS) {
    // No need to update if the function does not have kernel metadata to begin
    // with. We update the kernel metadata because the consumer of this code may
    // get confused if the metadata is not complete (level-zero is known to
    // crash).
    if (!F->hasMetadata("kernel_arg_addr_space"))
      // Assuming that other kernel metadata kinds are absent if this one is.
      return;

    IntegerType *I32Type = IntegerType::get(M.getContext(), 32);
    MDNode *MD = MDNode::get(
        M.getContext(),
        ConstantAsMetadata::get(ConstantInt::get(I32Type, SPIR_LOCAL_AS)));
    appendMD(F, "kernel_arg_addr_space", MD);

    MD = MDNode::get(M.getContext(), MDString::get(M.getContext(), "none"));
    appendMD(F, "kernel_arg_access_qual", MD);

    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    ArgTypeWithoutAS->print(rso);
    std::string res(rso.str());

    MD = MDNode::get(M.getContext(), MDString::get(M.getContext(), res));
    appendMD(F, "kernel_arg_type", MD);
    appendMD(F, "kernel_arg_base_type", MD);

    MD = MDNode::get(M.getContext(), MDString::get(M.getContext(), ""));
    appendMD(F, "kernel_arg_type_qual", MD);
  }

  /* clones a function with an additional argument */
  static Function *cloneFunctionWithDynMemArg(Function *F, Module &M,
                                              GlobalVariable *GV) {

    SmallVector<Type *, 8> Parameters;

    // [1024 * float] AS3*
    PointerType *GVT = GV->getType();

    // AT & ELT are only for OpenCL metadata.
    // [1024 * float]
    ArrayType *AT = dyn_cast<ArrayType>(GVT->getElementType());
    // float
    Type *ElemT = AT->getElementType();
    // float addrspace(3)*
    PointerType *AS3_PTR = PointerType::get(ElemT, GV->getAddressSpace());

    for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
         i != e; ++i) {
      Parameters.push_back(i->getType());
    }
    Parameters.push_back(AS3_PTR);

    // Create the new function.
    FunctionType *FT =
        FunctionType::get(F->getReturnType(), Parameters, F->isVarArg());
    Function *NewF =
        Function::Create(FT, F->getLinkage(), F->getAddressSpace(), "", &M);
    NewF->takeName(F);
    F->setName("old_replaced_func");

    Function::arg_iterator AI = NewF->arg_begin();
    ValueToValueMapTy VV;
    for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
         i != e; ++i) {
      AI->setName(i->getName());
      VV[&*i] = &*AI;
      ++AI;
    }
    AI->setName(GV->getName() + "__hidden_dyn_local_mem");

    SmallVector<ReturnInst *, 1> RI;

#if LLVM_VERSION_MAJOR > 11
    CloneFunctionInto(NewF, F, VV, CloneFunctionChangeType::GlobalChanges, RI);
#else
    CloneFunctionInto(NewF, F, VV, true, RI);
#endif
    IRBuilder<> B(M.getContext());

    // float* (without AS, for MDNode)
    PointerType *AS0_PTR = PointerType::get(ElemT, 0);
    updateFunctionMD(NewF, M, AS0_PTR);

    // insert new function with dynamic mem = last argument
    M.getOrInsertFunction(NewF->getName(), NewF->getFunctionType(),
                          NewF->getAttributes());

    // find all calls/uses of this function...
    std::vector<CallInst *>  CallInstUses;
    for (const auto &U : F->users()) {
      //std::cerr << "Old version of Function " << NewF->getName().str() << " still has users, replacing now:\n";
      //std::cerr << "########## F user:\n";
      //U->dump();
      CallInst *CI = dyn_cast<CallInst>(U);
      if (CI) {
        CallInstUses.push_back(CI);
      } else {
        llvm_unreachable("unknown instruction - bug");
      }
    }

    // ... and replace them with calls to new function
    for (CallInst *CI : CallInstUses) {
      llvm::SmallVector<Value *, 12> Args;
      Function *CallerF = CI->getCaller();
      assert(CallerF);
      assert(CallerF->arg_size() > 0);
      for (Value *V : CI->args()) {
        Args.push_back(V);
      }
      Argument *LastArg = CallerF->getArg(CallerF->arg_size() - 1);
      Args.push_back(LastArg);
      B.SetInsertPoint(CI);
      CallInst *NewCI = B.CreateCall(FT, NewF, Args);
      CI->replaceAllUsesWith(NewCI);
      CI->eraseFromParent();
    }

    // now we can safely delete the old function
    if(F->getNumUses() != 0) llvm_unreachable("old function still has uses - bug!");
    F->eraseFromParent();

    Argument *last_arg = NewF->arg_end();
    --last_arg;

    // if the function uses dynamic shared memory (via the GVar),
    // replace all uses of GVar inside function with the new dyn mem Argument
    if (isGVarUsedInFunction(GV, NewF)) {
      //std::cerr << "@@@@@ GVAR: " << GV->getName().str() << " is USED in:  " << NewF->getName().str() << "\n";
      B.SetInsertPoint(NewF->getEntryBlock().getFirstNonPHI());

      // insert a bitcast of dyn mem argument to [N x Type] Array
      Value *BitcastV = B.CreateBitOrPointerCast(last_arg, GVT, "casted_last_arg");
      Instruction *LastArgBitcast = dyn_cast<Instruction>(BitcastV);

      // replace GVar references with the [N x Type] bitcast
      replaceGVarUsesWith(GV, NewF, BitcastV);

      // now the code should be without GVar references, but still potentially
      // contains [0 x ElemType] arrays; we need to get rid of those

      //std::cerr << "##################################################### APTER GV replace\n";
      //NewF->dump();
      //std::cerr << "#####################################################\n";

      // replace all [N x Type]* bitcast uses with direct use of ElemT*-type dyn mem argument
      recursivelyReplaceArrayWithPointer(last_arg, LastArgBitcast, AT, ElemT, NewF, B);

      //std::cerr << "##################################################### APTER LASTARG replace\n";
      //NewF->dump();
      //std::cerr << "#####################################################\n";

      // the bitcast to [N x Type] should now be unused
      if(LastArgBitcast->getNumUses() != 0) llvm_unreachable("Something still uses LastArg bitcast - bug!");
      LastArgBitcast->eraseFromParent();
    }

    return NewF;
  }

  static bool transformDynamicShMemVarsImpl(Module &M) {

    bool Modified = false;

    SmallVector<GlobalVariable *> GVars;

    /* unfortunately the M.global_begin/end iterators hide some of the
     * global variables, therefore are not usable here; must use VST */
    ValueSymbolTable &VST = M.getValueSymbolTable();
    ValueSymbolTable::iterator VSTI;

    //std::cerr << "##################################################### MODULE BEFORE\n";
    //M.dump();
    //std::cerr << "#####################################################\n";

    // find global variables that represent dynamic shared memory (__shared__)
    for (VSTI = VST.begin(); VSTI != VST.end(); ++VSTI) {

      Value *V = VSTI->getValue();
      GlobalVariable *GV = dyn_cast<GlobalVariable>(V);
      if (GV == nullptr)
        continue;

      PointerType *GVT = GV->getType();
      ArrayType *AT;

      // Dynamic shared arrays declared as "extern __shared__ int something[]"
      // are 0 sized, and this causes problems for SPIRV translator, so we need
      // to fix them by converting to pointers
      // Dynamic shared arrays declared with HIP_DYNAMIC_SHARED macro are declared as
      // "__shared__ type var[4294967295];"
      if (GV->hasName() == true && GVT->getAddressSpace() == SPIR_LOCAL_AS &&
          (AT = dyn_cast<ArrayType>(GVT->getElementType())) != nullptr &&
          (AT->getArrayNumElements() == 4294967295 || AT->getArrayNumElements() == 0)
          ) {
        GVars.push_back(GV);
      }
    }

    for (GlobalVariable *GV : GVars) {
      //std::cerr << "@@@@@@@@@@@@@ Processing GVar: " << GV->getName().str() << "\n";
      FSet DirectUserSet;
      // first, find functions that directly use the GVar. However, these may be
      // called from other functions, so we need to append the
      // dynamic shared memory argument recursively.
      recursivelyFindDirectUsers(GV, DirectUserSet);
      if (DirectUserSet.empty())
        continue;

      OrderedFSet IndirectUserSet;
      for (Function *F : DirectUserSet) {
        //std::cerr << "Direct User (Function) found: " << F->getName().str() << "\n";
        recursivelyFindIndirectUsers(F, IndirectUserSet);
      }

      // find the functions that indirectly use the GVar. These will be processed (cloned with dyn mem arg) first,
      // so that the direct users can rely on dyn mem argument being present in their caller.
      for (auto FI = IndirectUserSet.rbegin(); FI != IndirectUserSet.rend(); ++FI) {
        Function *F = *FI;
        //std::cerr << "Indirect User (Function) found: " << F->getName().str() << ", cloning\n";
        Function *NewF = cloneFunctionWithDynMemArg(F, M, GV);
        if(NewF == nullptr) llvm_unreachable("cloning failed");
      }

      // now clone the direct users and replace GVar references inside them
      for (Function *F : DirectUserSet) {
        //std::cerr << "CONVERTING FUNCTION " << F->getName().str() << "\n";

        Function *NewF = cloneFunctionWithDynMemArg(F, M, GV);
        if(NewF == nullptr) llvm_unreachable("cloning failed");
        Modified = true;
      }

      //std::cerr << "##################################################### MODULE AFTER CONV:\n";
      //M.dump();
      //std::cerr << "#####################################################\n";

      // it seems that the
      bool Deleted = true;
      while (Deleted && GV->getNumUses()) {
        Deleted = false;
        User *U = *GV->user_begin();
        //std::cerr << "@@@@@ USER: \n";
        //U->dump();
        if (Instruction *I = dyn_cast<Instruction>(U)) {
          //std::cerr << "Instruction, Used in function: " << I->getFunction()->getName().str() << "\n";
          if (I->getParent()) {
            I->eraseFromParent();
            Deleted = true;
          }
        } else
        if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U)) {
          //std::cerr << "is ConstExpr, Uses: " << U->getNumUses() << "\n";
          if (U->getNumUses() <= 1) {
            CE->destroyConstant();
            Deleted = true;
          }
        } else
        llvm_unreachable("unknown User of Global Variable - bug!");
      }

      if (GV->getNumUses() != 0)
        llvm_unreachable("Some uses still remain - bug!");
      GV->eraseFromParent();
    }

    return Modified;
  }

public:
  static char ID;
  HipDynMemExternReplacePass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    return transformDynamicShMemVarsImpl(M);
  }

  StringRef getPassName() const override {
    return "convert HIP dynamic shared memory to OpenCL kernel argument";
  }

  static bool transformDynamicShMemVars(Module &M) {
    return transformDynamicShMemVarsImpl(M);
  }
};

// Identifier variable for the pass
char HipDynMemExternReplacePass::ID = 0;
static RegisterPass<HipDynMemExternReplacePass>
    X("hip-dyn-mem",
      "convert HIP dynamic shared memory to OpenCL kernel argument");


// Pass hook for the new pass manager.
#if LLVM_VERSION_MAJOR > 11
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

PreservedAnalyses
HipDynMemExternReplaceNewPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (HipDynMemExternReplacePass::transformDynamicShMemVars(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-dyn-mem", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-dyn-mem") {
                    FPM.addPass(HipDynMemExternReplaceNewPass());
                    return true;
                  }
                  return false;
                });
          }};
}

#endif // LLVM_VERSION_MAJOR > 11
