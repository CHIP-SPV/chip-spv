// LLVM Pass to wrap kernels that take texture objects into new kernels that take
// OpenCL texture and sampler arguments

#ifndef LLVM_PASSES_HIP_TEXTURE
#define LLVM_PASSES_HIP_TEXTURE

#include "llvm/ADT/SmallPtrSet.h"
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

#include "llvm/IR/TypeFinder.h"

#include <iostream>

#include <vector>
#include <set>
#include <string>

using namespace llvm;
using namespace std;

#define SPIR_GLOBAL_AS 1
#define SPIR_LOCAL_AS  3
#define GENERIC_AS     4

#define DEFAULT_AS     0
#define OCL_IMAGE_AS   1
#define OCL_SAMPLER_AS 2

// Identify and maintain the OpenCL image, sample and Hip Texture related types
class TextureTypes {
protected:
  // HIP texture type
  Type* hipTextureTy;

  // OCL image type
  Type* oclImageTy;

  // OCL sampler type
  Type* oclSamplerTy;
  
public:
  TextureTypes(Module &M) {
    // Check through struct types
    TypeFinder OCLTypes;
    OCLTypes.run(M, true);

    StringRef imageTyName("opencl.image2d_ro_t");
    StringRef samplerTyName("opencl.sampler_t");
    StringRef textureTyName("struct.hipTextureObject_s");

    StringRef imageTyName_("struct.OCLImage");
    StringRef samplerTyName_("struct.OCLSampler");
    StringRef textureTyName_("struct.hipTextureObject_st");
    
    for (TypeFinder::iterator ti = OCLTypes.begin(), te = OCLTypes.end(); ti != te; ti ++) {
      StringRef structName = (* ti)->getStructName();
      if (structName.equals(imageTyName)) {
	this->oclImageTy = * ti;
      } else if (structName.equals(samplerTyName)) {
	this->oclSamplerTy = * ti;
      } else if (structName.equals(textureTyName)) {
	this->hipTextureTy = * ti;
      }
    }
  }

  Type* GetOCLImageType() {
    return this->oclImageTy;
  }

  // Retrieve the image pointer type with a specified address space
  Type* GetOCLImagePtrType(int as) {
    return PointerType::get(this->oclImageTy, as);
  }
  
  Type* GetOCLSamplerType() {
    return this->oclSamplerTy;
  }

  // Retrieve the sampler pointer type with a specified address space  
  Type* GetOCLSamplerPtrType(int as) {
    return PointerType::get(this->oclSamplerTy, as); 
  }
  
  Type* GetHipTextureType() {
    return this->hipTextureTy;
  }

  // Retrieve the texture pointer type with a specified address space  
  Type* GetHipTexturePtrType(int as) {
    return PointerType::get(this->hipTextureTy, as);
  }
  
  bool IsOCLImageTy(Type* Ty) {
    return Ty == this->oclImageTy;
  }

  bool IsOCLSamplerTy(Type* Ty) {
    return Ty == this->oclSamplerTy;
  }

  bool IsHipTextureTy(Type* Ty) {
    return Ty == this->hipTextureTy;
  }

  bool IsHipTexturePtrTy(Type* Ty) {
    if (!Ty->isPointerTy())
      return false;

    PointerType* ptrTy = dyn_cast<PointerType>(Ty);
    return IsHipTextureTy(ptrTy->getElementType());
  }
};

// Generate the OpenCL type based wrapper function for hipTexture type based kernel function
class OCLWrapperFunctions {
protected:
  // The map between the original  kernel function and its wrapper
  map<Function* , Function* > orig2Wrapper;
  
  // The map between the wrapper kernel function and its original version
  map<Function* , Function* > wrapper2Orig;
  
  // The LLVM module
  Module& M;
  
  // The type manager for OpenCL image, sampler and Hip Texture related types
  TextureTypes& textureTypes;

  // The SPIR functions, i.e. with SPIR_FUNC calling convention
  set<Function* > spirFuncs;
  
public:
  OCLWrapperFunctions(Module& M_, TextureTypes& textureTypes_) : M(M_), textureTypes(textureTypes_) {};
  
  // The candidate kernel functions that need to be wrapped,
  // 1. the spir kernel
  // 2. the function arguments contain hipTexture type
   bool collectCandidates() {
    for (auto fi = M.begin(), fe = M.end(); fi != fe; fi ++) {
      StringRef funcName = fi->getName();
      Function* F = M.getFunction(funcName);
      
      if (F->getCallingConv() == CallingConv::SPIR_KERNEL) {
        // Check through function arguments                                                          
        bool HasTextureType = false;
        for (Function::arg_iterator ai = F->arg_begin(), ae = F->arg_end(); ai != ae; ai ++) {
          if (textureTypes.IsHipTexturePtrTy(ai->getType())) {
            HasTextureType = true;
            	    
            break;
          }                                                                                     
        }

        if (HasTextureType) {
          orig2Wrapper[F] = nullptr;
	}
      } else if(F->getCallingConv() == CallingConv::SPIR_FUNC) {
	// Check through function arguments
	bool HasTextureType = false;
        for (Function::arg_iterator ai = F->arg_begin(), ae = F->arg_end(); ai != ae; ai ++) {
          if (textureTypes.IsHipTexturePtrTy(ai->getType())) {
            HasTextureType = true;

	    break;
	  }
	}

	if (HasTextureType) {
	  spirFuncs.insert(F);
	}
      }
    }

    return orig2Wrapper.size() > 0;
  }

  // Generate the wrapper functions for the candidate kernels
  bool wrapKernels() {
    // Reset the attributes of spir functinos
    for (Function* spirF : spirFuncs) {
      // Reset the original function's attribute by eliminating the prohibition of optimizations 
      assignAttributes(spirF, spirF);
    }
    
    // Create all wrapper functions
    for (map<Function* , Function* >::iterator wi = orig2Wrapper.begin(), we = orig2Wrapper.end();
	 wi != we; ++ wi) {
      Function* F = wi->first;

      // The map of argument ID from wrapper function to original function
      SmallVector<int, 16> argMap;
      // Create the wrapper function
      Function* wrapperF = createWrapperFunction(F, argMap);
      
      // Fix the original function regarding its texture argument's address space
      F = fixTextureArgsAddrSpace(F, GENERIC_AS, SPIR_GLOBAL_AS);

      // Fill in the content 
      fillContent(wrapperF, F, argMap);
      
      // Register wrapper
      wrapper2Orig[wrapperF] = F;

      // Dump wrapper function
      // wrapperF->dump();
    }
    
    return false;
  }

protected:
  // Create the wrapper function
  Function* createWrapperFunction(Function* F, SmallVector<int, 16>& argMap) {
    // Collect arugment types                                                                       
    SmallVector<Type *, 16> Parameters;
    // The index mapping of the arguments from wrappe function to original function                   
    int origIdx = 0;
    int wrapperIdx = 0;
    for (Function::const_arg_iterator ai = F->arg_begin(), ae = F->arg_end(); ai != ae; ++ ai) {
      if (textureTypes.IsHipTexturePtrTy(ai->getType())) {
        // Here we replace HipTexture type with OpenCL image and sampler types                        
        Parameters.push_back(textureTypes.GetOCLImagePtrType(OCL_IMAGE_AS)); 
        Parameters.push_back(textureTypes.GetOCLSamplerPtrType(OCL_SAMPLER_AS)); 

        argMap.push_back(origIdx);
        argMap.push_back(origIdx);
      } else {
        Parameters.push_back(ai->getType());

        argMap.push_back(-1);
      }

      // Increment original function argument's ID 
      origIdx ++;
    }

    // Create the wrapper function
    FunctionType * FuncTy = FunctionType::get(F->getReturnType(), Parameters, F->isVarArg());
    Function * wrapperF = Function::Create(FuncTy, F->getLinkage(), F->getAddressSpace(), "", &M);
    string wrapperName = "new_wrapper";
    string origName = F->getName().str();
    F->setName(origName + "_impl");
  
    wrapperF->setName(origName);
    
    // Inherit the attributes from original function
    assignAttributes(wrapperF, F);
  
    // Set visibility
    wrapperF->setVisibility(GlobalValue::VisibilityTypes::HiddenVisibility);
    
    // Set wrapper function's calling convention
    wrapperF->setCallingConv(CallingConv::SPIR_KERNEL);

    // Add wrapper function into module
    M.getOrInsertFunction(wrapperF->getName(), wrapperF->getFunctionType(), wrapperF->getAttributes());
    
    // Switch the oirignal function to implemmentation function by reset the name and calling convention
    // Reset the calling convention to SPIR_FUNC
    F->setCallingConv(CallingConv::SPIR_FUNC);

    // Reset the original function's attribute by eliminating the prohibition of optimizations
    assignAttributes(F, F);
    
    return wrapperF;
  }

  // Get function argument                                            
  Argument* GetFunctionArg(Function* F, int idx) {
    Function::arg_iterator iter = F->arg_begin();
    return iter += idx;
  }

  // Fill in the content                                            
  void fillContent(Function* wrapperF, Function* origF, SmallVector<int, 16>& argMap) {
    IRBuilder<> B(M.getContext());

    // Create the basic block                                                       
    BasicBlock* contentBB = BasicBlock::Create(wrapperF->getContext(), "wrap.texture.struct",
                                               wrapperF);

    // Fill in the texture struct creation                                            
    // Collect the argument to call implementationn function                                   
    SmallVector<Value *, 16> Args;
    Instruction* prevTextureAlloc = nullptr;
    for (int i = 0; i < argMap.size(); i ++) {
      if (argMap[i] >= 0) {
        Argument* imageArg = GetFunctionArg(wrapperF, i ++); // argMap[i ++]);
        Argument* samplerArg = GetFunctionArg(wrapperF, i); // argMap[i]);

        // Alloc texture struct                                             
        Instruction* textureAlloc = allocTextureStruct(wrapperF, contentBB, prevTextureAlloc, i);

        // Initialize texture struct with image and sampler                             
        initTextureStruct(wrapperF, contentBB, textureAlloc, imageArg, samplerArg);

	// Cast texture object's address space
	Instruction* textureASCast = castAddrSpace(textureAlloc, GENERIC_AS, SPIR_GLOBAL_AS, // DEFAULT_AS,
						   contentBB);
	
        // Carry the HipTexture_t argument                                   
        // Args.push_back(textureAlloc);
	Args.push_back(textureASCast);
	
        // Set current texture allocation instruction as previou texture allocation instruction    
        prevTextureAlloc = textureAlloc;
      } else {
        Argument* arg = GetFunctionArg(wrapperF, i);
        // Carry normal argument                                                
        Args.push_back(arg);
      }
    }

    // Create the call site for the original/implementation function
    callImplementation(wrapperF, origF, contentBB, Args);
    // ReturnInst* retInst = ReturnInst::Create(wrapperF->getContext(), nullptr, contentBB);
    // callDummy(wrapperF, origF, contentBB, Args);
  }
  
  // Alloc texture struct
  Instruction* allocTextureStruct(Function* wrapperF, BasicBlock* contentBB,
                                  Instruction* prevTextureAlloc, int idx) {
    Instruction* textureAlloc = nullptr;
    // Create the alloc instruction                                     
    textureAlloc = new AllocaInst(textureTypes.GetHipTextureType(), // GENERIC_AS, 
				  // DEFAULT_AS,
				  SPIR_GLOBAL_AS,
				  "create texture object", contentBB);

    return textureAlloc;
  }

  // Initialize texture struct with image and sampler
  bool initTextureStruct(Function* wrapperF, BasicBlock* contentBB, Instruction* textureAlloc,
			 Argument* imageArg, Argument* samplerArg) {
    // Crete the type cast for image
    Instruction* castImageInst = new PtrToIntInst(imageArg,
						  Type::getInt64Ty(M.getContext()),
						  "cast_image_to_int64ptr",
						  contentBB);
    
    // Create the stroe related instruction                                     
    Value* imageIdxList[2] = {ConstantInt::get(Type::getInt32Ty(M.getContext()), 0),
                              ConstantInt::get(Type::getInt32Ty(M.getContext()), 0)};
    Instruction* getImagePtrInst = GetElementPtrInst::Create(textureTypes.GetHipTextureType(),
                                                             textureAlloc,
                                                             ArrayRef<Value*>(imageIdxList, 2),
                                                             "image_field_address",
                                                             contentBB);
    
    Instruction* storeImageInst = new StoreInst(castImageInst, getImagePtrInst, contentBB);

    // Create the type cast for sampler
     Instruction* castSamplerInst = new PtrToIntInst(samplerArg,
						    Type::getInt64Ty(M.getContext()),
						    "cast_sampler_to_int64ptr",
						    contentBB);
    
    // Create the store related instruction
    Value* samplerIdxList[2] = {ConstantInt::get(Type::getInt32Ty(M.getContext()), 0),
                                ConstantInt::get(Type::getInt32Ty(M.getContext()), 1)};
    Instruction* getSamplerPtrInst = GetElementPtrInst::Create(textureTypes.GetHipTextureType(),
                  		                               textureAlloc,
                                                               ArrayRef<Value*>(samplerIdxList, 2),
                                                               "sampler_field_address",
                                                               contentBB); 
    
    Instruction* storeSamplerInst = new StoreInst(castSamplerInst, getSamplerPtrInst, contentBB);

    return true;
  }

  // Create the call site for the original/implementation function
  Instruction* callImplementation(Function* wrapperF, Function* origF, BasicBlock* contentBB,
				  SmallVector<Value *, 16>& Args) {
    Value* ArgVals[16];
    int count = 0;
    for (Value* Arg : Args) {
      ArgVals[count ++] = Arg;
    }
 
    
    // Create the call site for the original function (i.e. the implementtion function) and append it to
    // the end of basic block
    CallInst* callInst = CallInst::Create(origF, ArrayRef<Value* >(ArgVals, count), "", contentBB);
    // Set the calling convention flag
    callInst->setCallingConv(CallingConv::SPIR_FUNC);
    
    // Create the return instruction
    ReturnInst* retInst = ReturnInst::Create(wrapperF->getContext(), nullptr, contentBB);

    return retInst;
  }

  // Assign attributes, the attributes are from the original funciton, and the attributes related to
  // prohibite code optimization are removed
  bool assignAttributes(Function* destF, Function* srcF) {
    AttributeList attrs = srcF->getAttributes();
    attrs = attrs.removeAttribute(M.getContext(), AttributeList::FunctionIndex, Attribute::OptimizeNone);
    attrs = attrs.removeAttribute(M.getContext(), AttributeList::FunctionIndex, Attribute::NoInline);
    attrs = attrs.removeAttribute(M.getContext(), AttributeList::FunctionIndex, "frame-pointer");

    // Inherit the attributes from original function   
    destF->setAttributes(attrs);

    return true;
  }

  // Cast the address space 
  Instruction* castAddrSpace(Value* ptrVal, unsigned newAS, unsigned origAS, BasicBlock* contentBB) {
    PointerType* ptrTy = dyn_cast<PointerType>(ptrVal->getType());
    if (!ptrTy)
      return nullptr;
    
    Type* elemTy = ptrTy->getElementType();
    if (origAS != ptrTy->getAddressSpace())
      return nullptr;
    
    Type* newTy = PointerType::get(elemTy, newAS);
    
    return new AddrSpaceCastInst(ptrVal, newTy, "newASVal", contentBB);
  }

  // Fix the original function regarding its texture argument's address space
  Function* fixTextureArgsAddrSpace(Function* origF, unsigned newAS, unsigned origAS) {
    if (newAS != GENERIC_AS || origAS != SPIR_GLOBAL_AS)
      return origF;

    SmallVector<Type *, 16> ParamTys;
    bool hasMutation = false;
    for (Function::const_arg_iterator ai = origF->arg_begin(), ae = origF->arg_end(); ai != ae; ++ ai) {
      Argument * arg = (Argument *)ai;

      if (arg->getType() == textureTypes.GetHipTexturePtrType(origAS)) {
	// Mutate argument type
	arg->mutateType(textureTypes.GetHipTexturePtrType(newAS));

	// Collect uses 
	SmallVector<Instruction *, 16> ascasts;
	for (auto au : arg->users()) {
          if (const AddrSpaceCastInst* ascast = dyn_cast<AddrSpaceCastInst>(au)) {
	    AddrSpaceCastInst* ascastInst = (AddrSpaceCastInst *)ascast;
       
	    // Record addresss cast instruction
	    ascasts.push_back(ascastInst);
	    hasMutation = true;
	  }
	}

	// Replace uses
	for (Instruction* ascast : ascasts) {
	  ascast->replaceAllUsesWith(arg);
	  // Erase address cast instruction
	  ascast->eraseFromParent();
	}

	// Record new type
	ParamTys.push_back(textureTypes.GetHipTexturePtrType(newAS));
      } else {
	// Record function argument types
	ParamTys.push_back(arg->getType());
      }
    }

    // Create the new function
    if (hasMutation) {
      FunctionType * FuncTy = FunctionType::get(origF->getReturnType(), ParamTys, origF->isVarArg());
      Function *NewF = Function::Create(FuncTy, origF->getLinkage(), origF->getAddressSpace(), "", &M);
      NewF->takeName(origF);

      Function::arg_iterator AI = NewF->arg_begin();
      ValueToValueMapTy VV;
      for (Function::const_arg_iterator i = origF->arg_begin(), e = origF->arg_end(); i != e; ++i) {
	AI->setName(i->getName());
	VV[&*i] = &*AI;
	++ AI;
      }

      // Clone the function content
      SmallVector<ReturnInst *, 1> RI;
      CloneFunctionInto(NewF, origF, VV, CloneFunctionChangeType::GlobalChanges, RI);

      // Reset the function attribute
      assignAttributes(NewF, origF);
      
      // Add new function into module
      M.getOrInsertFunction(NewF->getName(), NewF->getFunctionType(), NewF->getAttributes());
      // Erase original function
      origF->eraseFromParent();

      return NewF;
    }
    
    return origF;
  }
  
  // Create a dummy function that is only for test
  Instruction* callDummy(Function* wrapperF, Function* origF, BasicBlock* contentBB,
			 SmallVector<Value *, 16>& Args) {
    Value* ArgVals[16];
    for (Value* Arg : Args) {
      if (Arg->getType() == textureTypes.GetHipTexturePtrType(GENERIC_AS)) // SPIR_GLOBAL_AS))
       	ArgVals[0] = Arg;
    }

    // Create the dummy function
    SmallVector<Type *, 16> Parameters;
    Parameters.push_back(textureTypes.GetHipTexturePtrType(GENERIC_AS)); // SPIR_GLOBAL_AS));
    
    FunctionType * FuncTy = FunctionType::get(origF->getReturnType(), Parameters, origF->isVarArg());
    Function * dummyF = Function::Create(FuncTy, GlobalValue::LinkageTypes::InternalLinkage,
					 origF->getAddressSpace(), "", &M);
    // Set name
    dummyF->setName("dummy_call_texture");
    // Set calling convention
    dummyF->setCallingConv(CallingConv::SPIR_FUNC);
    
    // Inherit the attributes from original function 
    assignAttributes(dummyF, origF);
    
    // Create the basic block  
    BasicBlock* dummyBB = BasicBlock::Create(dummyF->getContext(), "wrap.texture.struct", dummyF);

    // Fill in content
    // Return void
    Instruction* retInst = ReturnInst::Create(dummyF->getContext(), nullptr, dummyBB);

    // Register dummy function
    M.getOrInsertFunction(dummyF->getName(), dummyF->getFunctionType(), dummyF->getAttributes());
    
    // Call the dummy function
    CallInst* callInst = CallInst::Create(dummyF, ArrayRef<Value* >(ArgVals, 1), "", contentBB);
    callInst->setCallingConv(CallingConv::SPIR_FUNC);
    
    // Return void
    retInst = ReturnInst::Create(wrapperF->getContext(), nullptr, contentBB);
    
    return retInst;
  }

public:

  static bool runTexture(Module& M) {
     // Collect relevant types
    TextureTypes textureTypes(M);

    // Create the wrapper function generator
    OCLWrapperFunctions oclWrapper(M, textureTypes);
    // Collect relevant kernel functions
    if (oclWrapper.collectCandidates()) {
      // If there is candidate kernels for wrapping, then generate wrapper functions
      oclWrapper.wrapKernels();
    }
    
    // M.dump();

    return false;
  }
};

#if LLVM_VERSION_MAJOR > 11
class HipTextureExternReplaceNewPass
    : public PassInfoMixin<HipTextureExternReplaceNewPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};
#endif

#endif
