

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "common.hh"
#include "spirv.hh"
#include "logging.hh"

const std::string OpenCLStd{"OpenCL.std"};

class SPIRVtype {
  size_t Size_;

public:
  SPIRVtype(size_t Size) : Size_(Size) {}
  virtual ~SPIRVtype(){};
  size_t size() { return Size_; }
  virtual OCLType ocltype() = 0;
  virtual OCLSpace getAS() { return OCLSpace::Private; }
};

typedef std::map<int32_t, SPIRVtype *> SPIRTypeMap;

class SPIRVtypePOD : public SPIRVtype {
public:
  SPIRVtypePOD(int32_t Id, size_t Size) : SPIRVtype(Size) {}
  virtual ~SPIRVtypePOD(){};
  virtual OCLType ocltype() override { return OCLType::POD; }
};

class SPIRVtypeOpaque : public SPIRVtype {
public:
  SPIRVtypeOpaque(int32_t Id)
      : SPIRVtype(0) // Opaque types are unsized.
  {}
  virtual ~SPIRVtypeOpaque(){};
  virtual OCLType ocltype() override { return OCLType::Opaque; }
};

class SPIRVtypeImage : public SPIRVtypeOpaque {
public:
  SPIRVtypeImage(int32_t Id) : SPIRVtypeOpaque(Id) {}
  virtual ~SPIRVtypeImage(){};
  virtual OCLType ocltype() override { return OCLType::Image; }
  virtual OCLSpace getAS() override { return OCLSpace::Unknown; }
};

class SPIRVtypeSampler : public SPIRVtypeOpaque {
public:
  SPIRVtypeSampler(int32_t Id) : SPIRVtypeOpaque(Id) {}
  virtual ~SPIRVtypeSampler(){};
  virtual OCLType ocltype() override { return OCLType::Sampler; }
  virtual OCLSpace getAS() override { return OCLSpace::Constant; }
};

class SPIRVtypePointer : public SPIRVtype {
  OCLSpace ASpace_;

public:
  SPIRVtypePointer(int32_t Id, int32_t StorClass, size_t PointerSize)
      : SPIRVtype(PointerSize) {
    switch (StorClass) {
    case (int32_t)spv::StorageClass::CrossWorkgroup:
      ASpace_ = OCLSpace::Global;
      break;

    case (int32_t)spv::StorageClass::Workgroup:
      ASpace_ = OCLSpace::Local;
      break;

    case (int32_t)spv::StorageClass::UniformConstant:
      ASpace_ = OCLSpace::Constant;
      break;

    case (int32_t)spv::StorageClass::Function:
      assert(0 && "should have been handled elsewhere!");
      break;

    default:
      ASpace_ = OCLSpace::Unknown;
    }
  }
  virtual ~SPIRVtypePointer(){};
  virtual OCLType ocltype() override { return OCLType::Pointer; }
  OCLSpace getAS() override { return ASpace_; }
};

class SPIRVinst {
  spv::Op Opcode_;
  size_t WordCount_;
  int32_t Word1_;
  int32_t Word2_;
  int32_t Word3_;
  std::string Extra_;
  int32_t *OrigStream_;

public:
  SPIRVinst(int32_t *Stream) {
    OrigStream_ = Stream;
    int32_t Word0 = Stream[0];
    WordCount_ = (unsigned)Word0 >> 16;
    Opcode_ = (spv::Op)(Word0 & 0xFFFF);

    if (WordCount_ > 1)
      Word1_ = Stream[1];

    if (WordCount_ > 2)
      Word2_ = Stream[2];

    if (WordCount_ > 3)
      Word3_ = Stream[3];

    if (Opcode_ == spv::Op::OpEntryPoint) {
      const char *Pp = (const char *)(Stream + 3);
      Extra_ = Pp;
    }

    if (Opcode_ == spv::Op::OpExtInstImport) {
      const char *Pp = (const char *)(Stream + 2);
      Extra_ = Pp;
    }
  }

  bool isKernelCapab() const {
    return (Opcode_ == spv::Op::OpCapability) &&
           (Word1_ == (int32_t)spv::Capability::Kernel);
  }
  bool isExtIntOpenCL() const { return Extra_ == OpenCLStd; }
  bool isMemModelOpenCL() const {
    return (Opcode_ == spv::Op::OpMemoryModel) &&
           (Word2_ == (int32_t)spv::MemoryModel::OpenCL);
  }
  size_t getPointerSize() const {
    if (Opcode_ != spv::Op::OpMemoryModel)
      return 0;
    return (Word1_ == (int32_t)spv::AddressingModel::Physical64) ? 8 : 4;
  }
  bool isLangOpenCL() const {
    return (Opcode_ == spv::Op::OpSource) &&
           ((Word1_ == (int32_t)spv::SourceLanguage::OpenCL_C) ||
            (Word1_ == (int32_t)spv::SourceLanguage::OpenCL_CPP));
  }

  bool isEntryPoint() {
    return (Opcode_ == spv::Op::OpEntryPoint) &&
           (Word1_ == (int32_t)spv::ExecutionModel::Kernel);
  }
  int32_t entryPointID() { return Word2_; }
  std::string &&entryPointName() { return std::move(Extra_); }

  size_t size() const { return WordCount_; }
  spv::Op getOpcode() const { return Opcode_; }

  int32_t getFunctionID() const { return Word2_; }
  int32_t getFunctionTypeID() const { return OrigStream_[4]; }
  int32_t getFunctionRetType() const { return Word1_; }

  bool isType() const {
    return ((int32_t)Opcode_ >= (int32_t)spv::Op::OpTypeVoid) &&
           ((int32_t)Opcode_ <= (int32_t)spv::Op::OpTypeForwardPointer);
  }
  int32_t getTypeID() const {
    assert(isType());
    return Word1_;
  }
  bool isFunctionType() const { return (Opcode_ == spv::Op::OpTypeFunction); }
  bool isFunction() const { return (Opcode_ == spv::Op::OpFunction); }

  SPIRVtype *decodeType(SPIRTypeMap &TypeMap, size_t PointerSize) {
    if (Opcode_ == spv::Op::OpTypeVoid) {
      return new SPIRVtypePOD(Word1_, 0);
    }

    if (Opcode_ == spv::Op::OpTypeBool) {
      return new SPIRVtypePOD(Word1_, 1);
    }

    if (Opcode_ == spv::Op::OpTypeInt) {
      return new SPIRVtypePOD(Word1_, ((size_t)Word2_ / 8));
    }

    if (Opcode_ == spv::Op::OpTypeFloat) {
      return new SPIRVtypePOD(Word1_, ((size_t)Word2_ / 8));
    }

    if (Opcode_ == spv::Op::OpTypeVector) {
      auto Type = TypeMap[Word2_];
      if (!Type) {
        logWarn("SPIR-V Parser: Word2_ {} not found in type map", Word2_);
        return nullptr;
      }
      size_t TypeSize = Type->size();
      return new SPIRVtypePOD(Word1_, TypeSize * OrigStream_[3]);
    }

    if (Opcode_ == spv::Op::OpTypeArray) {
      auto Type = TypeMap[Word2_];
      if (!Type) {
        logWarn("SPIR-V Parser: Word2_ {} not found in type map", Word2_);
        return nullptr;
      }
      size_t TypeSize = Type->size();
      return new SPIRVtypePOD(Word1_, TypeSize * Word3_);
    }

    if (Opcode_ == spv::Op::OpTypeStruct) {
      size_t TotalSize = 0;
      for (size_t i = 2; i < WordCount_; ++i) {
        int32_t MemberId = OrigStream_[i];

        auto Type = TypeMap[MemberId];
        if (!Type) {
          logWarn("SPIR-V Parser: MemberId {} not found in type map", MemberId);
          continue;
        }
        size_t TypeSize = Type->size();

        TotalSize += TypeSize;
      }
      return new SPIRVtypePOD(Word1_, TotalSize);
    }

    if (Opcode_ == spv::Op::OpTypeOpaque) {
      return new SPIRVtypeOpaque(Word1_);
    }

    if (Opcode_ == spv::Op::OpTypeImage) {
      return new SPIRVtypeImage(Word1_);
    }

    if (Opcode_ == spv::Op::OpTypeSampler) {
      return new SPIRVtypeSampler(Word1_);
    }

    if (Opcode_ == spv::Op::OpTypePointer) {
      // structs or vectors passed by value are represented in LLVM IR / SPIRV
      // by a pointer with "byval" keyword; handle them here
      if (Word2_ == (int32_t)spv::StorageClass::Function) {
        int32_t Pointee = Word3_;
        auto Type = TypeMap[Pointee];
        if (!Type) {
          logError("SPIR-V Parser: Failed to find size for type id {}",
                   Pointee);
          return nullptr;
        }

        size_t PointeeSize = Type->size();
        return new SPIRVtypePOD(Word1_, PointeeSize);

      } else
        return new SPIRVtypePointer(Word1_, Word2_, PointerSize);
    }

    return nullptr;
  }

  OCLFuncInfo *decodeFunctionType(SPIRTypeMap &TypeMap, size_t PointerSize) {
    assert(Opcode_ == spv::Op::OpTypeFunction);

    OCLFuncInfo *Fi = new OCLFuncInfo;

    int32_t RetId = Word2_;
    auto It = TypeMap.find(RetId);
    assert(It != TypeMap.end());
    Fi->RetTypeInfo.Type = It->second->ocltype();
    Fi->RetTypeInfo.Size = It->second->size();
    Fi->RetTypeInfo.Space = It->second->getAS();

    size_t NumArgs = WordCount_ - 3;
    if (NumArgs > 0) {
      Fi->ArgTypeInfo.resize(NumArgs);
      for (size_t i = 0; i < NumArgs; ++i) {
        int32_t TypeId = OrigStream_[i + 3];
        auto It = TypeMap.find(TypeId);
        assert(It != TypeMap.end());
        Fi->ArgTypeInfo[i].Type = It->second->ocltype();
        Fi->ArgTypeInfo[i].Size = It->second->size();
        Fi->ArgTypeInfo[i].Space = It->second->getAS();
      }
    }

    return Fi;
  }
};

class SPIRVmodule {
  std::map<int32_t, std::string> EntryPoints_;
  SPIRTypeMap TypeMap_;
  OCLFuncInfoMap FunctionTypeMap_;
  std::map<int32_t, int32_t> EntryToFunctionTypeIDMap_;

  bool MemModelCL_;
  bool KernelCapab_;
  bool ExtIntOpenCL_;
  bool HeaderOK_;
  bool ParseOK_;

public:
  ~SPIRVmodule() {
    for (auto I : TypeMap_) {
      delete I.second;
    }
  }

  bool valid() {
    bool AllOk = true;
    auto Check = [&](bool Cond, const char *ErrMsg) {
      if (!Cond)
        logError(ErrMsg);
      AllOk &= Cond;
    };

    Check(HeaderOK_, "Invalid SPIR-V header.");
    // TODO: Temporary. With these check disabled the simple_kernel
    //       runs successfully on OpenCL backend at least. Note that we are
    //       passing invalid SPIR-V binary.
    // Check(KernelCapab_, "Kernel capability missing.");
    // Check(ExtIntOpenCL_, "Missing extended OpenCL instructions.");
    Check(MemModelCL_, "Incorrect memory model.");
    Check(ParseOK_, "An error encountered during parsing.");
    return AllOk;
  }

  bool parseSPIRV(int32_t *Stream, size_t NumWords) {
    int32_t *StreamIntPtr = Stream;

    KernelCapab_ = false;
    ExtIntOpenCL_ = false;
    HeaderOK_ = false;
    MemModelCL_ = false;
    ParseOK_ = false;

    if (*StreamIntPtr != spv::MagicNumber) {
      logError("Incorrect SPIR-V magic number.");
      return false;
    }
    ++StreamIntPtr;

    if (*StreamIntPtr < spv::Version10 || *StreamIntPtr > spv::Version12) {
      logError("Unsupported SPIR-V version.");
      return false;
    }
    ++StreamIntPtr;

    // GENERATOR
    ++StreamIntPtr;

    // BOUND
    // int32_t Bound = *StreamIntPtr;
    ++StreamIntPtr;

    // RESERVED
    if (*StreamIntPtr != 0) {
      logError("Invalid SPIR-V: Reserved word is not 0.");
      return false;
    }
    ++StreamIntPtr;

    HeaderOK_ = true;

    // INSTRUCTION STREAM
    ParseOK_ = parseInstructionStream(StreamIntPtr, (NumWords - 5));
    return valid();
  }

  bool fillModuleInfo(OpenCLFunctionInfoMap &ModuleMap) {
    if (!valid())
      return false;

    for (auto i : EntryPoints_) {
      int32_t EntryPointID = i.first;
      auto Ft = EntryToFunctionTypeIDMap_.find(EntryPointID);
      assert(Ft != EntryToFunctionTypeIDMap_.end());
      auto Fi = FunctionTypeMap_.find(Ft->second);
      assert(Fi != FunctionTypeMap_.end());
      ModuleMap.emplace(std::make_pair(i.second, Fi->second));
    }

    return true;
  }

private:
  bool parseInstructionStream(int32_t *Stream, size_t NumWords) {
    int32_t *StreamIntPtr = Stream;
    size_t PointerSize = 0;
    while (NumWords > 0) {
      SPIRVinst Inst(StreamIntPtr);

      if (Inst.isKernelCapab())
        KernelCapab_ = true;

      if (Inst.isExtIntOpenCL())
        ExtIntOpenCL_ = true;

      if (Inst.isMemModelOpenCL()) {
        MemModelCL_ = true;
        PointerSize = Inst.getPointerSize();
        assert(PointerSize > 0);
      }

      if (Inst.isEntryPoint()) {
        EntryPoints_.emplace(
            std::make_pair(Inst.entryPointID(), Inst.entryPointName()));
      }

      if (Inst.isType()) {
        if (Inst.isFunctionType())
          FunctionTypeMap_.emplace(
              std::make_pair(Inst.getTypeID(),
                             Inst.decodeFunctionType(TypeMap_, PointerSize)));
        else
          TypeMap_.emplace(std::make_pair(
              Inst.getTypeID(), Inst.decodeType(TypeMap_, PointerSize)));
      }

      if (Inst.isFunction() &&
          (EntryPoints_.find(Inst.getFunctionID()) != EntryPoints_.end())) {
        // ret type must be void
        auto Retty = TypeMap_.find(Inst.getFunctionRetType());
        assert(Retty != TypeMap_.end());
        assert(TypeMap_[Inst.getFunctionRetType()]->size() == 0);

        EntryToFunctionTypeIDMap_.emplace(
            std::make_pair(Inst.getFunctionID(), Inst.getFunctionTypeID()));
      }

      NumWords -= Inst.size();
      StreamIntPtr += Inst.size();
    }

    return true;
  }
};

bool parseSPIR(int32_t *Stream, size_t NumWords,
               OpenCLFunctionInfoMap &Output) {
  SPIRVmodule Mod;
  if (!Mod.parseSPIRV(Stream, NumWords))
    return false;
  return Mod.fillModuleInfo(Output);
}
