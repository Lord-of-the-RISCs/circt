add_circt_conversion_library(CIRCTArcToLLVM
  LowerArcToLLVM.cpp
  SHARED
  

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTComb
  CIRCTSeq
  CIRCTCombToLLVM
  CIRCTHWToLLVM
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRSCFToControlFlow
  MLIRTransforms
)
