add_circt_dialect_library(CIRCTArcTransforms
  AddTaps.cpp
  AllocateState.cpp
  ArcCanonicalizer.cpp
  Dedup.cpp
  GroupResetsAndEnables.cpp
  InferMemories.cpp
  InferStateProperties.cpp
  InlineArcs.cpp
  InlineModules.cpp
  IsolateClocks.cpp
  LatencyRetiming.cpp
  LegalizeStateUpdate.cpp
  LowerArcsToFuncs.cpp
  LowerClocksToFuncs.cpp
  LowerLUT.cpp
  LowerState.cpp
  LowerVectorizations.cpp
  MakeTables.cpp
  MuxToControlFlow.cpp
  SimplifyVariadicOps.cpp
  SplitLoops.cpp
  StripSV.cpp

  SHARED
  
  DEPENDS
  CIRCTArcTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTArcExternalInterfaces
  CIRCTComb
  CIRCTHW
  CIRCTOM
  CIRCTSV
  CIRCTSeq
  CIRCTSupport
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
