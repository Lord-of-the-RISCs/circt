add_circt_library(CIRCTOMEvaluator
  Evaluator.cpp

  SHARED

  DEPENDS
  MLIROMIncGen
  MLIROMAttrIncGen
  MLIROMOpInterfacesIncGen

  LINK_LIBS
  CIRCTOM
  MLIRIR
)
