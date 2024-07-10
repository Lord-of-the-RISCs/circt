add_circt_dialect_library(CIRCTArcExternalInterfaces
  RuntimeCostEstimateInterfaceImpl.cpp

  SHARED
  
  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  MLIRIR
  MLIRSCFDialect
)
