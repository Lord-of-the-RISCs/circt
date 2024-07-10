set(LLVM_OPTIONAL_SOURCES
  DebugAnalysis.cpp
  DebugInfo.cpp
  DependenceAnalysis.cpp
  SchedulingAnalysis.cpp
  TestPasses.cpp
)

add_circt_library(CIRCTDebugAnalysis
  DebugAnalysis.cpp
  DebugInfo.cpp

  SHARED

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTDebug
  CIRCTHW
  MLIRIR
)

add_circt_library(CIRCTDependenceAnalysis
  DependenceAnalysis.cpp

  SHARED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineUtils
  MLIRTransformUtils
)

add_circt_library(CIRCTSchedulingAnalysis
  SchedulingAnalysis.cpp

  SHARED

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRIR
  CIRCTDependenceAnalysis
  CIRCTScheduling
)

add_circt_library(CIRCTAnalysisTestPasses
  TestPasses.cpp

  SHARED

  LINK_LIBS PUBLIC
  CIRCTDebugAnalysis
  CIRCTDependenceAnalysis
  CIRCTSchedulingAnalysis
  CIRCTHW
  MLIRPass
)
