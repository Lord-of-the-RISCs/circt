add_circt_translation_library(CIRCTExportChiselInterface
  ExportChiselInterface.cpp

  SHARED

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  MLIRPass
  MLIRSideEffectInterfaces
)
