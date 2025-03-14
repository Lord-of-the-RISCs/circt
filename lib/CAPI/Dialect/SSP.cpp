//===- SSP.cpp - C interface for the SSP dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/SSP.h"
#include "circt/Dialect/SSP/SSPDialect.h"
#include "circt/Dialect/SSP/SSPPasses.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SSP, ssp, circt::ssp::SSPDialect)

void registerSSPPasses() { circt::ssp::registerPasses(); }
