import 'package:flutter/foundation.dart';
import 'package:injectable/injectable.dart';
import 'package:logger/logger.dart';

import '../common/logger_filter.dart';

// required by code generation
@microPackageInit
void initKernelModule() {}

@module
abstract class KernelModule {
  ///
  /// Logger configuration
  ///

  @factoryMethod
  PrettyPrinter getLoggerPrinter() {
    return LoggerPrinter();
  }

  @factoryMethod
  LogFilter getLoggerFilter() => LoggerFilter(enableLogger: !kReleaseMode);

  @singleton
  Logger getLogger(PrettyPrinter printer, LogFilter filter) =>
      Logger(printer: printer, filter: filter);
}
