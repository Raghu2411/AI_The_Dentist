//@GeneratedMicroModule;KernelPackageModule;package:kernel/src/di/kernel_module.module.dart
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// coverage:ignore-file

// ignore_for_file: no_leading_underscores_for_library_prefixes
import 'dart:async' as _i687;

import 'package:injectable/injectable.dart' as _i526;
import 'package:kernel/src/di/kernel_module.dart' as _i396;
import 'package:logger/logger.dart' as _i974;

class KernelPackageModule extends _i526.MicroPackageModule {
// initializes the registration of main-scope dependencies inside of GetIt
  @override
  _i687.FutureOr<void> init(_i526.GetItHelper gh) {
    final kernelModule = _$KernelModule();
    gh.factory<_i974.PrettyPrinter>(() => kernelModule.getLoggerPrinter());
    gh.factory<_i974.LogFilter>(() => kernelModule.getLoggerFilter());
    gh.singleton<_i974.Logger>(() => kernelModule.getLogger(
          gh<_i974.PrettyPrinter>(),
          gh<_i974.LogFilter>(),
        ));
  }
}

class _$KernelModule extends _i396.KernelModule {}
