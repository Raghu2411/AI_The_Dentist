//@GeneratedMicroModule;DataPackageModule;package:data/src/di/data_module.module.dart
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// coverage:ignore-file

// ignore_for_file: no_leading_underscores_for_library_prefixes
import 'dart:async' as _i687;

import 'package:connectivity_plus/connectivity_plus.dart' as _i895;
import 'package:data/src/common/data_object_mapper.dart' as _i793;
import 'package:data/src/datasource/api/auth_handler.dart' as _i502;
import 'package:data/src/datasource/api/dio_handler.dart' as _i265;
import 'package:data/src/datasource/api/essex_dental_api.dart' as _i233;
import 'package:data/src/datasource/interceptor/auth_interceptor.dart' as _i54;
import 'package:data/src/datasource/interceptor/connection_interceptor.dart'
    as _i416;
import 'package:data/src/di/data_module.dart' as _i797;
import 'package:data/src/repository/essex_dental_api_repository_impl.dart'
    as _i699;
import 'package:data/src/repository/storage_repository_impl.dart' as _i510;
import 'package:dio/dio.dart' as _i361;
import 'package:domain/common/auth_handler.dart' as _i442;
import 'package:domain/domain.dart' as _i494;
import 'package:injectable/injectable.dart' as _i526;
import 'package:logger/logger.dart' as _i974;
import 'package:pretty_dio_logger/pretty_dio_logger.dart' as _i528;

class DataPackageModule extends _i526.MicroPackageModule {
// initializes the registration of main-scope dependencies inside of GetIt
  @override
  _i687.FutureOr<void> init(_i526.GetItHelper gh) {
    final dataModule = _$DataModule();
    gh.singleton<_i895.Connectivity>(() => dataModule.connectivity);
    gh.singleton<_i528.PrettyDioLogger>(() => dataModule.prettyDioLogger);
    gh.singleton<_i974.Logger>(() => dataModule.logger);
    gh.singleton<_i793.DataObjectMapper>(() => _i793.DataObjectMapper());
    gh.singleton<_i442.AuthHandler>(() => _i502.AuthHandlerImpl());
    gh.singleton<_i494.StorageRepository>(() => _i510.StorageRepositoryImpl());
    gh.singleton<_i416.ConnectionInterceptor>(
        () => _i416.ConnectionInterceptor(gh<_i895.Connectivity>()));
    gh.singleton<_i54.AuthInterceptor>(() => _i54.AuthInterceptor(
          gh<_i494.StorageRepository>(),
          gh<_i442.AuthHandler>(),
        ));
    gh.singleton<_i265.DioHandler>(() => dataModule.dioHandler(
          gh<_i54.AuthInterceptor>(),
          gh<_i416.ConnectionInterceptor>(),
          gh<_i528.PrettyDioLogger>(),
        ));
    gh.lazySingleton<_i361.Dio>(() => dataModule.dio(gh<_i265.DioHandler>()));
    gh.singleton<_i233.EssexDentalApi>(
        () => _i233.EssexDentalApi(gh<_i361.Dio>()));
    gh.singleton<_i494.EssexDentalApiRepository>(
        () => _i699.EssexDentalApiRepositoryImpl(
              essexDentalApi: gh<_i233.EssexDentalApi>(),
              objectMapper: gh<_i793.DataObjectMapper>(),
              logger: gh<_i974.Logger>(),
            ));
  }
}

class _$DataModule extends _i797.DataModule {}
