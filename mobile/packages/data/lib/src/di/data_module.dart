import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:data/src/datasource/api/dio_handler.dart';
import 'package:data/src/datasource/interceptor/auth_interceptor.dart';
import 'package:data/src/datasource/interceptor/connection_interceptor.dart';
import 'package:dio/dio.dart';
import 'package:injectable/injectable.dart';
import 'package:logger/logger.dart';
import 'package:pretty_dio_logger/pretty_dio_logger.dart';

@microPackageInit
void initDataModule() {}

@module
abstract class DataModule {
  @singleton
  Connectivity get connectivity => Connectivity();

  @singleton
  PrettyDioLogger get prettyDioLogger =>
      PrettyDioLogger(requestBody: true, responseBody: true);

  @singleton
  DioHandler dioHandler(
    AuthInterceptor authInterceptor,
    ConnectionInterceptor connectionInterceptor,
    PrettyDioLogger prettyDioLogger,
  ) {
    return DioHandler(
      authInterceptor: authInterceptor,
      connectionInterceptor: connectionInterceptor,
      prettyDioLogger: prettyDioLogger,
    );
  }

  @lazySingleton
  Dio dio(DioHandler dioHandler) => dioHandler.createDio();

  @singleton
  Logger get logger => Logger();
}
