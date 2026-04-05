import 'package:data/src/datasource/interceptor/auth_interceptor.dart';
import 'package:data/src/datasource/interceptor/connection_interceptor.dart';
import 'package:dio/dio.dart';
import 'package:pretty_dio_logger/pretty_dio_logger.dart';

class DioHandler {
  final AuthInterceptor authInterceptor;
  final ConnectionInterceptor connectionInterceptor;
  final PrettyDioLogger prettyDioLogger;

  DioHandler({
    required this.authInterceptor,
    required this.connectionInterceptor,
    required this.prettyDioLogger,
  });

  Dio createDio() {
    final dio = Dio(
      BaseOptions(
        baseUrl: 'https://tym24-ai-the-dentist.hf.space/',
        // baseUrl: 'https://babar24134-the-dental-studio.hf.space/',
        receiveTimeout: Duration(minutes: 5),
        connectTimeout: Duration(minutes: 5),
        sendTimeout: Duration(minutes: 5),
        contentType: 'application/json',
      ),
    );

    dio.interceptors.addAll([
      authInterceptor,
      connectionInterceptor,
      prettyDioLogger,
    ]);

    return dio;
  }
}
