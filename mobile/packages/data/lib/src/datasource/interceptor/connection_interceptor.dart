import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:dio/dio.dart';
import 'package:injectable/injectable.dart';

@singleton
class ConnectionInterceptor extends Interceptor {
  final Connectivity _connectivity;

  ConnectionInterceptor(this._connectivity);

  @override
  Future<void> onRequest(
    RequestOptions options,
    RequestInterceptorHandler handler,
  ) async {
    var connectivityResult = await _connectivity.checkConnectivity();
    if (connectivityResult.contains(ConnectivityResult.none)) {
      return handler.reject(
        DioException(
          error: 'No internet connection.',
          requestOptions: options,
          type: DioExceptionType.connectionTimeout,
        ),
      );
    }

    return super.onRequest(options, handler);
  }
}
