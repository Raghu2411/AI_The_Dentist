import 'package:logger/logger.dart';

class LoggerFilter extends LogFilter {
  final bool enableLogger;

  LoggerFilter({required this.enableLogger});

  @override
  bool shouldLog(LogEvent event) => enableLogger;
}

class LoggerPrinter extends PrettyPrinter {

  LoggerPrinter();

  @override
  List<String> log(LogEvent event) {
    // Format the log message(s) as usual.
    final message = super.log(event);
    return message;
  }
}
