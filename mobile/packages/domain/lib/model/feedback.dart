import 'package:equatable/equatable.dart';

class Feedback extends Equatable {
  final String message;
  final bool success;

  const Feedback({required this.message, required this.success});

  @override
  List<Object> get props => [message, success];
}
