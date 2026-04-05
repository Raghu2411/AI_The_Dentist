import 'package:equatable/equatable.dart';
import 'package:injectable/injectable.dart';

@injectable
class ChatHistory extends Equatable {
  final List<Map<String, String>> history;

  const ChatHistory({required this.history});

  @override
  List<Object> get props => [history];
}
