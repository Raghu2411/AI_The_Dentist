import 'package:equatable/equatable.dart';

class ChatbotMessage extends Equatable {
  final String response;

  const ChatbotMessage({required this.response});

  @override
  List<Object> get props => [response];
}
