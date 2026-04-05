import 'package:json_annotation/json_annotation.dart';

part 'chatbot_dto.g.dart';

@JsonSerializable(explicitToJson: true, includeIfNull: false)
class ChatbotDto {
  final String? response;

  const ChatbotDto({this.response});

  factory ChatbotDto.fromJson(Map<String, dynamic> json) =>
      _$ChatbotDtoFromJson(json);

  Map<String, dynamic> toJson() => _$ChatbotDtoToJson(this);
}
