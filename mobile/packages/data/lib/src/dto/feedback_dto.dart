import 'package:json_annotation/json_annotation.dart';

part 'feedback_dto.g.dart';

@JsonSerializable(explicitToJson: true, includeIfNull: false)
class FeedbackDto {
  final String? message;
  final bool? success;

  const FeedbackDto({this.message, this.success});

  factory FeedbackDto.fromJson(Map<String, dynamic> json) =>
      _$FeedbackDtoFromJson(json);

  Map<String, dynamic> toJson() => _$FeedbackDtoToJson(this);
}
