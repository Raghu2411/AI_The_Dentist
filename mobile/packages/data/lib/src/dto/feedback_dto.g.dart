// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'feedback_dto.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

FeedbackDto _$FeedbackDtoFromJson(Map<String, dynamic> json) => FeedbackDto(
  message: json['message'] as String?,
  success: json['success'] as bool?,
);

Map<String, dynamic> _$FeedbackDtoToJson(FeedbackDto instance) =>
    <String, dynamic>{
      'message': ?instance.message,
      'success': ?instance.success,
    };
