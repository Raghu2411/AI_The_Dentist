// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'api_urls_dto.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

ApiUrlsDto _$ApiUrlsDtoFromJson(Map<String, dynamic> json) => ApiUrlsDto(
  original_image: json['original_image'] as String?,
  predicted_image: json['predicted_image'] as String?,
  pdf_report: json['pdf_report'] as String?,
  caption_text: json['caption_text'] as String?,
  medical_report_text: json['medical_report_text'] as String?,
);

Map<String, dynamic> _$ApiUrlsDtoToJson(ApiUrlsDto instance) =>
    <String, dynamic>{
      'original_image': ?instance.original_image,
      'predicted_image': ?instance.predicted_image,
      'pdf_report': ?instance.pdf_report,
      'caption_text': ?instance.caption_text,
      'medical_report_text': ?instance.medical_report_text,
    };
