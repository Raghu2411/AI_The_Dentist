import 'package:json_annotation/json_annotation.dart';

part 'api_urls_dto.g.dart';

@JsonSerializable(explicitToJson: true, includeIfNull: false)
class ApiUrlsDto {
  final String? original_image;
  final String? predicted_image;
  final String? pdf_report;
  final String? caption_text;
  final String? medical_report_text;

  const ApiUrlsDto({
    this.original_image,
    this.predicted_image,
    this.pdf_report,
    this.caption_text,
    this.medical_report_text,
  });

  factory ApiUrlsDto.fromJson(Map<String, dynamic> json) =>
      _$ApiUrlsDtoFromJson(json);

  Map<String, dynamic> toJson() => _$ApiUrlsDtoToJson(this);
}
