import 'package:equatable/equatable.dart';

class ApiUrls extends Equatable {
  final String caption_text;
  final String original_image;
  final String pdf_report;
  final String predicted_image;
  final String report_text;

  const ApiUrls({
    required this.caption_text,
    required this.original_image,
    required this.pdf_report,
    required this.predicted_image,
    required this.report_text,
  });

  @override
  List<Object> get props => [
    caption_text,
    original_image,
    pdf_report,
    predicted_image,
    report_text,
  ];
}
