import 'package:flutter/material.dart';
import 'package:pdfrx/pdfrx.dart';

import '../../core/constants/app_colors.dart';
import '../../views/app_progress_indicator.dart';

class AppPdfViewer extends StatelessWidget {
  final String? file;
  final String? url;

  const AppPdfViewer({super.key, this.file, this.url})
    : assert(
        (file != null && url == null) || (file == null && url != null),
        'Either file or url must be provided, not both',
      );

  @override
  Widget build(BuildContext context) {
    final PdfViewerController controller = PdfViewerController();

    return url != null
        ? PdfViewer.uri(
            Uri.parse(url ?? ''),
            controller: controller,
            params: params,
          )
        : PdfViewer.file(file ?? '', controller: controller, params: params);
  }

  PdfViewerParams get params {
    return PdfViewerParams(
      backgroundColor: AppColors.black,
      pageDropShadow: BoxShadow(
        color: Color.fromRGBO(0, 0, 0, 0.06),
        blurRadius: 8.0,
        offset: Offset(0.0, 4.0),
      ),
      loadingBannerBuilder: (context, bytes, _) {
        return const AppProgressIndicator();
      },
      errorBannerBuilder: (context, error, stack, ref) {
        return SizedBox.shrink();
      },
    );
  }
}
