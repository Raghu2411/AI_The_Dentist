import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:injectable/injectable.dart';

import '../../../../core/cubit/app_cubit.dart';

part 'select_image_bottom_sheet_cubit.freezed.dart';

@injectable
class SelectImageBottomSheetCubit
    extends AppCubit<SelectImageBottomSheetState> {
  SelectImageBottomSheetCubit() : super(const _Initial());
}

@freezed
sealed class SelectImageBottomSheetState with _$SelectImageBottomSheetState {
  const factory SelectImageBottomSheetState.initial() = _Initial;
}
